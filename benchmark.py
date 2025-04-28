# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import datetime
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import logging

import torch
import transformers
from tqdm import tqdm
from data import EvaluationExample
import csv

from torchmetrics.text import BLEUScore, ROUGEScore, EditDistance
# TODO: create ExactMatch torchmetrics.text

from torcheval.metrics.aggregation.mean import Mean
from torcheval.metrics.metric import Metric

from data import get_data, LowercaseProcessingFunction, get_valid_dataset_formats, EvaluationExample
from generate import load_model_and_tokenizer, setup
from utils import ROUGEScoreWrapper

import arguments
from arguments import Arguments, simple_parse_args_string
from self_speculation.autoregressive_generator import AutoRegressiveGenerationStrategy
from self_speculation.self_speculation_generator import SelfSpeculativeGenerationStrategy
from self_speculation.layer_drop_generator import LayerDropGenerationStrategy
from self_speculation.generator_base import (
    GenerationConfig,
    GenerationResult,
    GenerationStrategy,
    HuggingfaceLlamaGenerator,
)

from self_speculation.depth_adaptive_token_generator import DepthAdaptiveTokenGenerationStrategy
from data import extract_answer_from_gsm8k
from data import extract_answer_from_math
from self_speculation.depth_adaptive_sequence_generator import DepthAdaptiveSequenceGenerationStrategy

log = logging.getLogger(__name__)

class LlamaFLOPsCounter:
    """Counts FLOPs for Llama model operations during generation."""
    
    def __init__(self, model: transformers.LlamaForCausalLM):
        self.model = model
        self.reset_counters()
        self.hooks = []
        
        # Model dimensions for FLOP calculations
        self.hidden_size = model.config.hidden_size
        self.intermediate_size = model.config.intermediate_size
        self.num_attention_heads = model.config.num_attention_heads
        self.head_dim = model.config.hidden_size // model.config.num_attention_heads
        self.num_key_value_heads = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
        self.vocab_size = model.config.vocab_size
        self.model_type = model.config.model_type
        
        # Cache for sequence lengths to avoid repeated computation
        self.seq_lens_cache = {}
        
    def reset_counters(self):
        """Reset all FLOP counters."""
        self.total_flops = 0
        self.layer_flops = {}
        self.operation_flops = {
            "attention": 0,
            "mlp": 0,
            "layernorm": 0,
            "embedding": 0,
            "lm_head": 0,
            "residual": 0
        }
        self.layer_counts = {}
        
    def register_hooks(self):
        """Register hooks on model components to count FLOPs."""
        self.reset_counters()
        
        # Remove any existing hooks
        self.remove_hooks()
        
        # Initialize layer counts for all layers upfront
        for i in range(len(self.model.model.layers)):
            self.layer_counts[i] = 0
            self.layer_flops[i] = 0
            
        # Define a more robust hook function factory
        def make_linear_hook(layer_idx, component_name, op_type):
            def hook(module, inputs, outputs):
                if not isinstance(inputs, tuple):
                    inputs = (inputs,)
                input_tensor = inputs[0]
                self._count_flops_linear(module, input_tensor, outputs, f"layer_{layer_idx}_{component_name}", op_type)
            return hook
            
        def make_embedding_hook():
            def hook(module, inputs, outputs):
                if not isinstance(inputs, tuple):
                    inputs = (inputs,)
                input_tensor = inputs[0]
                self._count_flops_embedding(module, input_tensor, outputs)
            return hook
            
        # Register hooks for all transformer layers
        for layer_idx, layer in enumerate(self.model.model.layers):
            # Attention hooks
            self.hooks.append(layer.self_attn.q_proj.register_forward_hook(
                make_linear_hook(layer_idx, "q_proj", "attention")
            ))
            self.hooks.append(layer.self_attn.k_proj.register_forward_hook(
                make_linear_hook(layer_idx, "k_proj", "attention")
            ))
            self.hooks.append(layer.self_attn.v_proj.register_forward_hook(
                make_linear_hook(layer_idx, "v_proj", "attention")
            ))
            self.hooks.append(layer.self_attn.o_proj.register_forward_hook(
                make_linear_hook(layer_idx, "o_proj", "attention")
            ))
            
            # MLP hooks
            self.hooks.append(layer.mlp.gate_proj.register_forward_hook(
                make_linear_hook(layer_idx, "gate_proj", "mlp")
            ))
            self.hooks.append(layer.mlp.up_proj.register_forward_hook(
                make_linear_hook(layer_idx, "up_proj", "mlp")
            ))
            self.hooks.append(layer.mlp.down_proj.register_forward_hook(
                make_linear_hook(layer_idx, "down_proj", "mlp")
            ))
            
        # Hooks for embedding and LM head
        self.hooks.append(self.model.model.embed_tokens.register_forward_hook(
            make_embedding_hook()
        ))
        self.hooks.append(self.model.lm_head.register_forward_hook(
            make_linear_hook(-1, "lm_head", "lm_head")
        ))
        
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def _count_flops_linear(self, module, x, output, name, op_type):
        """Count FLOPs for linear layer (matmul + bias)."""
        try:
            if x.dim() == 2:
                # Forward pass: [batch_size, in_features] x [in_features, out_features]
                batch_size, in_features = x.shape
                out_features = output.shape[1]
                # 2 FLOPs per multiplication-addition
                flops = batch_size * in_features * out_features * 2
            else:
                # Sequence input: [batch_size, seq_len, in_features]
                batch_size, seq_len, in_features = x.shape
                out_features = output.shape[2]
                # 2 FLOPs per multiplication-addition
                flops = batch_size * seq_len * in_features * out_features * 2
            
            # Add to counters
            self.total_flops += flops
            self.operation_flops[op_type] += flops
            
            # Extract layer number from name (if applicable)
            if "layer_" in name:
                try:
                    layer_idx = int(name.split("_")[1])
                    # Make sure layer_idx exists in our dictionaries
                    if layer_idx not in self.layer_flops:
                        self.layer_flops[layer_idx] = 0
                    if layer_idx not in self.layer_counts:
                        self.layer_counts[layer_idx] = 0
                    
                    # Update counters
                    self.layer_flops[layer_idx] += flops
                    self.layer_counts[layer_idx] += 1
                except (IndexError, ValueError, KeyError) as e:
                    # Just log the error and continue
                    print(f"Warning: Error processing layer idx from {name}: {e}")
        except Exception as e:
            # Catch any errors to prevent hooks from crashing the model
            print(f"Error in _count_flops_linear: {e} for {name}")
            import traceback
            traceback.print_exc()
    
    def _count_flops_embedding(self, module, x, output):
        """Count FLOPs for embedding lookup."""
        try:
            # Embedding lookup is essentially free in terms of FLOPs
            # but we'll count it for completeness
            batch_size = x.shape[0]
            seq_len = x.shape[1]
            embed_dim = output.shape[2]
            
            # Only counting the actual embedding lookup
            flops = batch_size * seq_len * embed_dim
            
            self.total_flops += flops
            self.operation_flops["embedding"] += flops
        except Exception as e:
            # Catch any errors to prevent hooks from crashing the model
            print(f"Error in _count_flops_embedding: {e}")
            import traceback
            traceback.print_exc()
    
    def track_layer_usage(self, layer_idx):
        """Record that a particular layer was used during generation."""
        if layer_idx in self.layer_counts:
            self.layer_counts[layer_idx] += 1
    
    def get_summary(self):
        """Return a summary of FLOP usage."""
        return {
            "total_flops": self.total_flops,
            "operation_flops": self.operation_flops,
            "layer_flops": self.layer_flops,
            "layer_counts": self.layer_counts
        }
    
    def estimate_early_exit_savings(self, exit_layer):
        """Estimate FLOPs saved by early exiting at a given layer."""
        if not self.layer_flops:
            return 0
        
        total_layer_flops = sum(self.layer_flops.values())
        saved_flops = sum(self.layer_flops.get(i, 0) for i in range(exit_layer, len(self.model.model.layers)))
        
        return {
            "total_layer_flops": total_layer_flops,
            "saved_flops": saved_flops,
            "saving_percentage": (saved_flops / total_layer_flops) * 100 if total_layer_flops > 0 else 0
        }

    def humanize_flops(self, flops):
        """Convert raw FLOP count to human-readable format."""
        if flops < 1e9:
            return f"{flops / 1e6:.2f} MFLOPs"
        elif flops < 1e12:
            return f"{flops / 1e9:.2f} GFLOPs"
        else:
            return f"{flops / 1e12:.2f} TFLOPs"

@dataclass
class BenchmarkArguments:
    dataset: str
    data_path: Optional[str] = None
    random_shuffle: bool = True
    num_samples: Optional[int] = None
    n_shot: Optional[int] = 0
    template: Optional[str] = None

# @dataclass
# class EvaluationExample:
#     input: str
#     output: str

class ExactMatch(Metric):
    def __init__(self, device: Optional[torch.device] = None) -> None:
        super().__init__(device=device)
        self.correct_count = 0
        self.total_count = 0

    def update(self, prediction: str, target: str) -> None:
        self.total_count += 1
        if prediction.strip() == target.strip():  # remove trailing spaces
            self.correct_count += 1

    def compute(self) -> torch.Tensor:
        if self.total_count == 0:
            return torch.tensor(0.0, device=self.device)  # Handle the case where no updates were made
        return torch.tensor(self.correct_count / self.total_count, device=self.device)

    def merge_state(self, metrics: "ExactMatch") -> None:
        """
        Merges the state from another ExactMatch metric instance.
        This is crucial for distributed training/evaluation.
        """
        self.correct_count += metrics.correct_count
        self.total_count += metrics.total_count


@dataclass
class EvaluationMetrics:
    predicted_text: Dict[str, Metric]
    acceptance_rate: Dict[str, Metric]
    total_time: Dict[str, Metric]
    time_per_token: Dict[str, Metric]
    tokens_per_second: Dict[str, Metric]
    total_flops: Dict[str, Metric]  # New field for total FLOPs
    flops_per_token: Dict[str, Metric]  # New field for FLOPs per token
    
    def update(
    self,
    evaluation_example: Dict[str, str], 
    generation_result: GenerationResult,
    flops_data: Optional[Dict] = None,
    ) -> None:
        try:
            prediction = generation_result.decoded_prediction
            target = evaluation_example["output"]
    
            # For multiple choice datasets (MMLU, RACE)
            if 'accuracy' in self.predicted_text and 'exact_match' in self.predicted_text:
                # Extract the answer letter from prediction for multiple choice
                predicted_letter = None
                if prediction.strip().startswith(('A', 'B', 'C', 'D')):
                    predicted_letter = prediction.strip()[0]  # Just take the first character
                else:
                    # Try to find first occurrence of A. B. C. or D.
                    for letter in ['A', 'B', 'C', 'D']:
                        pattern = letter + '.'
                        if pattern in prediction:
                            predicted_letter = letter
                            break
    
                # Extract the answer letter from target
                target_letter = target.strip()
                if len(target_letter) > 0:
                    target_letter = target_letter[0]  # Just take the first character
                
                print(f"Extracted prediction: {predicted_letter}, Extracted target: {target_letter}")
                
                # Calculate correctness
                is_correct = 0.0
                if predicted_letter and target_letter and predicted_letter == target_letter:
                    is_correct = 1.0
                    print("[CORRECT]")
                else:
                    print("[INCORRECT]")
                
                # Update metrics
                for metric_name, metric in self.predicted_text.items():
                    if metric_name == "exact_match":
                        metric.update(prediction, target)
                    elif metric_name == "accuracy":
                        metric.update(torch.tensor(is_correct))
            else:
                # For other datasets, directly update metrics with the prediction and target
                for metric_name, metric in self.predicted_text.items():
                    metric.update(prediction, target)
    
            # Rest of the updates for other metrics
            for metric in self.acceptance_rate.values():
                if generation_result.generation_strategy_result.acceptance_rate is None:
                    acceptance_rate = torch.tensor(0.0)
                else:
                    acceptance_rate = torch.tensor(
                        generation_result.generation_strategy_result.acceptance_rate
                    )
                metric.update(acceptance_rate)
    
            for metric in self.total_time.values():
                metric.update(torch.tensor(generation_result.total_time))
    
            for metric in self.time_per_token.values():
                metric.update(torch.tensor(generation_result.time_per_token))
    
            for metric in self.tokens_per_second.values():
                metric.update(torch.tensor(generation_result.tokens_per_second))
            
            # Update FLOPs metrics if data is provided
            if flops_data:
                total_flops = flops_data.get("total_flops", 0)
                tokens_generated = generation_result.num_tokens_generated
                
                for metric in self.total_flops.values():
                    metric.update(torch.tensor(float(total_flops)))
                    
                for metric in self.flops_per_token.values():
                    flops_per_token = total_flops / tokens_generated if tokens_generated > 0 else 0
                    metric.update(torch.tensor(float(flops_per_token)))
        except Exception as e:
            print(f"Error in EvaluationMetrics.update: {e}")
            import traceback
            traceback.print_exc()

    def compute(self) -> Dict[str, torch.Tensor]:
        result = {
            "predicted_text": {
                metric_name: metric.compute().item()
                for metric_name, metric in self.predicted_text.items()
            },
            "acceptance_rate": {
                metric_name: metric.compute().item()
                for metric_name, metric in self.acceptance_rate.items()
            },
            "total_time": {
                metric_name: metric.compute().item()
                for metric_name, metric in self.total_time.items()
            },
            "time_per_token": {
                metric_name: metric.compute().item()
                for metric_name, metric in self.time_per_token.items()
            },
            "tokens_per_second": {
                metric_name: metric.compute().item()
                for metric_name, metric in self.tokens_per_second.items()
            },
        }
        
        # Add FLOPs metrics if available
        if self.total_flops:
            result["total_flops"] = {
                metric_name: metric.compute().item()
                for metric_name, metric in self.total_flops.items()
            }
            
        if self.flops_per_token:
            result["flops_per_token"] = {
                metric_name: metric.compute().item()
                for metric_name, metric in self.flops_per_token.items()
            }
            
        return result

    @classmethod
    def build_metrics(cls) -> "EvaluationMetrics":
        return cls(
            predicted_text={
                "exact_match": ExactMatch(),
                "accuracy": Mean(), # example accuracy. can be replaced with ExactMatch
            },
            acceptance_rate={"mean": Mean()},
            total_time={"mean": Mean()},
            time_per_token={"mean": Mean()},
            tokens_per_second={"mean": Mean()},
            total_flops={"mean": Mean()},  # Add FLOPs metrics
            flops_per_token={"mean": Mean()},
        )

def setup_math_reasoning_metrics(dataset_name):
    """Set up metrics for math reasoning datasets"""
    class MathAccuracy(Mean):
        def __init__(self, dataset_name, device: Optional[torch.device] = None) -> None:
            super().__init__(device=device)
            self.dataset_name = dataset_name
            
        def update(self, value: torch.Tensor) -> None:
            # This is already a tensor with value 0.0 or 1.0
            super().update(value)
    
    return EvaluationMetrics(
        predicted_text={
            "accuracy": MathAccuracy(dataset_name),
        },
        acceptance_rate={"mean": Mean()},
        total_time={"mean": Mean()},
        time_per_token={"mean": Mean()},
        tokens_per_second={"mean": Mean()},
        total_flops={"mean": Mean()},  # Add FLOPs metrics
        flops_per_token={"mean": Mean()},
    )


def normalize_math_answer(answer_str):
    """
    Normalize a math answer for better comparison.
    
    Handles:
    - Numbers with commas
    - Dollar signs
    - Spaces
    - Case sensitivity for text answers
    """
    if not answer_str:
        return ""
    
    import re
    
    # Convert to lowercase and strip whitespace
    normalized = answer_str.lower().strip()
    
    # Remove commas from numbers (e.g., "1,234" -> "1234")
    normalized = re.sub(r"(\d),(\d)", r"\1\2", normalized)
    
    # Remove dollar signs
    normalized = normalized.replace("$", "")
    
    # Remove extra whitespace
    normalized = re.sub(r"\s+", " ", normalized)
    
    # Remove trailing zeros after decimal point (e.g., "5.0" -> "5")
    if re.match(r"^\d+\.\d+$", normalized):
        normalized = normalized.rstrip('0').rstrip('.') if '.' in normalized else normalized
    
    # Normalize fractions like "1/2" (ensure spaces around division)
    normalized = re.sub(r"(\d+)/(\d+)", r"\1 / \2", normalized)
    
    return normalized


def benchmark(
        model: torch.nn.Module,
        tokenizer: transformers.PreTrainedTokenizerBase,
        benchmark_arguments: BenchmarkArguments,
        generation_config: GenerationConfig,
        seed = None,
        measure_flops: bool = True,  # New parameter to control FLOPs measurement
    ):
    """
    Benchmark function that handles various dataset types.
    For MBPP and HUMAN_EVAL, it only generates code and saves to CSV without computing metrics.
    
    New parameter:
    - measure_flops: If True, FLOPs will be measured during generation
    """
    # Check dataset type
    is_multiple_choice = benchmark_arguments.dataset in ["mmlu", "race_m", "race_h"]
    is_code_generation = benchmark_arguments.dataset in ["mbpp", "human_eval"]
    is_math_reasoning = benchmark_arguments.dataset in ["gsm8k", "math"]
    
    if not is_multiple_choice and not is_code_generation and not is_math_reasoning:
        print(f"Using standard benchmark for dataset: {benchmark_arguments.dataset}")
        # Call the original benchmark function for other datasets
        return original_benchmark(model, tokenizer, benchmark_arguments, generation_config, seed, measure_flops)
    
    # Configure for different dataset types
    if is_multiple_choice:
        print(f"Optimizing generation config for multiple-choice dataset: {benchmark_arguments.dataset}")
        # Limit token generation for multiple choice
        generation_config.max_steps = min(generation_config.max_steps, 20)
        # Lower temperature for more deterministic responses
        generation_config.temperature = min(generation_config.temperature, 0.3)
    
    elif is_code_generation:
        print(f"Optimizing generation config for code generation: {benchmark_arguments.dataset}")
        # Increase max tokens for code generation
        generation_config.max_steps = max(generation_config.max_steps, 512)
        # Set a moderate temperature for code
        generation_config.temperature = min(generation_config.temperature, 0.7)
        
    elif is_math_reasoning:
        print(f"Optimizing generation config for math reasoning: {benchmark_arguments.dataset}")
        # Set appropriate tokens for math reasoning (need space for step-by-step)
        generation_config.max_steps = max(generation_config.max_steps, 300)
        # Lower temperature for math to get more precise calculations
        generation_config.temperature = min(generation_config.temperature, 0.3)
    
    print(f"Updated generation config: max_steps={generation_config.max_steps}, "
          f"temperature={generation_config.temperature}")

    if generation_config.generation_strategy == "autoregressive":
        generation_strategy: GenerationStrategy = AutoRegressiveGenerationStrategy()
    elif generation_config.generation_strategy == "self_speculative":
        generation_strategy: GenerationStrategy = SelfSpeculativeGenerationStrategy()
    elif generation_config.generation_strategy == "layerdrop":
        generation_strategy: GenerationStrategy = LayerDropGenerationStrategy(
            dropout_rate=generation_config.dropout_rate,
            seed=generation_config.layerdrop_seed or seed
        )
    elif generation_config.generation_strategy == "depth_adaptive_token":
        generation_strategy: GenerationStrategy = DepthAdaptiveTokenGenerationStrategy(
            halting_threshold=generation_config.halting_threshold,
            min_layers=generation_config.min_layers,
            max_layers=generation_config.max_layers,
        )
    elif generation_config.generation_strategy == "depth_adaptive_sequence":
        generation_strategy: GenerationStrategy = DepthAdaptiveSequenceGenerationStrategy(
            halting_threshold=generation_config.halting_threshold,
            min_layers=generation_config.min_layers,
            max_layers=generation_config.max_layers,
        )
    else:
        raise ValueError(
            f"Unrecognized generation strategy: {generation_config.generation_strategy}"
        )

    # Set up appropriate metrics based on dataset type (only for datasets where we compute metrics)
    if is_multiple_choice:
        evaluation_metrics = setup_multiple_choice_metrics()
    elif is_math_reasoning:
        evaluation_metrics = setup_math_reasoning_metrics(benchmark_arguments.dataset)
    else:
        # For code generation (MBPP, HUMAN_EVAL), we just use placeholder metrics
        evaluation_metrics = EvaluationMetrics(
            predicted_text={"accuracy": Mean()},  # Just a placeholder
            acceptance_rate={"mean": Mean()},
            total_time={"mean": Mean()},
            time_per_token={"mean": Mean()},
            tokens_per_second={"mean": Mean()},
            total_flops={"mean": Mean()},
            flops_per_token={"mean": Mean()},
        )
    
    # Initialize generator
    generator = HuggingfaceLlamaGenerator(
        tokenizer=tokenizer, model=model, generation_strategy=generation_strategy
    )
    
    # Initialize FLOPs counter if measurement is enabled
    flops_counter = None
    if measure_flops:
        print("Initializing FLOPs counter")
        try:
            flops_counter = LlamaFLOPsCounter(model)
            flops_counter.register_hooks()
            print(f"FLOPs counter registered with {len(flops_counter.hooks)} hooks")
        except Exception as e:
            print(f"Error initializing FLOPs counter: {str(e)}")
            import traceback
            traceback.print_exc()
            measure_flops = False

    evaluation_data_points = get_data(
        random_shuffle=benchmark_arguments.random_shuffle,
        num_samples=benchmark_arguments.num_samples,
        dataset=benchmark_arguments.dataset,
        data_path=benchmark_arguments.data_path,
        n_shot=benchmark_arguments.n_shot,
        seed=seed,
        template=benchmark_arguments.template,
    )

    print(f"Benchmarking on {benchmark_arguments.dataset.upper()} with {len(evaluation_data_points)} samples...")
    print(f"FLOPs measurement is {'enabled' if measure_flops else 'disabled'}")
    
    # Create a list to store all results for CSV export
    all_results = []
    total_questions = 0
    total_flops_by_strategy = {
        "strategy": generation_config.generation_strategy,
        "config": generation_config.__dict__,
        "total_flops": 0,
        "total_tokens": 0,
        "flops_per_token": 0,
        "samples": []
    }

    for idx, data_point in enumerate(tqdm(evaluation_data_points, desc=f"Benchmarking {benchmark_arguments.dataset.upper()}")):
        if not hasattr(data_point, 'input') or not hasattr(data_point, 'output'):
            print(f"WARNING: Unexpected data point format: {data_point}")
            continue
            
        input_text = data_point.input
        expected_output = data_point.output

        # Reset FLOPs counter before generation
        if measure_flops and flops_counter:
            flops_counter.reset_counters()
            
        # Generate response
        try:
            generation_result = generator.generate(
                prompt=input_text,
                generation_config=generation_config,
            )
            
            predicted_answer = generation_result.decoded_prediction.strip()
            generation_success = True
            
            # Collect FLOPs data after successful generation
            flops_data = None
            if measure_flops and flops_counter:
                flops_data = flops_counter.get_summary()
                
                # Calculate per-token metrics
                tokens_generated = generation_result.num_tokens_generated
                total_flops = flops_data["total_flops"]
                flops_per_token = total_flops / tokens_generated if tokens_generated > 0 else 0
                
                # Add to total FLOPs statistics
                total_flops_by_strategy["total_flops"] += total_flops
                total_flops_by_strategy["total_tokens"] += tokens_generated
                
                # Record sample results
                sample_result = {
                    "prompt": input_text[:100] + "..." if len(input_text) > 100 else input_text,
                    "tokens_generated": tokens_generated,
                    "total_flops": total_flops,
                    "flops_per_token": flops_per_token,
                }
                total_flops_by_strategy["samples"].append(sample_result)
                
                print(f"FLOPs for sample {idx}: {flops_counter.humanize_flops(total_flops)}, "
                      f"Per token: {flops_counter.humanize_flops(flops_per_token)}")
                
        except Exception as e:
            print(f"Error during generation: {e}")
            import traceback
            traceback.print_exc()
            predicted_answer = "ERROR: Generation failed"
            generation_success = False
            flops_data = None
        
        # Update total questions count
        total_questions += 1
        
        # For code generation datasets (MBPP, HUMAN_EVAL), just log and save results without computing metrics
        if is_code_generation:
            # Extract task ID if present in the prompt (or use index)
            task_id = idx
            try:
                # If the data format includes task ID info, extract it
                import re
                if benchmark_arguments.dataset == "human_eval":
                    # For HumanEval, try to extract the function name as an identifier
                    match = re.search(r"def\s+(\w+)\s*\(", input_text)
                    if match:
                        task_id = match.group(1)
                else:  # MBPP
                    match = re.search(r"task\s*id[:\s]+(\d+)", input_text.lower())
                    if match:
                        task_id = int(match.group(1))
            except:
                pass
                
            # Store result for CSV export
            result_entry = {
                "idx": idx,
                "task_id": task_id,
                "prompt": input_text,
                "expected_code": expected_output,
                "generated_code": predicted_answer,
                "success": generation_success
            }
            
            # Add generation metrics if available
            if generation_success:
                result_entry.update({
                    "acceptance_rate": generation_result.generation_strategy_result.acceptance_rate or 0,
                    "total_time": generation_result.total_time,
                    "tokens_per_second": generation_result.tokens_per_second,
                    "num_tokens": generation_result.num_tokens_generated
                })
                
            all_results.append(result_entry)
            
            # Print truncated results
            print(f"Task ID: {task_id}")
            print(f"Prompt (truncated): {input_text[:200]}...")
            print(f"Generated Code (truncated): {predicted_answer[:200]}...")
            print("-" * 50)
            
            # No metrics calculation for code generation
            continue
        
        # For math reasoning, extract answers for evaluation
        elif is_math_reasoning:
            # Log the full inputs/outputs
            print(f"Question: {input_text[:300]}...")
            print(f"Model Response: {predicted_answer[:300]}...")
            
            # Extract answers from reference and prediction
            if benchmark_arguments.dataset == "gsm8k":
                from data import extract_answer_from_gsm8k
                expected_answer = extract_answer_from_gsm8k(expected_output)
                pred_answer = extract_answer_from_gsm8k(predicted_answer)
            else:  # MATH dataset
                from data import extract_answer_from_math
                expected_answer = extract_answer_from_math(expected_output)
                pred_answer = extract_answer_from_math(predicted_answer)
            
            print(f"Extracted expected answer: {expected_answer}")
            print(f"Extracted predicted answer: {pred_answer}")
            
            # Check if answer is correct
            is_correct = False
            if pred_answer and expected_answer:
                # Normalize and compare
                norm_pred = normalize_math_answer(pred_answer)
                norm_expected = normalize_math_answer(expected_answer)
                is_correct = (norm_pred == norm_expected)
                
                if is_correct:
                    print("[CORRECT]")
                else:
                    print("[INCORRECT]")
                    print(f"Normalized expected: {norm_expected}")
                    print(f"Normalized predicted: {norm_pred}")
            
            # Store result for CSV export
            result_entry = {
                "idx": idx,
                "question": input_text,
                "full_expected_solution": expected_output,
                "extracted_expected_answer": expected_answer,
                "full_generated_solution": predicted_answer,
                "extracted_predicted_answer": pred_answer,
                "is_correct": is_correct if (pred_answer and expected_answer) else "Unknown"
            }
            
            # Add generation metrics if available
            if generation_success:
                result_entry.update({
                    "acceptance_rate": generation_result.generation_strategy_result.acceptance_rate or 0,
                    "total_time": generation_result.total_time,
                    "tokens_per_second": generation_result.tokens_per_second,
                    "num_tokens": generation_result.num_tokens_generated
                })
            
            all_results.append(result_entry)
            
            # Update metrics for math reasoning
            if generation_success:
                for metric_name, metric in evaluation_metrics.predicted_text.items():
                    if metric_name == "accuracy":
                        # For accuracy, we pass 1.0 if correct, 0.0 if incorrect
                        accuracy_value = 1.0 if is_correct else 0.0
                        metric.update(torch.tensor(accuracy_value))
            
        # Update appropriate metrics based on dataset type
        elif is_multiple_choice:
            for metric_name, metric in evaluation_metrics.predicted_text.items():
                metric.update(predicted_answer, expected_output)
            
        # Common metrics for all dataset types with successful generation
        if generation_success:
            example_dict = {"input": input_text, "output": expected_output}
            evaluation_metrics.update(example_dict, generation_result, flops_data)
        
        print("-" * 50)

    # Save results to CSV if it's a code generation or math reasoning task
    if is_code_generation or is_math_reasoning:
        csv_path = save_results_to_csv(all_results, benchmark_arguments.dataset)
        print(f"Results saved to {csv_path}")
        
        if is_code_generation:
            # For code generation, just return the path to CSV
            return {"predicted_text": {"saved_to_csv": csv_path}}
    
    # For other datasets, compute and return metrics as usual
    final_metrics = evaluation_metrics.compute()
    
    # Calculate final FLOPs statistics if available
    if measure_flops and total_flops_by_strategy["total_tokens"] > 0:
        total_flops_by_strategy["flops_per_token"] = (
            total_flops_by_strategy["total_flops"] / total_flops_by_strategy["total_tokens"]
        )
        
        # Add timestamp
        total_flops_by_strategy["timestamp"] = datetime.datetime.now().isoformat()
        
        # Save FLOPs results
        os.makedirs("./flops_data", exist_ok=True)
        flops_output_file = f"./flops_data/flops_{benchmark_arguments.dataset}_{generation_config.generation_strategy}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(flops_output_file, "w") as f:
            json.dump(total_flops_by_strategy, f, indent=2)
        
        print(f"FLOPs measurements saved to {flops_output_file}")
        print(f"Average FLOPs per token: {flops_counter.humanize_flops(total_flops_by_strategy['flops_per_token'])}")
        
        # Add FLOPs to final metrics
        if "total_flops" not in final_metrics:
            final_metrics["total_flops"] = {}
        if "flops_per_token" not in final_metrics:
            final_metrics["flops_per_token"] = {}
            
        final_metrics["total_flops"]["total"] = total_flops_by_strategy["total_flops"]
        final_metrics["flops_per_token"]["average"] = total_flops_by_strategy["flops_per_token"]
    
    print(f"\n--- Final Metrics ({benchmark_arguments.dataset.upper()}) ---")
    for metric_name, values in final_metrics.items():
        if isinstance(values, dict):
            for sub_name, value in values.items():
                print(f"{metric_name}.{sub_name}: {value:.4f}")
        else:
            print(f"{metric_name}: {values:.4f}")
    print(f"Total Questions: {total_questions}")
    
    # Clean up FLOPs counter if it was used
    if measure_flops and flops_counter:
        flops_counter.remove_hooks()

    return final_metrics

def setup_multiple_choice_metrics():
    """Set up metrics for multiple-choice QA datasets"""
    class MultipleChoiceExactMatch(Metric):
        def __init__(self, device: Optional[torch.device] = None) -> None:
            super().__init__(device=device)
            self.correct_count = 0
            self.total_count = 0

        def update(self, prediction, target=None) -> None:
            # Increment total count
            self.total_count += 1
            
            # Handle pre-computed case
            if target is None and isinstance(prediction, torch.Tensor):
                # Just use the given correctness value
                if prediction.item() > 0.5:  # Assuming binary accuracy
                    self.correct_count += 1
                return
            
            # Extract answer letter from prediction
            pred_letter = None
            if isinstance(prediction, str):
                if prediction.strip().startswith(('A', 'B', 'C', 'D')):
                    pred_letter = prediction.strip()[0]
                else:
                    # Try to find first occurrence of A. B. C. or D.
                    for letter in ['A', 'B', 'C', 'D']:
                        pattern = letter + '.'
                        if pattern in prediction:
                            pred_letter = letter
                            break
            
            # Extract answer letter from target
            target_letter = None
            if isinstance(target, str):
                target_letter = target.strip()
                if len(target_letter) > 0:
                    target_letter = target_letter[0]
            
            print(f"Extracted: prediction={pred_letter}, target={target_letter}")
            
            if pred_letter and target_letter and pred_letter == target_letter:
                self.correct_count += 1
                print("[CORRECT]")
            else:
                print("[INCORRECT]")

        def compute(self) -> torch.Tensor:
            if self.total_count == 0:
                return torch.tensor(0.0, device=self.device)
            return torch.tensor(self.correct_count / self.total_count, device=self.device)

        def merge_state(self, metrics: "MultipleChoiceExactMatch") -> None:
            self.correct_count += metrics.correct_count
            self.total_count += metrics.total_count
    
    class MultipleChoiceAccuracy(Mean):
        def update(self, prediction, target=None) -> None:
            # Handle the case where a pre-computed value is passed
            if target is None:
                # Prediction is already a correctness value
                super().update(prediction)
                return
            
            # Otherwise, extract letters and compute correctness
            # Extract answer letter from prediction
            pred_letter = None
            if isinstance(prediction, str):
                if prediction.strip().startswith(('A', 'B', 'C', 'D')):
                    pred_letter = prediction.strip()[0]
                else:
                    # Try to find first occurrence of A. B. C. or D.
                    for letter in ['A', 'B', 'C', 'D']:
                        pattern = letter + '.'
                        if pattern in prediction:
                            pred_letter = letter
                            break
            
            # Extract answer letter from target
            target_letter = None
            if isinstance(target, str):
                target_letter = target.strip()
                if len(target_letter) > 0:
                    target_letter = target_letter[0]
            
            # Pass 1.0 if correct, 0.0 if incorrect
            is_correct = 1.0 if pred_letter and target_letter and pred_letter == target_letter else 0.0
            super().update(torch.tensor(is_correct))

    return EvaluationMetrics(
        predicted_text={
            "exact_match": MultipleChoiceExactMatch(),
            "accuracy": MultipleChoiceAccuracy(),
        },
        acceptance_rate={"mean": Mean()},
        total_time={"mean": Mean()},
        time_per_token={"mean": Mean()},
        tokens_per_second={"mean": Mean()},
        total_flops={"mean": Mean()},  # Add FLOPs metrics
        flops_per_token={"mean": Mean()},
    )


def save_results_to_csv(results, dataset_name):
    """Save benchmark results to CSV file"""
    import csv
    import datetime
    import os
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{dataset_name}_results_{timestamp}.csv"
    
    print(f"Saving results to {filename}...")
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        # Identify fields from the first result
        if results:
            fieldnames = results[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                # Truncate very long fields for CSV manageability
                for key, value in result.items():
                    if isinstance(value, str) and len(value) > 32767:  # Excel limit
                        result[key] = value[:32767]
                writer.writerow(result)
            
            print(f"Saved {len(results)} results to {filename}")
        else:
            print("No results to save")
    
    return os.path.abspath(filename)


def original_benchmark(
        model: torch.nn.Module, 
        tokenizer: transformers.PreTrainedTokenizerBase, 
        benchmark_arguments: BenchmarkArguments, 
        generation_config: GenerationConfig,
        seed = None,
        measure_flops: bool = True,
    ):
    """The original benchmark function with FLOPs measurement"""
    if generation_config.generation_strategy == "autoregressive":
        generation_strategy: GenerationStrategy = AutoRegressiveGenerationStrategy()
    elif generation_config.generation_strategy == "self_speculative":
        generation_strategy: GenerationStrategy = SelfSpeculativeGenerationStrategy()
    elif generation_config.generation_strategy == "layerdrop":
            generation_strategy: GenerationStrategy = LayerDropGenerationStrategy(
                dropout_rate=generation_config.dropout_rate,
                seed=generation_config.layerdrop_seed or seed
            )
    elif generation_config.generation_strategy == "depth_adaptive_token":
        generation_strategy: GenerationStrategy = DepthAdaptiveTokenGenerationStrategy(
            halting_threshold=generation_config.halting_threshold,
            min_layers=generation_config.min_layers,
            max_layers=generation_config.max_layers,
        )
    elif generation_config.generation_strategy == "depth_adaptive_sequence":
        generation_strategy: GenerationStrategy = DepthAdaptiveSequenceGenerationStrategy(
            halting_threshold=generation_config.halting_threshold,
            min_layers=generation_config.min_layers,
            max_layers=generation_config.max_layers,
        )
    else:
        raise Exception(
            f"Unsupported generation strategy: {generation_config.generation_strategy}"
        )

    # Initialize generator
    generator = HuggingfaceLlamaGenerator(
        tokenizer=tokenizer, model=model, generation_strategy=generation_strategy
    )
    
    # Initialize FLOPs counter if measurement is enabled
    flops_counter = None
    if measure_flops:
        print("Initializing FLOPs counter for original benchmark")
        try:
            flops_counter = LlamaFLOPsCounter(model)
            flops_counter.register_hooks()
            print(f"FLOPs counter registered with {len(flops_counter.hooks)} hooks")
        except Exception as e:
            print(f"Error initializing FLOPs counter: {str(e)}")
            import traceback
            traceback.print_exc()
            measure_flops = False
    
    # Track total FLOPs by strategy
    total_flops_by_strategy = {
        "strategy": generation_config.generation_strategy,
        "config": generation_config.__dict__,
        "total_flops": 0,
        "total_tokens": 0,
        "flops_per_token": 0,
        "samples": []
    }

    evaluation_set = get_data(
        random_shuffle=benchmark_arguments.random_shuffle,
        num_samples=benchmark_arguments.num_samples,
        dataset=benchmark_arguments.dataset,
        n_shot=benchmark_arguments.n_shot,
        seed=seed,
        data_path=benchmark_arguments.data_path,
        template=benchmark_arguments.template,
    )
    
    metrics = EvaluationMetrics.build_metrics()
    for i, example in enumerate(tqdm(evaluation_set)):
        # Reset FLOPs counter before generation
        if measure_flops and flops_counter:
            flops_counter.reset_counters()
        
        response: GenerationResult = generator.generate(
            prompt=example.input,
            generation_config=generation_config,
        )
        
        # Collect FLOPs data after successful generation
        flops_data = None
        if measure_flops and flops_counter:
            flops_data = flops_counter.get_summary()
            
            # Calculate per-token metrics
            tokens_generated = response.num_tokens_generated
            if tokens_generated > 0:
                total_flops = flops_data["total_flops"]
                flops_per_token = total_flops / tokens_generated
                
                # Add to total FLOPs statistics
                total_flops_by_strategy["total_flops"] += total_flops
                total_flops_by_strategy["total_tokens"] += tokens_generated
                
                # Record sample results
                sample_result = {
                    "prompt": example.input[:100] + "..." if len(example.input) > 100 else example.input,
                    "tokens_generated": tokens_generated,
                    "total_flops": total_flops,
                    "flops_per_token": flops_per_token,
                }
                total_flops_by_strategy["samples"].append(sample_result)
                
                print(f"FLOPs for sample {i}: {flops_counter.humanize_flops(total_flops)}, "
                      f"Per token: {flops_counter.humanize_flops(flops_per_token)}")
        
        print(f"[Prompt]:\n{example.input}")
        print(f"[Reference Response]:\n{example.output}")
        print(f"[Model Response]:\n{response.decoded_prediction}")
        
        if response.generation_strategy_result.acceptance_rate is not None:
            print(f"[Acceptance Rate]: {response.generation_strategy_result.acceptance_rate}")
        
        if response.num_tokens_generated == 0:
            print("Skipping metrics of empty generation")
            continue
        
        example_dict = {"input": example.input, "output": example.output}
        metrics.update(example_dict, response, flops_data)
    
    # Calculate final metrics    
    metric_result = metrics.compute()
    
    # Add final FLOPs statistics if available
    if measure_flops and total_flops_by_strategy["total_tokens"] > 0:
        total_flops_by_strategy["flops_per_token"] = (
            total_flops_by_strategy["total_flops"] / total_flops_by_strategy["total_tokens"]
        )
        
        # Add timestamp
        total_flops_by_strategy["timestamp"] = datetime.datetime.now().isoformat()
        
        # Save FLOPs results
        os.makedirs("./flops_data", exist_ok=True)
        flops_output_file = f"./flops_data/flops_{benchmark_arguments.dataset}_{generation_config.generation_strategy}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(flops_output_file, "w") as f:
            json.dump(total_flops_by_strategy, f, indent=2)
        
        print(f"FLOPs measurements saved to {flops_output_file}")
        print(f"Average FLOPs per token: {flops_counter.humanize_flops(total_flops_by_strategy['flops_per_token'])}")
        
        # Add FLOPs to result metrics
        if "total_flops" not in metric_result:
            metric_result["total_flops"] = {}
        if "flops_per_token" not in metric_result:
            metric_result["flops_per_token"] = {}
            
        metric_result["total_flops"]["total"] = total_flops_by_strategy["total_flops"]
        metric_result["flops_per_token"]["average"] = total_flops_by_strategy["flops_per_token"]
    
    # Clean up FLOPs counter if it was used
    if measure_flops and flops_counter:
        flops_counter.remove_hooks()
        
    return metric_result





def main(args: Arguments, benchmark_arguments: BenchmarkArguments, generation_config: GenerationConfig, output_fname: str, measure_flops: bool = True):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Log arguments at beginning
    log.info(f"device={device}\n"
             f"args={args}\n"
             f"benchmark_arguments={benchmark_arguments}\n"
             f"generation_config={generation_config}\n"
             f"output_fname={output_fname}\n"
             f"measure_flops={measure_flops}\n")

    # Setup and Run Benchmark
    setup(args, device=device)
    model, tokenizer = load_model_and_tokenizer(args, device=device)
    metric_result = benchmark(model, tokenizer, benchmark_arguments, generation_config, measure_flops=measure_flops)
    print(metric_result)

    # Save config and results to file
    result_data = {
        "args": args.__dict__,
        "benchmark_arguments": benchmark_arguments.__dict__,
        "generation_config": generation_config.__dict__,
        "metrics": metric_result,
        "measure_flops": measure_flops,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    with open(output_fname, "w") as f:
        json.dump(result_data, f, indent=2)
    
    print(f"Results saved to {output_fname}")
    
    # Also save a CSV version of the results if metrics include FLOPs data
    if measure_flops and "flops_per_token" in metric_result:
        csv_filename = output_fname.replace(".json", ".csv")
        with open(csv_filename, "w", newline="") as csvfile:
            fieldnames = ["dataset", "strategy", "accuracy", "total_flops", "flops_per_token", "efficiency"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            
            # Calculate efficiency (accuracy per FLOP)
            accuracy = metric_result["predicted_text"].get("accuracy", 0)
            flops_per_token = metric_result["flops_per_token"].get("average", 0)
            efficiency = accuracy / flops_per_token if flops_per_token > 0 else 0
            
            writer.writerow({
                "dataset": benchmark_arguments.dataset,
                "strategy": generation_config.generation_strategy,
                "accuracy": accuracy,
                "total_flops": metric_result["total_flops"].get("total", 0),
                "flops_per_token": flops_per_token,
                "efficiency": efficiency
            })
            
        print(f"CSV results saved to {csv_filename}")

def process_cli_arguments() -> Tuple[arguments.Arguments, BenchmarkArguments, GenerationConfig, bool]:
    # Use HfArgumentParser but use parse_args_into_dataclasses which returns the correct type
    parser = transformers.HfArgumentParser((arguments.Arguments, BenchmarkArguments, GenerationConfig))
    
    try:
        # This will parse arguments into the specified dataclass types
        general_arguments, benchmark_arguments, generation_config = parser.parse_args_into_dataclasses()
        
        # Set default for measure_flops
        measure_flops = True
        
        # Check command line args for measure_flops flag
        for arg in sys.argv:
            if arg == "--no_measure_flops":
                measure_flops = False
                break
    
        assert benchmark_arguments.dataset in get_valid_dataset_formats(), f"{benchmark_arguments.dataset} is not a supported dataset!"
        
        if general_arguments.model_args:
            general_arguments.model_args = simple_parse_args_string(general_arguments.model_args)
        else:
            general_arguments.model_arg = {}
            
        return general_arguments, benchmark_arguments, generation_config, measure_flops
    except Exception as e:
        print(f"Error parsing arguments: {e}")
        print("Command line arguments:", sys.argv)
        raise

if __name__ == "__main__":
    args, benchmark_arguments, generation_config, measure_flops = process_cli_arguments()
    log.setLevel(level=logging.INFO) # TODO: set level based on argument
    os.makedirs(args.output_dir, exist_ok=True)
    output_fname = f"{args.output_dir}/benchmark_{benchmark_arguments.dataset}_{generation_config.generation_strategy}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    main(args, benchmark_arguments, generation_config, output_fname, measure_flops)

