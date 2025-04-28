import os
import json
import argparse
import logging
import torch
import transformers
from typing import Dict, Optional
import numpy as np
import datetime

from self_speculation.flops_counter import LlamaFLOPsCounter
from self_speculation.autoregressive_generator import AutoRegressiveGenerationStrategy
from self_speculation.self_speculation_generator import SelfSpeculativeGenerationStrategy
from self_speculation.layer_drop_generator import LayerDropGenerationStrategy
from self_speculation.depth_adaptive_sequence_generator import DepthAdaptiveSequenceGenerationStrategy
from self_speculation.generator_base import (
    GenerationConfig,
    HuggingfaceLlamaGenerator,
    GenerationStrategy,
)
from generate import load_model_and_tokenizer, setup
from arguments import Arguments, simple_parse_args_string

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def measure_flops(
    model: transformers.LlamaForCausalLM,
    tokenizer: transformers.LlamaTokenizer,
    prompts: list,
    generation_config: GenerationConfig,
    strategy_name: str,
    num_samples: int = 10,
    device: str = "cuda",
    log_dir: str = "./flops_logs",
) -> Dict:
    """
    Measure FLOPs for a specific generation strategy on a set of prompts.
    """
    logger.info(f"Measuring FLOPs for {strategy_name} strategy")
    
    # Create FLOPs counter
    flops_counter = LlamaFLOPsCounter(model)
    
    # Choose strategy
    if strategy_name == "autoregressive":
        generation_strategy = AutoRegressiveGenerationStrategy()
    elif strategy_name == "self_speculative":
        generation_strategy = SelfSpeculativeGenerationStrategy()
    elif strategy_name == "layerdrop":
        generation_strategy = LayerDropGenerationStrategy(
            dropout_rate=generation_config.dropout_rate,
            seed=generation_config.layerdrop_seed
        )
    elif strategy_name == "depth_adaptive_sequence":
        generation_strategy = DepthAdaptiveSequenceGenerationStrategy(
            halting_threshold=generation_config.halting_threshold,
            min_layers=generation_config.min_layers,
            max_layers=generation_config.max_layers,
        )
    else:
        raise ValueError(f"Unknown strategy name: {strategy_name}")
    
    # Initialize generator
    generator = HuggingfaceLlamaGenerator(
        tokenizer=tokenizer,
        model=model,
        generation_strategy=generation_strategy,
    )
    
    # Prepare output
    results = {
        "strategy": strategy_name,
        "config": generation_config.__dict__,
        "total_flops": 0,
        "total_tokens": 0,
        "flops_per_token": 0,
        "samples": [],
    }
    
    # Register hooks to count FLOPs
    flops_counter.register_hooks()
    
    # Run generation on prompts
    for i, prompt in enumerate(prompts[:num_samples]):
        logger.info(f"Processing prompt {i+1}/{min(num_samples, len(prompts))}")
        
        # Reset counters for this prompt
        flops_counter.reset_counters()
        
        try:
            # Generate response
            with torch.inference_mode():
                generation_result = generator.generate(
                    prompt=prompt, 
                    generation_config=generation_config
                )
            
            # Get FLOPs summary
            flops_summary = flops_counter.get_summary()
            
            # Calculate per-token metrics
            tokens_generated = generation_result.num_tokens_generated
            total_flops = flops_summary["total_flops"]
            flops_per_token = total_flops / tokens_generated if tokens_generated > 0 else 0
            
            # Record sample results
            sample_result = {
                "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                "response": generation_result.decoded_prediction[:100] + "..." if len(generation_result.decoded_prediction) > 100 else generation_result.decoded_prediction,
                "tokens_generated": tokens_generated,
                "total_flops": total_flops,
                "flops_per_token": flops_per_token,
                "time_per_token": generation_result.time_per_token,
                "total_time": generation_result.total_time,
                "acceptance_rate": generation_result.generation_strategy_result.acceptance_rate,
                "flops_by_operation": flops_summary["operation_flops"],
            }
            
            # Add early exit savings if applicable
            if strategy_name == "autoregressive" and generation_config.exit_layer > 0:
                sample_result["early_exit_savings"] = flops_counter.estimate_early_exit_savings(generation_config.exit_layer)
            
            # Add to results
            results["samples"].append(sample_result)
            results["total_flops"] += total_flops
            results["total_tokens"] += tokens_generated
            
        except Exception as e:
            logger.error(f"Error processing prompt {i}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Remove hooks
    flops_counter.remove_hooks()
    
    # Calculate averages
    if results["total_tokens"] > 0:
        results["flops_per_token"] = results["total_flops"] / results["total_tokens"]
    
    # Add timestamp
    results["timestamp"] = datetime.datetime.now().isoformat()
    
    # Save results
    os.makedirs(log_dir, exist_ok=True)
    output_file = os.path.join(
        log_dir, 
        f"flops_{strategy_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"FLOPs measurements saved to {output_file}")
    logger.info(f"Average FLOPs per token: {flops_counter.humanize_flops(results['flops_per_token'])}")
    
    return results

def get_sample_prompts(dataset_name: str, num_samples: int = 10) -> list:
    """
    Get a list of sample prompts from a dataset.
    """
    from data import get_data
    
    evaluation_data_points = get_data(
        random_shuffle=True,
        num_samples=num_samples,
        dataset=dataset_name,
        seed=42,
    )
    
    prompts = []
    for example in evaluation_data_points:
        if hasattr(example, 'input'):
            prompts.append(example.input)
        elif isinstance(example, dict) and 'input' in example:
            prompts.append(example['input'])
    
    if not prompts:
        logger.warning(f"No prompts found in dataset {dataset_name}. Using dummy prompts.")
        # Fallback to dummy prompts
        prompts = [f"This is a test prompt for the {dataset_name} dataset. " * 10] * num_samples
    
    logger.info(f"Loaded {len(prompts)} prompts from dataset {dataset_name}")
    return prompts

def collect_flops_data_for_all_strategies(
    model_name_or_path: str, 
    dataset: str,
    output_dir: str = "./flops_data",
    num_samples: int = 10,
    device: str = "cuda"
):
    """Collect FLOPs data for all generation strategies on a dataset."""
    
    # Initialize model
    try:
        args = Arguments(model=model_name_or_path, distributed=False)
        setup(args, device=device)
        model, tokenizer = load_model_and_tokenizer(args, device=device)
        
        # Verify model is properly loaded
        if model is None or not isinstance(model, transformers.LlamaForCausalLM):
            logger.error(f"Failed to load model properly. Model type: {type(model)}")
            raise ValueError("Model not loaded correctly")
            
        # Verify tokenizer is properly loaded
        if tokenizer is None or not isinstance(tokenizer, transformers.PreTrainedTokenizerBase):
            logger.error(f"Failed to load tokenizer properly. Tokenizer type: {type(tokenizer)}")
            raise ValueError("Tokenizer not loaded correctly")
        
        logger.info(f"Successfully loaded model '{model_name_or_path}' and tokenizer")
        logger.info(f"Model has {model.config.num_hidden_layers} layers")
    except Exception as e:
        logger.error(f"Error loading model and tokenizer: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise
        
    # Get prompts
    try:
        prompts = get_sample_prompts(dataset, num_samples)
        if not prompts:
            logger.error(f"No prompts found for dataset {dataset}")
            raise ValueError(f"No prompts found for dataset {dataset}")
        logger.info(f"Successfully loaded {len(prompts)} prompts from dataset {dataset}")
    except Exception as e:
        logger.error(f"Error loading prompts: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    
    # Strategies to benchmark
    strategies = [
        # Autoregressive full model
        {
            "name": "autoregressive",
            "config": GenerationConfig(
                max_steps=128,
                exit_layer=-1,
                generation_strategy="autoregressive",
            ),
            "description": "Autoregressive (Full Model)",
        },
        # Autoregressive with early exit
        {
            "name": "autoregressive",
            "config": GenerationConfig(
                max_steps=128,
                exit_layer=8,
                generation_strategy="autoregressive",
            ),
            "description": "Autoregressive (Early Exit at Layer 8)",
        },
        # Self-speculative
        {
            "name": "self_speculative",
            "config": GenerationConfig(
                max_steps=128,
                exit_layer=8,
                num_speculations=6,
                generation_strategy="self_speculative",
            ),
            "description": "Self-Speculative (Draft: Layer 8, Spec: 6)",
        },
        # LayerDrop
        {
            "name": "layerdrop",
            "config": GenerationConfig(
                max_steps=128,
                dropout_rate=0.2,
                layerdrop_seed=42,
                generation_strategy="layerdrop",
            ),
            "description": "LayerDrop (Dropout Rate: 0.2)",
        },
        # Depth Adaptive Sequence
        {
            "name": "depth_adaptive_sequence",
            "config": GenerationConfig(
                max_steps=128,
                halting_threshold=0.99,
                min_layers=4,
                generation_strategy="depth_adaptive_sequence",
            ),
            "description": "Depth Adaptive Sequence (Halting: 0.99)",
        },
    ]
    
    # Output for all strategies
    all_results = {
        "model": model_name_or_path,
        "dataset": dataset,
        "timestamp": datetime.datetime.now().isoformat(),
        "strategies": [],
        "base_flops_per_token": 0.0,  # Will set this from first strategy
        "total_layers": model.config.num_hidden_layers
    }
    
    # Run each strategy
    for i, strategy in enumerate(strategies):
        logger.info(f"Measuring FLOPs for {strategy['description']}")
        
        try:
            # Measure FLOPs
            strategy_result = measure_flops(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                generation_config=strategy["config"],
                strategy_name=strategy["name"],
                num_samples=num_samples,
                device=device,
                log_dir=output_dir
            )
            
            # Add to results
            result = {
                "name": strategy["name"],
                "description": strategy["description"],
                "config": strategy["config"].__dict__,
                "flops_per_token": strategy_result.get("flops_per_token", 0.0),
                "time_per_token": strategy_result.get("time_per_token", 0.0),
            }
            
            # Save base flops from the first strategy (autoregressive full model)
            if i == 0:
                all_results["base_flops_per_token"] = max(result["flops_per_token"], 1.0)  # Ensure non-zero
            
            # Calculate relative efficiency compared to base model
            base_flops = all_results["base_flops_per_token"]
            if base_flops > 0 and result["flops_per_token"] > 0:
                result["relative_efficiency"] = base_flops / result["flops_per_token"]
            else:
                # Handle division by zero cases
                result["relative_efficiency"] = 1.0  # Default to 1.0 for failed measurements
                
            all_results["strategies"].append(result)
            
        except Exception as e:
            logger.error(f"Error measuring FLOPs for {strategy['description']}: {e}")
            # Add placeholder result
            result = {
                "name": strategy["name"],
                "description": strategy["description"],
                "config": strategy["config"].__dict__,
                "flops_per_token": 0.0,
                "time_per_token": 0.0,
                "relative_efficiency": 1.0,
                "error": str(e)
            }
            all_results["strategies"].append(result)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"flops_summary_{dataset}.json")
    
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"FLOPs data collected and saved to {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser(description="Measure FLOPs for different generation strategies")
    parser.add_argument("--model", type=str, required=True, help="Path to the model")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use for prompts")
    parser.add_argument("--output_dir", type=str, default="./flops_data", help="Output directory for FLOPs data")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to use for FLOPs measurement")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for inference")
    
    args = parser.parse_args()
    
    collect_flops_data_for_all_strategies(
        model_name_or_path=args.model,
        dataset=args.dataset,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        device=args.device,
    )

if __name__ == "__main__":
    main() 