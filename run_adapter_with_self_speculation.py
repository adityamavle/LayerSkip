#!/usr/bin/env python3
"""
Script to run adapter-enhanced self-speculative inference on RACE-M or RACE-H datasets.
"""

import argparse
import torch
import json
import logging
import os
import time
import csv
from typing import List, Optional, Tuple
from tqdm import tqdm
from datasets import load_dataset
import numpy as np
import colorama

from arguments import Arguments
from generate import load_model_and_tokenizer, setup
from LlamaLayerExtractor import LlamaLayerExtractor
from DebertaAdapter import DebertaAdapter

from self_speculation.generator_base import (
    GenerationConfig,
    GenerationResult,
    GenerationStrategy,
    HuggingfaceLlamaGenerator,
    GenerationStrategyResult
)
from self_speculation.self_speculation_generator import SelfSpeculativeGenerationStrategy
from self_speculation.llama_model_utils import (
    crop_past_key_values,
    decode_next_token,
    forward_early,
    forward_remainder,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdapterEnhancedSpeculativeStrategy(SelfSpeculativeGenerationStrategy):
    """
    Enhances the self-speculative generation strategy with a DeBERTa adapter.
    """
    def __init__(self, adapter: DebertaAdapter, layer_extractor: LlamaLayerExtractor):
        super().__init__()
        self.adapter = adapter
        self.layer_extractor = layer_extractor
        self.adapter.eval()  # Set adapter to evaluation mode
    
    def generate_token_ids(
        self,
        model: torch.nn.Module,
        input_ids: List[int],
        eos_token_ids: List[int],
        generation_config: GenerationConfig,
        logits_processors = None,
        stopping_criteria = None,
        streamer = None,
    ) -> GenerationStrategyResult:
        """
        Override the generate_token_ids method to integrate the adapter.
        """
        past_key_values = None

        input_ids_list = input_ids
        input_ids_tensor: torch.Tensor = torch.tensor([input_ids_list]).to(model.device)
        output_ids: List[int] = []

        calls: int = 0
        total_draft_matches = 0
        total_generations = 0
        
        # Save a reference to the model for the layer extractor
        self.layer_extractor.model = model
        
        while len(output_ids) < generation_config.max_steps:
            (
                input_ids_tensor,
                output_ids,
                past_key_values,
                number_of_matches,
                num_speculations,
            ) = self.single_step_speculation_with_adapter(
                model=model,
                input_ids_list=input_ids_list,
                input_ids=input_ids_tensor,
                output_ids=output_ids,
                num_speculations=min(
                    generation_config.num_speculations,
                    generation_config.max_steps - len(output_ids) - 1,
                ),
                past_key_values=past_key_values,
                exit_layer=generation_config.exit_layer,
                eos_token_ids=eos_token_ids,
                calls=calls,
                sample=generation_config.sample,
                temperature=generation_config.temperature,
                top_k=generation_config.top_k,
                top_p=generation_config.top_p,
                logits_processors=logits_processors,
                stopping_criteria=stopping_criteria,
                streamer=streamer,
            )
            calls += 1
            total_draft_matches += number_of_matches
            total_generations += num_speculations
            
            # Check for EOS
            eos_found = False
            for eos_token_id in eos_token_ids:
                if eos_token_id in output_ids:
                    # remove the EOS token id
                    output_ids = output_ids[: output_ids.index(eos_token_id)]
                    eos_found = True
                    break
                    
            if eos_found:
                break
                
            if stopping_criteria:
                if torch.all(stopping_criteria(input_ids_tensor, scores=None)):
                    break
                    
        return GenerationStrategyResult(
            predicted_tokens=output_ids,
            acceptance_rate=total_draft_matches / total_generations if total_generations > 0 else 0,
        )
    
    def single_step_speculation_with_adapter(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        input_ids_list: List[int],
        output_ids: List[int],
        num_speculations: int,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
        eos_token_ids: List[int],
        calls: int,
        exit_layer: int,
        sample: Optional[bool] = False,
        temperature: Optional[float] = 0.7,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.95,
        logits_processors = None,
        stopping_criteria = None,
        streamer = None
    ):
        zero_division_count = 0  # Counter for ZeroDivisionErrors
        
        prompt_length: int = input_ids.size(1)
        draft_input_ids = input_ids.clone()
        draft_output_ids: List[int] = []
        
        if sample:
            draft_probabilities: List[torch.Tensor] = []
        
        exit_query_cache = None

        for _ in range(num_speculations):
            # Get the draft token using early exit and adapter enhancement
            draft_result = forward_early(
                model,
                draft_input_ids,
                past_key_values,
                exit_layer,
                exit_query_cache,
            )
            past_key_values = draft_result.past_key_values
            exit_query_cache = draft_result.exit_query_cache
            
            # Get hidden states using the layer extractor
            with torch.no_grad():
                hidden_states = self.layer_extractor.get_layer_output(
                    draft_input_ids, 
                    past_key_values=past_key_values
                )
                
                # Get the last token's hidden state
                last_hidden = hidden_states[:, -1, :]
                
                # Enhance with adapter
                adapter_logits = self.adapter(last_hidden)
                
                # Combine with original draft logits
                alpha = 0.7  # Weight for adapter (adjust as needed)
                draft_logits = draft_result.logits
                enhanced_logits = alpha * adapter_logits + (1 - alpha) * draft_logits
                
                if logits_processors:
                    enhanced_logits = logits_processors(draft_input_ids, enhanced_logits)
                
                draft_next_token, draft_next_prob = decode_next_token(
                    logits=enhanced_logits, 
                    token_idx=-1, 
                    sample=sample, 
                    temperature=temperature, 
                    top_k=top_k, 
                    top_p=top_p
                )
            
            draft_next_token = draft_next_token.item()
            draft_output_ids.append(draft_next_token)
            
            if sample:
                draft_probabilities.append(draft_next_prob)
            
            draft_input_ids = torch.tensor([[draft_next_token]]).to(draft_input_ids)
            
            if draft_next_token in eos_token_ids:
                break

        # Convert draft output IDs to tensor
        draft_output_ids = torch.tensor(draft_output_ids).unsqueeze(0).to(input_ids)
        prefill_token_ids = torch.cat([input_ids, draft_output_ids], dim=-1)

        if streamer:
            if hasattr(streamer, 'is_draft') and hasattr(streamer, 'put'):
                print(colorama.Fore.LIGHTMAGENTA_EX, end="")
                streamer.put(draft_output_ids, is_draft=True)

        # Get verification logits
        verify_results = forward_remainder(
            model,
            prefill_token_ids.int(),
            past_key_values,
            exit_layer,
            exit_query_cache,
        )
        logits = verify_results.logits
        
        if logits_processors:
            logits = logits_processors(prefill_token_ids, logits)
        
        past_key_values = verify_results.past_key_values

        verification_logits = logits[:, prompt_length - 1:, :]
        verified_tokens, verified_probabilities = decode_next_token(
            logits=verification_logits, 
            sample=sample, 
            temperature=temperature, 
            top_k=top_k, 
            top_p=top_p
        )

        verified_tokens = verified_tokens.to(prefill_token_ids)
        verified = draft_output_ids[:, :] == verified_tokens[:, :-1]

        if not sample:
            number_of_matches = ((~(verified)).cumsum(dim=-1) < 1).sum().item()
        else:
            number_of_matches = 0
            rand = torch.rand_like(draft_output_ids, dtype=torch.float)

            for i in range(draft_output_ids.numel()):
                denominator = draft_probabilities[i][0, draft_output_ids[0, i]].item()

                # Handle potential division by zero
                if denominator == 0 or denominator < 1e-8:
                    zero_division_count += 1
                    denominator = 1e-8  # Small epsilon to avoid division by zero
                
                # Compute acceptance ratio
                acceptance_ratio = verified_probabilities[i, draft_output_ids[0, i]].item() / denominator
                
                # Accept or reject based on probability ratio
                if rand[0, i] < min(1, acceptance_ratio):
                    number_of_matches += 1
                else:
                    # Compute probability distribution for rejection sampling
                    prob_dist = torch.clamp(verified_probabilities[i, :] - draft_probabilities[i], min=0)
                    
                    # Handle degenerate case where all probabilities are zero
                    if prob_dist.sum().item() <= 1e-8:
                        prob_dist = torch.ones_like(prob_dist) / prob_dist.shape[-1]
                    else:
                        prob_dist = prob_dist / prob_dist.sum()  # Normalize
                    
                    # Sample the next token
                    verified_tokens[0][number_of_matches] = torch.multinomial(prob_dist, num_samples=1).item()
                    break

        input_ids = verified_tokens[:, number_of_matches:number_of_matches + 1]
        output_ids.extend(draft_output_ids[0, :number_of_matches].tolist())
        output_ids.extend(verified_tokens[0][number_of_matches:number_of_matches + 1].tolist())

        if streamer:
            if hasattr(streamer, 'delete') and hasattr(streamer, 'put'):
                streamer.delete(len(draft_output_ids[0, :]))
                print(colorama.Fore.GREEN, end="")
                streamer.put(draft_output_ids[0, :number_of_matches])
                print(colorama.Style.RESET_ALL, end="")
                streamer.put(verified_tokens[0][number_of_matches:number_of_matches + 1])
            elif hasattr(streamer, 'put'):
                streamer.put(torch.LongTensor(output_ids[len(output_ids)-number_of_matches-1:]))

        past_key_values = crop_past_key_values(
            past_key_values, len(input_ids_list) + len(output_ids) - 1
        )

        return (
            input_ids,
            output_ids,
            past_key_values,
            number_of_matches,
            draft_output_ids.numel(),
        )

def prepare_race_dataset(dataset_name, split='test', max_samples=None):
    """
    Prepare RACE-M or RACE-H dataset for evaluation.
    
    Args:
        dataset_name: 'race_m' or 'race_h'
        split: Dataset split to use ('train', 'validation', 'test')
        max_samples: Maximum number of samples to use (optional)
    
    Returns:
        List of formatted examples
    """
    race_type = 'middle' if dataset_name == 'race_m' else 'high'
    dataset = load_dataset("race", race_type, split=split)
    
    if max_samples is not None:
        indices = list(range(min(max_samples, len(dataset))))
        dataset = dataset.select(indices)
    
    examples = []
    for item in dataset:
        article = item['article']
        questions = item['questions']
        options = item['options']
        answers = item['answers']  # These are letter indices like 'A', 'B', etc.
        
        for q_idx, (question, opts, answer) in enumerate(zip(questions, options, answers)):
            # Format as input to the model
            formatted_text = f"Article: {article}\n\nQuestion: {question}\n"
            for j, option in enumerate(opts):
                formatted_text += f"{chr(65 + j)}. {option}\n"
            formatted_text += "Answer:"
            
            # Create evaluation example
            examples.append({
                'id': f"{item['id']}_{q_idx}",
                'prompt': formatted_text,
                'answer': answer,  # The correct answer letter (A, B, C, D)
                'options': opts
            })
    
    return examples

def extract_answer_letter(text):
    """
    Extract the answer letter (A, B, C, D) from model output.
    """
    # First check if the output starts with A, B, C, or D
    text = text.strip()
    if text and text[0] in 'ABCD':
        return text[0]
    
    # Look for "The answer is X" pattern
    import re
    match = re.search(r"[Tt]he answer is\s+([ABCD])", text)
    if match:
        return match.group(1)
    
    # Look for patterns like "A.", "B.", etc.
    for letter in 'ABCD':
        pattern = f"{letter}\."
        if re.search(pattern, text):
            return letter
    
    # If nothing found, return the first letter that appears in the text
    for char in text:
        if char in 'ABCD':
            return char
    
    # If still nothing found, return None
    return None

def evaluate_accuracy(predictions, ground_truth):
    """
    Calculate accuracy of predictions.
    """
    correct = sum(1 for p, gt in zip(predictions, ground_truth) if p == gt)
    return correct / len(predictions) if predictions else 0

def main():
    parser = argparse.ArgumentParser(description="Run adapter-enhanced self-speculative inference on RACE dataset")
    parser.add_argument("--model", type=str, required=True, help="Path to base LLaMA model")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to trained adapter weights")
    parser.add_argument("--adapter_config", type=str, required=True, help="Path to adapter config JSON")
    parser.add_argument("--dataset", type=str, choices=["race_m", "race_h"], default="race_m", 
                      help="Dataset to evaluate on")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to evaluate")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--max_length", type=int, default=20, help="Maximum length of generated text")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--exit_layer", type=int, default=8, help="Layer to exit for speculation")
    parser.add_argument("--num_speculations", type=int, default=5, help="Number of tokens to speculate")
    parser.add_argument("--output", type=str, default=None, help="Output file for results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                      help="Device to run inference on")
    parser.add_argument("--compare", action="store_true", help="Compare with regular self-speculative generation")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for dataset loading")
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    logger.info(f"Loading model from {args.model}")
    model_args = Arguments(model=args.model, seed=42, output_dir="./outputs")
    setup(model_args)
    model, tokenizer = load_model_and_tokenizer(model_args, device=args.device)
    
    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load adapter configuration
    with open(args.adapter_config, 'r') as f:
        adapter_config = json.load(f)
    
    # Initialize adapter
    logger.info("Initializing DeBERTa adapter")
    adapter = DebertaAdapter(
        llama_hidden_size=adapter_config.get("llama_hidden_size", model.config.hidden_size),
        llama_vocab_size=adapter_config.get("llama_vocab_size", model.config.vocab_size),
        deberta_hidden_size=adapter_config.get("deberta_hidden_size", 768),
        deberta_num_layers=adapter_config.get("deberta_num_layers", 2),
        deberta_num_attention_heads=adapter_config.get("deberta_num_heads", 12),
        dropout_prob=adapter_config.get("dropout_prob", 0.1)
    )
    
    # Load adapter weights
    adapter.load_state_dict(torch.load(args.adapter_path, map_location=args.device))
    adapter.to(args.device)
    logger.info(f"Loaded adapter weights from {args.adapter_path}")
    
    # Initialize layer extractor
    target_layer = adapter_config.get("target_layer", args.exit_layer)
    layer_extractor = LlamaLayerExtractor(model, target_layer=target_layer)
    
    # Create enhanced speculative strategy
    enhanced_strategy = AdapterEnhancedSpeculativeStrategy(
        adapter=adapter,
        layer_extractor=layer_extractor
    )
    
    # Initialize generator
    enhanced_generator = HuggingfaceLlamaGenerator(
        tokenizer=tokenizer,
        model=model,
        generation_strategy=enhanced_strategy
    )
    
    # Configure generation
    generation_config = GenerationConfig(
        max_steps=args.max_length,
        exit_layer=args.exit_layer,
        num_speculations=args.num_speculations,
        generation_strategy="self_speculative",  # This is just a label
        sample=args.temperature > 0,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )
    
    # Initialize regular speculative generator for comparison if requested
    if args.compare:
        regular_strategy = SelfSpeculativeGenerationStrategy()
        regular_generator = HuggingfaceLlamaGenerator(
            tokenizer=tokenizer,
            model=model,
            generation_strategy=regular_strategy
        )
    
    # Prepare dataset
    logger.info(f"Preparing {args.dataset} dataset ({args.split} split)")
    examples = prepare_race_dataset(args.dataset, args.split, args.max_samples)
    logger.info(f"Loaded {len(examples)} examples")
    
    # Initialize results tracking
    results = []
    enhanced_predictions = []
    ground_truth = [example['answer'] for example in examples]
    
    if args.compare:
        regular_predictions = []
    
    # Track acceptance rates
    enhanced_acceptance_rates = []
    if args.compare:
        regular_acceptance_rates = []
    
    # Run inference
    logger.info("Starting inference with adapter-enhanced model...")
    start_time = time.time()
    
    for i, example in enumerate(tqdm(examples, desc="Processing examples")):
        # Generate with adapter-enhanced model
        try:
            enhanced_result = enhanced_generator.generate(
                prompt=example['prompt'],
                generation_config=generation_config
            )
            
            enhanced_text = enhanced_result.decoded_prediction
            enhanced_answer = extract_answer_letter(enhanced_text)
            enhanced_predictions.append(enhanced_answer)
            enhanced_acceptance_rates.append(enhanced_result.generation_strategy_result.acceptance_rate)
            
            # Generate with regular model if comparison is requested
            if args.compare:
                regular_result = regular_generator.generate(
                    prompt=example['prompt'],
                    generation_config=generation_config
                )
                
                regular_text = regular_result.decoded_prediction
                regular_answer = extract_answer_letter(regular_text)
                regular_predictions.append(regular_answer)
                regular_acceptance_rates.append(regular_result.generation_strategy_result.acceptance_rate)
            
            # Store result
            result_item = {
                'id': example['id'],
                'prompt': example['prompt'],
                'ground_truth': example['answer'],
                'enhanced_prediction': enhanced_answer,
                'enhanced_text': enhanced_text,
                'enhanced_acceptance_rate': enhanced_result.generation_strategy_result.acceptance_rate
            }
            
            if args.compare:
                result_item.update({
                    'regular_prediction': regular_answer,
                    'regular_text': regular_text,
                    'regular_acceptance_rate': regular_result.generation_strategy_result.acceptance_rate
                })
            
            results.append(result_item)
            
            # Log progress
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(examples)} examples")
                
                # Calculate and log current accuracy
                current_enhanced_acc = evaluate_accuracy(enhanced_predictions, ground_truth[:len(enhanced_predictions)])
                current_enhanced_avg_acceptance = sum(enhanced_acceptance_rates) / len(enhanced_acceptance_rates)
                logger.info(f"Current enhanced accuracy: {current_enhanced_acc:.4f}")
                logger.info(f"Current enhanced avg acceptance rate: {current_enhanced_avg_acceptance:.4f}")
                
                if args.compare and regular_predictions:
                    current_regular_acc = evaluate_accuracy(regular_predictions, ground_truth[:len(regular_predictions)])
                    current_regular_avg_acceptance = sum(regular_acceptance_rates) / len(regular_acceptance_rates)
                    logger.info(f"Current regular accuracy: {current_regular_acc:.4f}")
                    logger.info(f"Current regular avg acceptance rate: {current_regular_avg_acceptance:.4f}")
                
        except Exception as e:
            logger.error(f"Error processing example {example['id']}: {e}")
            # Add a placeholder for failed examples
            enhanced_predictions.append(None)
            enhanced_acceptance_rates.append(None)
            if args.compare:
                regular_predictions.append(None)
                regular_acceptance_rates.append(None)
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Calculate final accuracy
    enhanced_accuracy = evaluate_accuracy(
        [p for p in enhanced_predictions if p is not None], 
        [gt for p, gt in zip(enhanced_predictions, ground_truth) if p is not None]
    )
    
    enhanced_avg_acceptance = sum([r for r in enhanced_acceptance_rates if r is not None]) / \
                             len([r for r in enhanced_acceptance_rates if r is not None])
    
    if args.compare:
        regular_accuracy = evaluate_accuracy(
            [p for p in regular_predictions if p is not None],
            [gt for p, gt in zip(regular_predictions, ground_truth) if p is not None]
        )
        
        regular_avg_acceptance = sum([r for r in regular_acceptance_rates if r is not None]) / \
                                len([r for r in regular_acceptance_rates if r is not None])
    
    # Print results
    logger.info("\n======= EVALUATION RESULTS =======")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Number of examples: {len(examples)}")
    logger.info(f"Total time: {total_time:.2f} seconds")
    logger.info(f"Average time per example: {total_time / len(examples):.2f} seconds")
    logger.info(f"Enhanced model accuracy: {enhanced_accuracy:.4f}")
    logger.info(f"Enhanced model average acceptance rate: {enhanced_avg_acceptance:.4f}")
    
    if args.compare:
        logger.info(f"Regular model accuracy: {regular_accuracy:.4f}")
        logger.info(f"Regular model average acceptance rate: {regular_avg_acceptance:.4f}")
        logger.info(f"Accuracy improvement: {enhanced_accuracy - regular_accuracy:.4f}")
        logger.info(f"Acceptance rate improvement: {enhanced_avg_acceptance - regular_avg_acceptance:.4f}")
    
    # Save results to CSV if output path is provided
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        with open(args.output, 'w', newline='') as csvfile:
            fieldnames = ['id', 'prompt', 'ground_truth', 'enhanced_prediction', 
                         'enhanced_text', 'enhanced_acceptance_rate']
            
            if args.compare:
                fieldnames.extend(['regular_prediction', 'regular_text', 'regular_acceptance_rate'])
                
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                writer.writerow(result)
        
        logger.info(f"Results saved to {args.output}")
        
        # Save summary statistics
        summary_path = os.path.splitext(args.output)[0] + "_summary.json"
        summary = {
            "dataset": args.dataset,
            "split": args.split,
            "num_examples": len(examples),
            "model": args.model,
            "adapter": args.adapter_path,
            "exit_layer": args.exit_layer,
            "num_speculations": args.num_speculations,
            "enhanced_accuracy": enhanced_accuracy,
            "enhanced_avg_acceptance_rate": enhanced_avg_acceptance,
            "total_time": total_time,
            "avg_time_per_example": total_time / len(examples),
        }
        
        if args.compare:
            summary.update({
                "regular_accuracy": regular_accuracy,
                "regular_avg_acceptance_rate": regular_avg_acceptance,
                "accuracy_improvement": enhanced_accuracy - regular_accuracy,
                "acceptance_rate_improvement": enhanced_avg_acceptance - regular_avg_acceptance,
            })
            
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Summary statistics saved to {summary_path}")
    
    # Clean up
    layer_extractor.remove_hooks()
    logger.info("Evaluation complete!")

if __name__ == "__main__":
    main()
