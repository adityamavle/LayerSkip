#!/usr/bin/env python3
"""
Script to run adapter-enhanced autoregressive inference on RACE-M or RACE-H datasets.
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
from self_speculation.autoregressive_generator import AutoRegressiveGenerationStrategy
from self_speculation.llama_model_utils import decode_next_token, forward

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdapterEnhancedAutoregressiveStrategy(AutoRegressiveGenerationStrategy):
    """
    Enhances the autoregressive generation strategy with a DeBERTa adapter.
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

        # Convert input_ids list to tensor
        input_ids_tensor: torch.Tensor = torch.tensor([input_ids]).to(model.device)
        output_ids: List[int] = []

        # Save a reference to the model for the layer extractor
        self.layer_extractor.model = model
        
        for _ in range(generation_config.max_steps):
            # Regular full model forward pass
            model_output = forward(model, input_ids_tensor, past_key_values)
            logits = model_output.logits
            past_key_values = model_output.past_key_values
            
            # Extract hidden states from target layer using layer extractor
            with torch.no_grad():
                hidden_states = self.layer_extractor.get_layer_output(input_ids_tensor)
                
                # Get the last token's hidden state
                last_hidden = hidden_states[:, -1, :]
                
                # Pass through adapter to get enhanced logits
                adapter_logits = self.adapter(last_hidden)
                
                # Combine with original logits
                alpha = 0.7  # Weight for adapter logits (adjust as needed)
                enhanced_logits = alpha * adapter_logits + (1 - alpha) * logits
                
                if logits_processors:
                    enhanced_logits = logits_processors(input_ids_tensor, enhanced_logits)
                
                # Decode next token
                next_token, _ = decode_next_token(
                    logits=enhanced_logits,
                    token_idx=-1, 
                    sample=generation_config.sample,
                    temperature=generation_config.temperature,
                    top_k=generation_config.top_k,
                    top_p=generation_config.top_p
                )
            
            if streamer:
                streamer.put(next_token)
                
            next_token = next_token.item()
            
            if next_token in eos_token_ids:
                break
                
            if stopping_criteria:
                if torch.all(stopping_criteria(input_ids_tensor, scores=None)):
                    break
                    
            output_ids.append(next_token)
            # Use KV cache for efficiency - only need to process the new token
            input_ids_tensor = torch.tensor([[next_token]]).to(input_ids_tensor)

        return GenerationStrategyResult(
            predicted_tokens=output_ids,
            acceptance_rate=None,  # Not applicable for autoregressive
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
    parser = argparse.ArgumentParser(description="Run adapter-enhanced inference on RACE dataset")
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
    parser.add_argument("--target_layer", type=int, default=8, help="Layer to extract for adapter")
    parser.add_argument("--output", type=str, default=None, help="Output file for results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                      help="Device to run inference on")
    parser.add_argument("--compare", action="store_true", help="Compare with regular autoregressive generation")
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
    target_layer = adapter_config.get("target_layer", args.target_layer)
    layer_extractor = LlamaLayerExtractor(model, target_layer=target_layer)
    
    # Create enhanced autoregressive strategy
    enhanced_strategy = AdapterEnhancedAutoregressiveStrategy(
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
        generation_strategy="autoregressive",  # This is just a label
        sample=args.temperature > 0,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )
    
    # Initialize regular generator for comparison if requested
    if args.compare:
        regular_strategy = AutoRegressiveGenerationStrategy()
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
            
            # Generate with regular model if comparison is requested
            if args.compare:
                regular_result = regular_generator.generate(
                    prompt=example['prompt'],
                    generation_config=generation_config
                )
                
                regular_text = regular_result.decoded_prediction
                regular_answer = extract_answer_letter(regular_text)
                regular_predictions.append(regular_answer)
            
            # Store result
            result_item = {
                'id': example['id'],
                'prompt': example['prompt'],
                'ground_truth': example['answer'],
                'enhanced_prediction': enhanced_answer,
                'enhanced_text': enhanced_text,
            }
            
            if args.compare:
                result_item.update({
                    'regular_prediction': regular_answer,
                    'regular_text': regular_text,
                })
            
            results.append(result_item)
            
            # Log progress
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(examples)} examples")
                
                # Calculate and log current accuracy
                current_enhanced_acc = evaluate_accuracy(enhanced_predictions, ground_truth[:len(enhanced_predictions)])
                logger.info(f"Current enhanced accuracy: {current_enhanced_acc:.4f}")
                
                if args.compare and regular_predictions:
                    current_regular_acc = evaluate_accuracy(regular_predictions, ground_truth[:len(regular_predictions)])
                    logger.info(f"Current regular accuracy: {current_regular_acc:.4f}")
                
        except Exception as e:
            logger.error(f"Error processing example {example['id']}: {e}")
            # Add a placeholder for failed examples
            enhanced_predictions.append(None)
            if args.compare:
                regular_predictions.append(None)
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Calculate final accuracy
    enhanced_accuracy = evaluate_accuracy(
        [p for p in enhanced_predictions if p is not None], 
        [gt for p, gt in zip(enhanced_predictions, ground_truth) if p is not None]
    )
    
    if args.compare:
        regular_accuracy = evaluate_accuracy(
            [p for p in regular_predictions if p is not None],
            [gt for p, gt in zip(regular_predictions, ground_truth) if p is not None]
        )
    
    # Print results
    logger.info("\n======= EVALUATION RESULTS =======")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Number of examples: {len(examples)}")
    logger.info(f"Total time: {total_time:.2f} seconds")
    logger.info(f"Average time per example: {total_time / len(examples):.2f} seconds")
    logger.info(f"Enhanced model accuracy: {enhanced_accuracy:.4f}")
    
    if args.compare:
        logger.info(f"Regular model accuracy: {regular_accuracy:.4f}")
        logger.info(f"Accuracy improvement: {enhanced_accuracy - regular_accuracy:.4f}")
    
    # Save results to CSV if output path is provided
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        with open(args.output, 'w', newline='') as csvfile:
            fieldnames = ['id', 'prompt', 'ground_truth', 'enhanced_prediction', 'enhanced_text']
            
            if args.compare:
                fieldnames.extend(['regular_prediction', 'regular_text'])
                
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
            "enhanced_accuracy": enhanced_accuracy,
            "total_time": total_time,
            "avg_time_per_example": total_time / len(examples),
        }
        
        if args.compare:
            summary.update({
                "regular_accuracy": regular_accuracy,
                "accuracy_improvement": enhanced_accuracy - regular_accuracy,
            })
            
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Summary statistics saved to {summary_path}")
    
    # Clean up
    layer_extractor.remove_hooks()
    logger.info("Evaluation complete!")

if __name__ == "__main__":
    main()
