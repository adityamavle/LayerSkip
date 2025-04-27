#!/usr/bin/env python3
"""
Script to run inference with DeBERTa adapter enhanced autoregressive decoding.
"""

import argparse
import torch
import json
import logging
import os
import time
from typing import List, Optional, Tuple
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


def main():
    parser = argparse.ArgumentParser(description="Run adapter-enhanced autoregressive decoding")
    parser.add_argument("--model", type=str, required=True, help="Path to base LLaMA model")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to trained adapter weights")
    parser.add_argument("--adapter_config", type=str, required=True, help="Path to adapter config JSON")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt for generation")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum length of generated text")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--target_layer", type=int, default=8, help="Layer to extract for adapter")
    parser.add_argument("--output", type=str, default=None, help="Output file to save generation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to run inference on")
    parser.add_argument("--compare", action="store_true", 
                      help="Compare with regular autoregressive generation")
    
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
    
    # Record start time
    start_time = time.time()
    
    # Generate text with enhanced model
    logger.info("Generating with adapter-enhanced autoregressive decoding...")
    result = enhanced_generator.generate(
        prompt=args.prompt,
        generation_config=generation_config
    )
    
    # Record end time and calculate stats
    generation_time = time.time() - start_time
    
    # Display results
    print("\n======= ADAPTER-ENHANCED AUTOREGRESSIVE GENERATION =======")
    print(f"Prompt: {args.prompt}")
    print(f"\nGenerated text: {result.decoded_prediction}")
    print("\n===========================================================")
    
    # Print stats
    print(f"\nGeneration time: {generation_time:.2f} seconds")
    print(f"Tokens generated: {result.num_tokens_generated}")
    print(f"Tokens per second: {result.tokens_per_second:.2f}")
    
    # Compare with regular autoregressive if requested
    if args.compare:
        # Create regular autoregressive generator
        regular_strategy = AutoRegressiveGenerationStrategy()
        regular_generator = HuggingfaceLlamaGenerator(
            tokenizer=tokenizer,
            model=model,
            generation_strategy=regular_strategy
        )
        
        # Record start time
        start_time = time.time()
        
        # Generate with regular autoregressive
        logger.info("Generating with regular autoregressive decoding for comparison...")
        regular_result = regular_generator.generate(
            prompt=args.prompt,
            generation_config=generation_config
        )
        
        # Record end time
        regular_time = time.time() - start_time
        
        # Display comparison results
        print("\n======= REGULAR AUTOREGRESSIVE GENERATION =======")
        print(f"Generated text: {regular_result.decoded_prediction}")
        print("\n=================================================")
        
        # Print comparison stats
        print(f"\nRegular generation time: {regular_time:.2f} seconds")
        print(f"Regular tokens per second: {regular_result.tokens_per_second:.2f}")
        print(f"Speed comparison: {regular_time / generation_time:.2f}x")
    
    # Save to file if requested
    if args.output:
        with open(args.output, "w") as f:
            f.write(f"Prompt: {args.prompt}\n\n")
            f.write(f"Enhanced generated text: {result.decoded_prediction}\n\n")
            f.write(f"Generation time: {generation_time:.2f} seconds\n")
            f.write(f"Tokens generated: {result.num_tokens_generated}\n")
            f.write(f"Tokens per second: {result.tokens_per_second:.2f}\n")
            
            if args.compare:
                f.write("\n=== Comparison ===\n")
                f.write(f"Regular generated text: {regular_result.decoded_prediction}\n\n")
                f.write(f"Regular generation time: {regular_time:.2f} seconds\n")
                f.write(f"Regular tokens per second: {regular_result.tokens_per_second:.2f}\n")
                f.write(f"Speed comparison: {regular_time / generation_time:.2f}x\n")
                
        logger.info(f"Results saved to {args.output}")
    
    # Clean up
    layer_extractor.remove_hooks()

if __name__ == "__main__":
    main()
