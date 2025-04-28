import os
import argparse
import logging
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from self_speculation.llama_model_utils import load_tokenizer_and_model
from flops_analysis.measure_flops import measure_flops
from self_speculation.evaluation import evaluate_model_on_datasets
from flops_analysis.flops_accuracy_analyzer import FlopsAccuracyAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run FLOPs-Accuracy Benchmark")
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to the model or huggingface model name"
    )
    parser.add_argument(
        "--datasets", type=str, nargs="+", default=["mmlu", "race_m", "race_h"],
        help="List of datasets to evaluate (default: mmlu race_m race_h)"
    )
    parser.add_argument(
        "--num_samples", type=int, default=50,
        help="Number of samples to evaluate for each dataset"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=32,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--exit_layers", type=str, default="4,8,16,24,32",
        help="Comma-separated list of exit layers to test for early exit"
    )
    parser.add_argument(
        "--num_speculations", type=str, default="2,4,6,8",
        help="Comma-separated list of speculation counts to test"
    )
    parser.add_argument(
        "--dropout_rates", type=str, default="0.1,0.2,0.3,0.4,0.5",
        help="Comma-separated list of dropout rates to test"
    )
    parser.add_argument(
        "--halting_thresholds", type=str, default="0.7,0.8,0.9,0.95",
        help="Comma-separated list of halting thresholds to test"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./benchmark_results",
        help="Directory to save benchmark results"
    )
    return parser.parse_args()

def parse_comma_separated_values(comma_str: str, value_type=int) -> List:
    """Parse comma-separated values into a list of specified type."""
    if comma_str:
        return [value_type(x) for x in comma_str.split(",")]
    return []

def run_benchmark(
    model_name_or_path: str,
    datasets: List[str],
    output_dir: str,
    num_samples: int = 50,
    max_new_tokens: int = 32,
    exit_layers_str: str = "4,8,16,24,32",
    num_speculations_str: str = "2,4,6,8",
    dropout_rates_str: str = "0.1,0.2,0.3,0.4,0.5",
    halting_thresholds_str: str = "0.7,0.8,0.9,0.95",
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Run the FLOPs-Accuracy benchmark on the specified datasets.
    
    Args:
        model_name_or_path: Path to model or Hugging Face model name
        datasets: List of dataset names to evaluate
        output_dir: Directory to save results
        num_samples: Number of samples to evaluate per dataset
        max_new_tokens: Maximum number of tokens to generate
        exit_layers_str: Comma-separated string of exit layers to test
        num_speculations_str: Comma-separated string of speculation counts to test
        dropout_rates_str: Comma-separated string of dropout rates to test
        halting_thresholds_str: Comma-separated string of halting thresholds to test
        device: Device to run on ("cuda" or "cpu")
        
    Returns:
        Dictionary with benchmark results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse comma-separated values
    exit_layers = parse_comma_separated_values(exit_layers_str)
    num_speculations = parse_comma_separated_values(num_speculations_str)
    dropout_rates = parse_comma_separated_values(dropout_rates_str, float)
    halting_thresholds = parse_comma_separated_values(halting_thresholds_str, float)
    
    # Generate timestamp for this benchmark run
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    benchmark_id = f"benchmark_{timestamp}"
    
    # Prepare output files
    flops_summary_file = os.path.join(output_dir, f"flops_measurement_summary_{timestamp}.json")
    accuracy_summary_file = os.path.join(output_dir, f"accuracy_summary_{timestamp}.json")
    benchmark_summary_file = os.path.join(output_dir, f"benchmark_summary_{timestamp}.json")
    
    # 1. Measure FLOPs for all generation strategies on all datasets
    logger.info(f"Starting FLOPs measurements for all datasets: {datasets}")
    flops_results = {}
    
    for dataset_name in datasets:
        logger.info(f"Measuring FLOPs for dataset: {dataset_name}")
        
        flops_data = measure_flops(
            model_name_or_path=model_name_or_path,
            dataset_name=dataset_name,
            output_dir=output_dir,
            num_samples=5,  # Use fewer samples for FLOPs measurement
            max_new_tokens=max_new_tokens,
            device=device
        )
        
        flops_results[dataset_name] = flops_data
    
    # Save FLOPs summary
    with open(flops_summary_file, "w") as f:
        json.dump(flops_results, f, indent=2)
    
    logger.info(f"FLOPs measurements complete. Summary saved to {flops_summary_file}")
    
    # 2. Evaluate accuracy for all generation strategies on all datasets
    logger.info(f"Starting accuracy evaluation for all datasets: {datasets}")
    
    # Load model and tokenizer once
    logger.info(f"Loading model {model_name_or_path}")
    tokenizer, model = load_tokenizer_and_model(model_name_or_path, device=device)
    
    # Get number of layers
    num_layers = model.config.num_hidden_layers
    
    # Only use exit layers that are valid for this model
    valid_exit_layers = [layer for layer in exit_layers if layer <= num_layers]
    if not valid_exit_layers:
        # If no valid exit layers specified, use half the layers
        valid_exit_layers = [num_layers // 2]
    
    accuracy_results = {}
    
    # 2.1 Evaluate autoregressive (full model)
    logger.info("Evaluating autoregressive (full model)")
    auto_full_results = evaluate_model_on_datasets(
        model=model,
        tokenizer=tokenizer,
        dataset_names=datasets,
        num_samples=num_samples,
        max_new_tokens=max_new_tokens,
        strategy="autoregressive",
        strategy_params={},
        device=device
    )
    accuracy_results["autoregressive_full"] = auto_full_results
    
    # 2.2 Evaluate autoregressive with early exit
    for exit_layer in valid_exit_layers:
        logger.info(f"Evaluating autoregressive with exit_layer={exit_layer}")
        auto_exit_results = evaluate_model_on_datasets(
            model=model,
            tokenizer=tokenizer,
            dataset_names=datasets,
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
            strategy="autoregressive",
            strategy_params={"exit_layer": exit_layer},
            device=device
        )
        accuracy_results[f"autoregressive_exit{exit_layer}"] = auto_exit_results
    
    # 2.3 Evaluate self-speculative
    for exit_layer in valid_exit_layers:
        for num_spec in num_speculations:
            logger.info(f"Evaluating self-speculative with exit_layer={exit_layer}, num_speculations={num_spec}")
            self_spec_results = evaluate_model_on_datasets(
                model=model,
                tokenizer=tokenizer,
                dataset_names=datasets,
                num_samples=num_samples,
                max_new_tokens=max_new_tokens,
                strategy="self_speculative",
                strategy_params={
                    "exit_layer": exit_layer,
                    "num_speculations": num_spec
                },
                device=device
            )
            accuracy_results[f"self_speculative_exit{exit_layer}_spec{num_spec}"] = self_spec_results
    
    # 2.4 Evaluate LayerDrop
    for dropout_rate in dropout_rates:
        logger.info(f"Evaluating LayerDrop with dropout_rate={dropout_rate}")
        layerdrop_results = evaluate_model_on_datasets(
            model=model,
            tokenizer=tokenizer,
            dataset_names=datasets,
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
            strategy="layerdrop",
            strategy_params={"dropout_rate": dropout_rate},
            device=device
        )
        accuracy_results[f"layerdrop_dropout{dropout_rate}"] = layerdrop_results
    
    # 2.5 Evaluate Depth Adaptive Sequence
    for halting_threshold in halting_thresholds:
        logger.info(f"Evaluating Depth Adaptive Sequence with halting_threshold={halting_threshold}")
        das_results = evaluate_model_on_datasets(
            model=model,
            tokenizer=tokenizer,
            dataset_names=datasets,
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
            strategy="depth_adaptive_sequence",
            strategy_params={"halting_threshold": halting_threshold},
            device=device
        )
        accuracy_results[f"depth_adaptive_sequence_threshold{halting_threshold}"] = das_results
    
    # Save accuracy summary
    with open(accuracy_summary_file, "w") as f:
        json.dump(accuracy_results, f, indent=2)
    
    logger.info(f"Accuracy evaluation complete. Summary saved to {accuracy_summary_file}")
    
    # 3. Analyze and combine FLOPs and accuracy data
    logger.info("Analyzing FLOPs and accuracy data")
    
    # Create combined benchmark results dictionary
    benchmark_results = {
        "model": model_name_or_path,
        "datasets": datasets,
        "num_samples": num_samples,
        "max_new_tokens": max_new_tokens,
        "timestamp": timestamp,
        "flops_data": flops_results,
        "accuracy_data": accuracy_results
    }
    
    # Save benchmark summary
    with open(benchmark_summary_file, "w") as f:
        json.dump(benchmark_results, f, indent=2)
    
    logger.info(f"Benchmark summary saved to {benchmark_summary_file}")
    
    # 4. Generate visual report
    analyzer = FlopsAccuracyAnalyzer(output_dir)
    report_path = analyzer.generate_report(benchmark_id, save_csv=True)
    
    logger.info(f"Benchmark report generated: {report_path}")
    
    return benchmark_results

def main():
    args = setup_arguments()
    
    run_benchmark(
        model_name_or_path=args.model,
        datasets=args.datasets,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        exit_layers_str=args.exit_layers,
        num_speculations_str=args.num_speculations,
        dropout_rates_str=args.dropout_rates,
        halting_thresholds_str=args.halting_thresholds
    )

if __name__ == "__main__":
    main() 