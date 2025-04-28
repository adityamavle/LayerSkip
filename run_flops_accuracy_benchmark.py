import os
import sys
import argparse
import subprocess
import json
import time
import datetime
import logging

# Define the path to your virtual environment's Python interpreter
VENV_PYTHON = r"C:\Users\adity\CSE8803DLT-24Fall-Assignment2-Public\cuda_env\Scripts\python.exe"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the strategies and datasets to benchmark
STRATEGIES = [
    # Autoregressive with early exit
    {
        "name": "autoregressive_exit8",
        "config": {
            "generation_strategy": "autoregressive",
            "exit_layer": 8,
        }
    },
    # Autoregressive full model
    {
        "name": "autoregressive_full",
        "config": {
            "generation_strategy": "autoregressive",
        }
    },
    # Self-speculative
    {
        "name": "self_speculative_exit8_spec6",
        "config": {
            "generation_strategy": "self_speculative",
            "exit_layer": 8,
            "num_speculations": 6
        }
    },
    # LayerDrop
    {
        "name": "layerdrop_dropout0.2",
        "config": {
            "generation_strategy": "layerdrop",
            "dropout_rate": 0.2,
            "layerdrop_seed": 42
        }
    },
    # Depth Adaptive Sequence
    {
        "name": "depth_adaptive_sequence_threshold0.99",
        "config": {
            "generation_strategy": "depth_adaptive_sequence",
            "halting_threshold": 0.99
        }
    }
]

DATASETS = [
    "mmlu",
    "race_m",
    "race_h",
    "gsm8k",
    "math",
    "mbpp",
    "human_eval"
]

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("benchmark.log")
        ]
    )

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run FLOPs vs accuracy benchmark")
    parser.add_argument("--model", type=str, required=True, help="Path to model or model name")
    parser.add_argument("--datasets", nargs="+", default=["mmlu", "race_m", "race_h"], 
                        help="List of datasets to benchmark")
    parser.add_argument("--output_dir", type=str, default="./benchmark_results", 
                        help="Output directory for benchmark results")
    parser.add_argument("--num_samples", type=int, default=10, 
                        help="Number of samples per dataset")
    parser.add_argument("--n_shot", type=int, default=0, 
                        help="Number of examples for few-shot prompting")
    parser.add_argument("--flops_samples", type=int, default=10, help="Number of samples for FLOP measurements")
    parser.add_argument("--distributed", type=bool, default=False, help="Use distributed evaluation")
    parser.add_argument("--flops_datasets", type=str, nargs="+", default=["mmlu"], help="Datasets for FLOP measurement (default: mmlu)")
    parser.add_argument("--skip_benchmarks", action="store_true", help="Skip running benchmarks")
    parser.add_argument("--skip_flops", action="store_true", help="Skip measuring FLOPs")
    parser.add_argument("--device", type=str, default="cuda", help="Device for FLOP measurement")
    return parser.parse_args()

def run_benchmark(args, dataset, strategy_config, strategy_name):
    """Run benchmark for a specific dataset and strategy configuration."""
    output_dir = os.path.join(args.output_dir, dataset)
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"benchmark_{strategy_name}_{timestamp}.json")
    
    # Build command with all arguments
    cmd = [
        VENV_PYTHON,  # Use the virtual environment Python
        "benchmark.py",
        "--model", args.model,
        "--dataset", dataset,
        "--num_samples", str(args.num_samples),
        "--n_shot", str(args.n_shot),
        "--output_dir", output_dir
    ]
    
    # Add strategy-specific parameters
    for key, value in strategy_config.items():
        cmd.extend([f"--{key}", str(value)])
    
    logging.info(f"Running benchmark for {dataset} with {strategy_name}")
    logging.info(f"Command: {' '.join(cmd)}")
    
    # Run the benchmark process
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        logging.info(f"Benchmark completed for {dataset} with {strategy_name}")
        logging.debug(result.stdout)
        
        # Write stdout to a log file
        with open(f"{output_file}.log", "w") as f:
            f.write(result.stdout)
            
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running benchmark for {dataset} with {strategy_name}")
        logging.error(f"Exit code: {e.returncode}")
        logging.error(f"Stdout: {e.stdout}")
        logging.error(f"Stderr: {e.stderr}")
        return False

def measure_flops(model_path, output_dir, datasets=None, num_samples=10, device="cuda"):
    """Measure FLOPs for all strategies on the specified datasets."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter datasets if specified
    if not datasets:
        datasets = ["mmlu"]  # Default to just one dataset for FLOP measurement
    
    flops_results = {
        "model": model_path,
        "timestamp": datetime.datetime.now().isoformat(),
        "measurements": []
    }
    
    for dataset in datasets:
        logger.info(f"Measuring FLOPs for dataset={dataset}")
        
        # Build command
        cmd = [
            VENV_PYTHON, "measure_flops.py",
            "--model", model_path,
            "--dataset", dataset,
            "--output_dir", output_dir,
            "--num_samples", str(num_samples),
            "--device", device
        ]
        
        # Run the FLOP measurement
        logger.info(f"Command: {' '.join(cmd)}")
        start_time = time.time()
        
        try:
            process = subprocess.run(
                cmd, 
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout = process.stdout
            stderr = process.stderr
            exit_code = 0
        except subprocess.CalledProcessError as e:
            stdout = e.stdout
            stderr = e.stderr
            exit_code = e.returncode
        
        elapsed_time = time.time() - start_time
        
        # Log the output
        logger.info(f"FLOP measurement completed in {elapsed_time:.2f} seconds with exit code {exit_code}")
        
        if exit_code != 0:
            logger.error(f"FLOP measurement failed: {stderr}")
        
        # Record measurement run
        measurement_info = {
            "dataset": dataset,
            "exit_code": exit_code,
            "elapsed_time": elapsed_time,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        
        flops_results["measurements"].append(measurement_info)
    
    # Save summary of all FLOP measurements
    summary_path = os.path.join(output_dir, f"flops_measurement_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_path, "w") as f:
        json.dump(flops_results, f, indent=2)
    
    logger.info(f"FLOP measurement summary saved to {summary_path}")
    return summary_path

def generate_report(benchmark_summary, flops_summary, output_dir):
    """Generate a report combining benchmark and FLOP measurement results."""
    report_cmd = [
        VENV_PYTHON, "flops_accuracy_analyzer.py",
        "--logs_dir", output_dir,
        "--flops_data", os.path.join(output_dir, "flops_summary_mmlu.json"),
        "--output", os.path.join(output_dir, f"flops_accuracy_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
    ]
    
    logger.info(f"Generating report with command: {' '.join(report_cmd)}")
    
    try:
        process = subprocess.run(
            report_cmd, 
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        logger.info("Report generated successfully")
        logger.info(process.stdout)
    except subprocess.CalledProcessError as e:
        logger.error(f"Report generation failed: {e.stderr}")

def main():
    """Main function to run all benchmarks."""
    setup_logging()
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run benchmarks for each dataset and strategy
    results = {}
    for dataset in args.datasets:
        results[dataset] = {}
        for strategy in STRATEGIES:
            strategy_name = strategy["name"]
            strategy_config = strategy["config"]
            
            success = run_benchmark(args, dataset, strategy_config, strategy_name)
            results[dataset][strategy_name] = {"success": success}
    
    # Save summary of all runs
    summary_file = os.path.join(args.output_dir, f"benchmark_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_file, "w") as f:
        json.dump({
            "args": vars(args),
            "results": results
        }, f, indent=2)
    
    logging.info(f"Benchmark summary saved to {summary_file}")

    # Measure FLOPs
    if not args.skip_flops:
        measure_flops(
            model_path=args.model,
            output_dir=args.output_dir,
            datasets=args.flops_datasets,
            num_samples=args.flops_samples,
            device=args.device
        )

    # Generate report
    if results:
        generate_report(summary_file, None, args.output_dir)

if __name__ == "__main__":
    main() 