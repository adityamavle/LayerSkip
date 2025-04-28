import os
import sys
import argparse
import logging
import subprocess
import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(cmd, description=None):
    """Run a command and handle errors gracefully"""
    if description:
        logger.info(f"{description}")
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            check=False,  # Don't raise exception on non-zero exit
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            logger.info(f"Command completed successfully (exit code 0)")
            return True, result.stdout
        else:
            logger.error(f"Command failed with exit code {result.returncode}")
            logger.error(f"Error output: {result.stderr}")
            return False, result.stderr
    
    except subprocess.TimeoutExpired:
        logger.error(f"Command timed out after 300 seconds")
        return False, "Timeout"
    except Exception as e:
        logger.error(f"Error running command: {e}")
        return False, str(e)

def main():
    parser = argparse.ArgumentParser(description="Run a simplified benchmark")
    parser.add_argument("--model", required=True, help="Path to model")
    parser.add_argument("--datasets", nargs="+", default=["mmlu"], help="Datasets to evaluate")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples per dataset")
    parser.add_argument("--output_dir", default="./benchmark_results", help="Output directory")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate timestamp for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Record results
    results = {
        "timestamp": timestamp,
        "model": args.model,
        "datasets": args.datasets,
        "num_samples": args.num_samples,
        "benchmark_results": {}
    }
    
    # Run benchmark for each dataset
    for dataset in args.datasets:
        logger.info(f"Running benchmark for dataset: {dataset}")
        dataset_dir = os.path.join(args.output_dir, dataset)
        os.makedirs(dataset_dir, exist_ok=True)
        
        dataset_results = []
        
        # Define strategies to test
        strategies = [
            {"name": "autoregressive_full", "args": ["--generation_strategy", "autoregressive"]},
            {"name": "autoregressive_exit8", "args": ["--generation_strategy", "autoregressive", "--exit_layer", "8"]},
            {"name": "self_speculative", "args": ["--generation_strategy", "self_speculative", "--exit_layer", "8", "--num_speculations", "6"]},
            {"name": "layerdrop", "args": ["--generation_strategy", "layerdrop", "--dropout_rate", "0.2", "--layerdrop_seed", "42"]},
            {"name": "depth_adaptive_sequence", "args": ["--generation_strategy", "depth_adaptive_sequence", "--halting_threshold", "0.99"]}
        ]
        
        # Run each strategy
        for strategy in strategies:
            cmd = [
                sys.executable,
                "benchmark.py",
                "--model", args.model,
                "--dataset", dataset,
                "--num_samples", str(args.num_samples),
                "--output_dir", dataset_dir,
                "--n_shot", "0"
            ] + strategy["args"]
            
            success, output = run_command(cmd, f"Running {strategy['name']} on {dataset}")
            
            dataset_results.append({
                "strategy": strategy["name"],
                "success": success,
                "output": output[:500] + "..." if len(output) > 500 else output
            })
            
        results["benchmark_results"][dataset] = dataset_results
    
    # Save results summary
    summary_file = os.path.join(args.output_dir, f"benchmark_summary_{timestamp}.json")
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Benchmark complete. Summary saved to {summary_file}")
    
if __name__ == "__main__":
    main() 