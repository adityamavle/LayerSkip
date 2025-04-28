import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional
import argparse
import re

class BenchmarkAnalyzer:
    """
    Analyzes benchmark results to report FLOPs vs Accuracy tradeoffs
    for different generation strategies.
    """
    
    def __init__(self, logs_dir: str = "./logs"):
        self.logs_dir = logs_dir
        self.results = []
        self.flops_data = {}
        
    def load_benchmark_results(self, pattern: str = "benchmark_*.json"):
        """Load all benchmark results matching the pattern."""
        result_files = glob.glob(os.path.join(self.logs_dir, pattern))
        
        for file_path in result_files:
            try:
                with open(file_path, 'r') as f:
                    # Each file has multiple JSON objects concatenated
                    content = f.read()
                    # Split by }{ to separate JSON objects
                    json_parts = re.split(r'}\s*{', content)
                    
                    # Process first part (args)
                    first_part = json_parts[0] + '}'
                    args = json.loads(first_part)
                    
                    # Process second part (benchmark_arguments)
                    second_part = '{' + json_parts[1] + '}'
                    benchmark_arguments = json.loads(second_part)
                    
                    # Process third part (generation_config)
                    third_part = '{' + json_parts[2] + '}'
                    generation_config = json.loads(third_part)
                    
                    # Process fourth part (metrics)
                    fourth_part = '{' + json_parts[3]
                    metrics = json.loads(fourth_part)
                    
                    # Extract accuracy and other relevant metrics
                    accuracy = self._extract_accuracy(metrics)
                    
                    # Combine into a result entry
                    result = {
                        "dataset": benchmark_arguments.get("dataset", "unknown"),
                        "generation_strategy": generation_config.get("generation_strategy", "unknown"),
                        "exit_layer": generation_config.get("exit_layer", -1),
                        "num_speculations": generation_config.get("num_speculations", -1),
                        "dropout_rate": generation_config.get("dropout_rate", 0.0),
                        "halting_threshold": generation_config.get("halting_threshold", 0.0),
                        "accuracy": accuracy,
                        "file_path": file_path,
                        "metrics": metrics
                    }
                    
                    self.results.append(result)
                    print(f"Loaded results from {file_path}")
                    
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    def _extract_accuracy(self, metrics):
        """Extract accuracy from metrics dictionary."""
        try:
            # Try to get accuracy from predicted_text
            if "predicted_text" in metrics and "accuracy" in metrics["predicted_text"]:
                return metrics["predicted_text"]["accuracy"]
            
            # Or from exact_match which is also used for accuracy
            if "predicted_text" in metrics and "exact_match" in metrics["predicted_text"]:
                return metrics["predicted_text"]["exact_match"]
            
            return None
        except Exception:
            return None
    
    def load_flops_data(self, flops_file: str):
        """Load FLOPs data from file."""
        try:
            with open(flops_file, 'r') as f:
                self.flops_data = json.load(f)
            print(f"Loaded FLOPs data from {flops_file}")
        except Exception as e:
            print(f"Error loading FLOPs data: {e}")
    
    def compute_flops_estimates(self):
        """Compute FLOPs estimates for each generation strategy."""
        base_flops = self.flops_data.get("base_flops_per_token", 1.0)
        
        # Add FLOPs estimate to each result
        for result in self.results:
            strategy = result["generation_strategy"]
            
            if strategy == "autoregressive":
                if result["exit_layer"] > 0:
                    # Early exit at specified layer
                    layer_fraction = result["exit_layer"] / self.flops_data.get("total_layers", 32)
                    result["estimated_flops"] = base_flops * layer_fraction
                else:
                    # Full model
                    result["estimated_flops"] = base_flops
                    
            elif strategy == "self_speculative":
                # Speculative execution: exit_layer for draft + full layer verification
                # but with acceptance rate determining how often we use full model
                spec_layers = result["exit_layer"]
                num_spec = result["num_speculations"]
                acceptance_rate = result.get("metrics", {}).get("acceptance_rate", {}).get("mean", 0.5)
                
                # Draft model cost + verification cost
                draft_flops = base_flops * (spec_layers / self.flops_data.get("total_layers", 32))
                
                # For accepted tokens: draft_flops only 
                # For rejected tokens: draft_flops + full model verification
                result["estimated_flops"] = draft_flops + (1 - acceptance_rate) * base_flops
                
            elif strategy == "layerdrop":
                # LayerDrop randomly drops layers based on dropout_rate
                kept_fraction = 1.0 - result["dropout_rate"]
                result["estimated_flops"] = base_flops * kept_fraction
                
            elif strategy == "depth_adaptive_sequence":
                # Adaptive strategies dynamically choose layers - use actual measured data if available
                # For theoretical estimate, can use halting_threshold to approximate
                # Approximate based on halting threshold - higher threshold means more computation
                halt_factor = 0.5 + (result["halting_threshold"] / 2.0)  # Maps 0.0-1.0 to 0.5-1.0
                result["estimated_flops"] = base_flops * halt_factor
                
            else:
                # Default: full model
                result["estimated_flops"] = base_flops
    
    def create_dataframe(self):
        """Create pandas DataFrame from results for easier analysis."""
        df = pd.DataFrame(self.results)
        
        # Calculate efficiency (accuracy per FLOP)
        if 'estimated_flops' in df.columns and 'accuracy' in df.columns:
            df['efficiency'] = df['accuracy'] / df['estimated_flops']
            
        # Drop the metrics column as it contains nested data that doesn't work well in CSV
        if 'metrics' in df.columns:
            df = df.drop(columns=['metrics'])
            
        return df
    
    def save_dataframe(self, output_path: Optional[str] = None):
        """Save results DataFrame to CSV file."""
        df = self.create_dataframe()
        
        if output_path is None:
            # Generate default filename
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"flops_accuracy_results_{timestamp}.csv"
        
        # Make sure output path has .csv extension
        if not output_path.endswith('.csv'):
            output_path += '.csv'
            
        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"Results DataFrame saved to {output_path}")
        
        return output_path
    
    def plot_flops_vs_accuracy(self, dataset: Optional[str] = None, save_path: Optional[str] = None):
        """Plot FLOPs vs Accuracy for different generation strategies."""
        df = self.create_dataframe()
        
        if dataset:
            df = df[df["dataset"] == dataset]
            
        if df.empty:
            print("No data to plot")
            return
            
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Define markers and colors for different strategies
        markers = {
            "autoregressive": "o",
            "self_speculative": "^", 
            "layerdrop": "s",
            "depth_adaptive_sequence": "D"
            # Removed depth_adaptive_token
        }
        
        # Group by generation strategy and plot
        for strategy, group in df.groupby("generation_strategy"):
            plt.scatter(
                group["estimated_flops"], 
                group["accuracy"],
                label=strategy,
                marker=markers.get(strategy, "*"),
                s=100,
                alpha=0.7
            )
            
            # Add annotations for key parameters
            for _, row in group.iterrows():
                if strategy == "autoregressive" and row["exit_layer"] > 0:
                    label = f"L={row['exit_layer']}"
                elif strategy == "self_speculative":
                    label = f"L={row['exit_layer']},S={row['num_speculations']}"
                elif strategy == "layerdrop":
                    label = f"D={row['dropout_rate']}"
                elif strategy == "depth_adaptive_sequence":
                    label = f"H={row['halting_threshold']}"
                else:
                    label = ""
                    
                plt.annotate(
                    label,
                    (row["estimated_flops"], row["accuracy"]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8
                )
        
        # Fit a Pareto frontier line
        if not df.empty:
            plt.plot(
                [min(df["estimated_flops"]) * 0.9, max(df["estimated_flops"]) * 1.1],
                [min(df["accuracy"]) * 0.9, max(df["accuracy"]) * 1.1],
                '--', color='gray', alpha=0.5, label="FLOPs-Accuracy Tradeoff"
            )
        
        # Add labels and title
        title = f"FLOPs vs Accuracy Tradeoff for {dataset}" if dataset else "FLOPs vs Accuracy Tradeoff"
        plt.title(title)
        plt.xlabel("Estimated FLOPs (relative to base model)")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add Pareto-optimal point
        best_efficiency = df.loc[df["efficiency"].idxmax()]
        plt.scatter(
            best_efficiency["estimated_flops"],
            best_efficiency["accuracy"],
            marker="*", 
            s=200, 
            color="green",
            label="Best Efficiency"
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {save_path}")
        
        plt.close()
    
    def generate_report(self, output_path: str = "flops_accuracy_report.html", save_csv: bool = True):
        """Generate HTML report showing FLOPs vs Accuracy tradeoff."""
        import datetime
        
        # Compute FLOPs estimates
        self.compute_flops_estimates()
        
        # Create DataFrame
        df = self.create_dataframe()
        
        # Handle empty DataFrame case
        if df.empty:
            print("No data available for generating report")
            with open(output_path, 'w') as f:
                f.write("<html><body><h1>No data available for analysis</h1></body></html>")
            return output_path
            
        # Check if dataset column exists
        if 'dataset' not in df.columns:
            # Add a default dataset column if missing
            df['dataset'] = 'unknown'
        
        # Save dataframe to CSV if requested
        if save_csv:
            csv_path = output_path.replace(".html", ".csv")
            self.save_dataframe(csv_path)
        
        # Start HTML
        html = ["<!DOCTYPE html>", "<html>", "<head>"]
        html.append("<title>FLOPs vs Accuracy Tradeoff Report</title>")
        html.append('<meta charset="UTF-8">')
        html.append('<meta name="viewport" content="width=device-width, initial-scale=1.0">')
        
        # Add Bootstrap CSS
        html.append('<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">')
        
        # Add custom CSS
        html.append("<style>")
        html.append("body { font-family: Arial, sans-serif; padding: 20px; }")
        html.append(".strategy-card { margin-bottom: 20px; }")
        html.append(".plot-container { margin-top: 30px; margin-bottom: 30px; }")
        html.append("</style>")
        
        html.append("</head>")
        html.append("<body>")
        
        # Header
        html.append('<div class="container">')
        html.append('<h1 class="my-4">FLOPs vs Accuracy Tradeoff Report</h1>')
        html.append(f'<p class="text-muted">Generated on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>')
        
        # Summary statistics
        html.append('<h2 class="mt-4">Summary</h2>')
        html.append('<div class="row">')
        
        # Count metrics
        num_datasets = len(df["dataset"].unique())
        num_strategies = len(df["generation_strategy"].unique())
        num_results = len(df)
        
        # Add summary cards
        html.append('<div class="col-md-4">')
        html.append('<div class="card text-center bg-light">')
        html.append('<div class="card-body">')
        html.append(f'<h5 class="card-title">{num_datasets}</h5>')
        html.append('<p class="card-text">Datasets</p>')
        html.append('</div></div></div>')
        
        html.append('<div class="col-md-4">')
        html.append('<div class="card text-center bg-light">')
        html.append('<div class="card-body">')
        html.append(f'<h5 class="card-title">{num_strategies}</h5>')
        html.append('<p class="card-text">Generation Strategies</p>')
        html.append('</div></div></div>')
        
        html.append('<div class="col-md-4">')
        html.append('<div class="card text-center bg-light">')
        html.append('<div class="card-body">')
        html.append(f'<h5 class="card-title">{num_results}</h5>')
        html.append('<p class="card-text">Total Results</p>')
        html.append('</div></div></div>')
        
        html.append('</div>') # End row
        
        # Per-dataset plots and analysis
        for dataset in df["dataset"].unique():
            dataset_df = df[df["dataset"] == dataset]
            
            # Skip datasets with insufficient data
            if len(dataset_df) < 2:
                continue
                
            html.append(f'<h2 class="mt-5">{dataset.upper()} Dataset</h2>')
            
            # Generate plot
            plot_path = f"{dataset}_flops_vs_accuracy.png"
            full_plot_path = os.path.join(os.path.dirname(output_path), plot_path)
            self.plot_flops_vs_accuracy(dataset=dataset, save_path=full_plot_path)
            
            # Add plot
            html.append('<div class="plot-container text-center">')
            html.append(f'<img src="{plot_path}" class="img-fluid" alt="FLOPs vs Accuracy for {dataset}">')
            html.append('</div>')
            
            # Add results table
            html.append('<h3 class="mt-4">Results</h3>')
            html.append('<div class="table-responsive">')
            html.append('<table class="table table-striped table-hover">')
            
            # Table header
            html.append('<thead><tr>')
            html.append('<th>Strategy</th>')
            html.append('<th>Accuracy</th>')
            html.append('<th>Estimated FLOPs</th>')
            html.append('<th>Efficiency (Acc/FLOPs)</th>')
            html.append('<th>Parameters</th>')
            html.append('</tr></thead>')
            
            # Table body
            html.append('<tbody>')
            
            # Sort by efficiency (descending)
            for _, row in dataset_df.sort_values("efficiency", ascending=False).iterrows():
                html.append('<tr>')
                html.append(f'<td>{row["generation_strategy"]}</td>')
                html.append(f'<td>{row["accuracy"]:.4f}</td>')
                html.append(f'<td>{row["estimated_flops"]:.4f}</td>')
                html.append(f'<td>{row["efficiency"]:.4f}</td>')
                
                # Parameters column
                params = []
                if row["generation_strategy"] == "autoregressive" and row["exit_layer"] > 0:
                    params.append(f"Exit Layer: {row['exit_layer']}")
                elif row["generation_strategy"] == "self_speculative":
                    params.append(f"Exit Layer: {row['exit_layer']}")
                    params.append(f"Speculations: {row['num_speculations']}")
                elif row["generation_strategy"] == "layerdrop":
                    params.append(f"Dropout Rate: {row['dropout_rate']}")
                elif row["generation_strategy"] == "depth_adaptive_sequence":
                    params.append(f"Halting Threshold: {row['halting_threshold']}")
                
                html.append(f'<td>{", ".join(params)}</td>')
                html.append('</tr>')
            
            html.append('</tbody>')
            html.append('</table>')
            html.append('</div>')
            
            # Add insights
            html.append('<h3 class="mt-4">Insights</h3>')
            
            # Find best strategies
            best_accuracy = dataset_df.loc[dataset_df["accuracy"].idxmax()]
            best_efficiency = dataset_df.loc[dataset_df["efficiency"].idxmax()]
            lowest_flops = dataset_df.loc[dataset_df["estimated_flops"].idxmin()]
            
            html.append('<div class="row">')
            
            # Best accuracy
            html.append('<div class="col-md-4">')
            html.append('<div class="card strategy-card">')
            html.append('<div class="card-header bg-primary text-white">Best Accuracy</div>')
            html.append('<div class="card-body">')
            html.append(f'<h5 class="card-title">{best_accuracy["generation_strategy"]}</h5>')
            html.append(f'<p class="card-text">Accuracy: {best_accuracy["accuracy"]:.4f}</p>')
            html.append(f'<p class="card-text">FLOPs: {best_accuracy["estimated_flops"]:.4f}</p>')
            
            if best_accuracy["generation_strategy"] == "autoregressive" and best_accuracy["exit_layer"] > 0:
                html.append(f'<p class="card-text">Exit Layer: {best_accuracy["exit_layer"]}</p>')
            elif best_accuracy["generation_strategy"] == "self_speculative":
                html.append(f'<p class="card-text">Exit Layer: {best_accuracy["exit_layer"]}, Speculations: {best_accuracy["num_speculations"]}</p>')
            elif best_accuracy["generation_strategy"] == "layerdrop":
                html.append(f'<p class="card-text">Dropout Rate: {best_accuracy["dropout_rate"]}</p>')
            elif best_accuracy["generation_strategy"] == "depth_adaptive_sequence":
                html.append(f'<p class="card-text">Halting Threshold: {best_accuracy["halting_threshold"]}</p>')
                
            html.append('</div></div></div>')
            
            # Best efficiency
            html.append('<div class="col-md-4">')
            html.append('<div class="card strategy-card">')
            html.append('<div class="card-header bg-success text-white">Best Efficiency</div>')
            html.append('<div class="card-body">')
            html.append(f'<h5 class="card-title">{best_efficiency["generation_strategy"]}</h5>')
            html.append(f'<p class="card-text">Efficiency: {best_efficiency["efficiency"]:.4f}</p>')
            html.append(f'<p class="card-text">Accuracy: {best_efficiency["accuracy"]:.4f}</p>')
            html.append(f'<p class="card-text">FLOPs: {best_efficiency["estimated_flops"]:.4f}</p>')
            
            if best_efficiency["generation_strategy"] == "autoregressive" and best_efficiency["exit_layer"] > 0:
                html.append(f'<p class="card-text">Exit Layer: {best_efficiency["exit_layer"]}</p>')
            elif best_efficiency["generation_strategy"] == "self_speculative":
                html.append(f'<p class="card-text">Exit Layer: {best_efficiency["exit_layer"]}, Speculations: {best_efficiency["num_speculations"]}</p>')
            elif best_efficiency["generation_strategy"] == "layerdrop":
                html.append(f'<p class="card-text">Dropout Rate: {best_efficiency["dropout_rate"]}</p>')
            elif best_efficiency["generation_strategy"] == "depth_adaptive_sequence":
                html.append(f'<p class="card-text">Halting Threshold: {best_efficiency["halting_threshold"]}</p>')
                
            html.append('</div></div></div>')
            
            # Lowest FLOPs
            html.append('<div class="col-md-4">')
            html.append('<div class="card strategy-card">')
            html.append('<div class="card-header bg-warning text-dark">Lowest FLOPs</div>')
            html.append('<div class="card-body">')
            html.append(f'<h5 class="card-title">{lowest_flops["generation_strategy"]}</h5>')
            html.append(f'<p class="card-text">FLOPs: {lowest_flops["estimated_flops"]:.4f}</p>')
            html.append(f'<p class="card-text">Accuracy: {lowest_flops["accuracy"]:.4f}</p>')
            
            if lowest_flops["generation_strategy"] == "autoregressive" and lowest_flops["exit_layer"] > 0:
                html.append(f'<p class="card-text">Exit Layer: {lowest_flops["exit_layer"]}</p>')
            elif lowest_flops["generation_strategy"] == "self_speculative":
                html.append(f'<p class="card-text">Exit Layer: {lowest_flops["exit_layer"]}, Speculations: {lowest_flops["num_speculations"]}</p>')
            elif lowest_flops["generation_strategy"] == "layerdrop":
                html.append(f'<p class="card-text">Dropout Rate: {lowest_flops["dropout_rate"]}</p>')
            elif lowest_flops["generation_strategy"] == "depth_adaptive_sequence":
                html.append(f'<p class="card-text">Halting Threshold: {lowest_flops["halting_threshold"]}</p>')
                
            html.append('</div></div></div>')
            
            html.append('</div>') # End row
        
        # Footer
        html.append('<footer class="mt-5 text-center text-muted">')
        html.append('<p>FLOPs-Accuracy Analysis Tool</p>')
        html.append('</footer>')
        
        html.append('</div>') # End container
        html.append('</body>')
        html.append('</html>')
        
        # Write HTML to file
        with open(output_path, 'w') as f:
            f.write('\n'.join(html))
            
        print(f"Report generated and saved to {output_path}")
        return output_path

def main():
    parser = argparse.ArgumentParser(description='Analyze FLOPs vs Accuracy tradeoff')
    parser.add_argument('--logs_dir', type=str, default='./benchmark_results', 
                        help='Directory containing benchmark results')
    parser.add_argument('--flops_data', type=str, required=True,
                        help='Path to FLOPs summary file')
    parser.add_argument('--output', type=str, default='flops_accuracy_report.html',
                        help='Output path for HTML report')
    parser.add_argument('--save_csv', action='store_true', default=True,
                        help='Save dataframe to CSV file')
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = BenchmarkAnalyzer(logs_dir=args.logs_dir)
    
    # Load benchmark results and FLOPs data
    analyzer.load_benchmark_results()
    analyzer.load_flops_data(args.flops_data)
    
    # Generate report
    analyzer.generate_report(output_path=args.output, save_csv=args.save_csv)

if __name__ == "__main__":
    main() 