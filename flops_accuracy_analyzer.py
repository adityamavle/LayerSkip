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
        best_efficiency = df.loc[df["accuracy"].idxmax()]
        plt.scatter(
            best_efficiency["estimated_flops"],
            best_efficiency["accuracy"],
            marker='*', s=200, color='red', 
            label="Best Efficiency", zorder=5
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
            
    def generate_report(self, output_path: str = "flops_accuracy_report.html", save_csv: bool = True):
        """Generate a comprehensive HTML report of the analysis."""
        df = self.create_dataframe()
        
        # Save DataFrame to CSV if requested
        if save_csv:
            csv_path = output_path.replace('.html', '.csv')
            self.save_dataframe(csv_path)
        
        # Group by dataset
        datasets = df["dataset"].unique()
        
        # Start HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>LayerSkip FLOPs vs Accuracy Analysis</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .strategy-autoregressive {{ color: #3498db; }}
                .strategy-self_speculative {{ color: #e74c3c; }}
                .strategy-layerdrop {{ color: #2ecc71; }}
                .strategy-depth_adaptive_sequence {{ color: #9b59b6; }}
                .best {{ font-weight: bold; background-color: #e8f4f8; }}
            </style>
        </head>
        <body>
            <h1>LayerSkip FLOPs vs Accuracy Analysis</h1>
            <p>Analysis of different generation strategies across datasets to evaluate FLOPs-Accuracy tradeoff.</p>
        """
        
        # Create summary table for all datasets
        html_content += """
            <h2>Summary Across All Datasets</h2>
            <table>
                <tr>
                    <th>Dataset</th>
                    <th>Best Strategy</th>
                    <th>Parameters</th>
                    <th>Accuracy</th>
                    <th>Estimated FLOPs</th>
                    <th>Efficiency (Acc/FLOPs)</th>
                </tr>
        """
        
        for dataset in datasets:
            dataset_df = df[df["dataset"] == dataset]
            
            # Find best efficiency (accuracy per FLOP)
            dataset_df["efficiency"] = dataset_df["accuracy"] / dataset_df["estimated_flops"]
            best_row = dataset_df.loc[dataset_df["efficiency"].idxmax()]
            
            html_content += f"""
                <tr>
                    <td>{dataset}</td>
                    <td class="strategy-{best_row['generation_strategy']}">{best_row['generation_strategy']}</td>
                    <td>"""
            
            # Add strategy-specific parameters
            if best_row["generation_strategy"] == "autoregressive":
                if best_row["exit_layer"] > 0:
                    html_content += f"exit_layer={best_row['exit_layer']}"
                else:
                    html_content += "Full model"
            elif best_row["generation_strategy"] == "self_speculative":
                html_content += f"exit_layer={best_row['exit_layer']}, num_speculations={best_row['num_speculations']}"
            elif best_row["generation_strategy"] == "layerdrop":
                html_content += f"dropout_rate={best_row['dropout_rate']}"
            elif best_row["generation_strategy"] == "depth_adaptive_sequence":
                html_content += f"halting_threshold={best_row['halting_threshold']}"
                
            html_content += f"""
                    </td>
                    <td>{best_row['accuracy']:.4f}</td>
                    <td>{best_row['estimated_flops']:.4f}</td>
                    <td>{best_row['efficiency']:.4f}</td>
                </tr>
            """
        
        html_content += """
            </table>
        """
        
        # Add detailed section for each dataset
        for dataset in datasets:
            dataset_df = df[df["dataset"] == dataset]
            
            html_content += f"""
            <h2>Dataset: {dataset}</h2>
            <table>
                <tr>
                    <th>Generation Strategy</th>
                    <th>Parameters</th>
                    <th>Accuracy</th>
                    <th>Estimated FLOPs</th>
                    <th>Efficiency (Acc/FLOPs)</th>
                </tr>
            """
            
            # Sort by efficiency
            dataset_df["efficiency"] = dataset_df["accuracy"] / dataset_df["estimated_flops"]
            dataset_df = dataset_df.sort_values("efficiency", ascending=False)
            
            # Flag for best efficiency
            best_efficiency = dataset_df["efficiency"].max()
            
            for _, row in dataset_df.iterrows():
                is_best = (row["efficiency"] == best_efficiency)
                
                html_content += f"""
                <tr class="{'best' if is_best else ''}">
                    <td class="strategy-{row['generation_strategy']}">{row['generation_strategy']}</td>
                    <td>"""
                
                # Add strategy-specific parameters
                if row["generation_strategy"] == "autoregressive":
                    if row["exit_layer"] > 0:
                        html_content += f"exit_layer={row['exit_layer']}"
                    else:
                        html_content += "Full model"
                elif row["generation_strategy"] == "self_speculative":
                    html_content += f"exit_layer={row['exit_layer']}, num_speculations={row['num_speculations']}"
                elif row["generation_strategy"] == "layerdrop":
                    html_content += f"dropout_rate={row['dropout_rate']}"
                elif row["generation_strategy"] == "depth_adaptive_sequence":
                    html_content += f"halting_threshold={row['halting_threshold']}"
                    
                html_content += f"""
                    </td>
                    <td>{row['accuracy']:.4f}</td>
                    <td>{row['estimated_flops']:.4f}</td>
                    <td>{row['efficiency']:.4f}</td>
                </tr>
                """
            
            html_content += """
            </table>
            """
            
            # Generate plot for this dataset
            plot_path = f"{dataset}_flops_vs_accuracy.png"
            self.plot_flops_vs_accuracy(dataset=dataset, save_path=plot_path)
            
            # Add plot to HTML
            html_content += f"""
            <div>
                <img src="{plot_path}" style="max-width: 100%; height: auto;">
            </div>
            """
        
        # Close HTML
        html_content += """
        </body>
        </html>
        """
        
        # Write HTML to file
        with open(output_path, "w") as f:
            f.write(html_content)
            
        print(f"Report generated at {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyze FLOPs vs Accuracy tradeoffs")
    parser.add_argument("--logs_dir", type=str, default="./logs", help="Directory containing benchmark logs")
    parser.add_argument("--flops_data", type=str, required=True, help="JSON file with FLOPs data")
    parser.add_argument("--output", type=str, default="flops_accuracy_report.html", help="Output report path")
    parser.add_argument("--save_csv", action="store_true", default=True, help="Save results DataFrame to CSV")
    
    args = parser.parse_args()
    
    analyzer = BenchmarkAnalyzer(logs_dir=args.logs_dir)
    analyzer.load_benchmark_results()
    analyzer.load_flops_data(args.flops_data)
    analyzer.compute_flops_estimates()
    
    # Save DataFrame to CSV explicitly if requested
    if args.save_csv:
        csv_path = args.output.replace('.html', '.csv')
        analyzer.save_dataframe(csv_path)
    
    # Generate HTML report
    analyzer.generate_report(output_path=args.output, save_csv=False)  # Already saved CSV above

if __name__ == "__main__":
    main() 