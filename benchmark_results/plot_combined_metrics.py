import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the CSV data
df = pd.read_csv('benchmark_results/combined_metrics_full.csv')

# Create output directory for plots
os.makedirs('plots', exist_ok=True)

# Convert scientific notation to readable values
df['total_flops_readable'] = df['total_flops'] / 1e12  # Convert to TeraFLOPs
df['flops_per_token_readable'] = df['flops_per_token'] / 1e9  # Convert to GFLOPs
df['efficiency_readable'] = df['efficiency'] * 1e12  # Scale for readability

# Generate plots for each dataset
for dataset in df['dataset'].unique():
    dataset_df = df[df['dataset'] == dataset]

    # 1. Accuracy vs. FLOPs per token by strategy
    plt.figure(figsize=(12, 8))
    for strategy in dataset_df['strategy'].unique():
        strategy_df = dataset_df[dataset_df['strategy'] == strategy]
        plt.scatter(
            strategy_df['flops_per_token_readable'], 
            strategy_df['accuracy'], 
            label=strategy, 
            s=100, 
            alpha=0.7
        )
        
        # Add parameter labels for points
        for i, row in strategy_df.iterrows():
            if pd.notna(row['value']):
                plt.annotate(
                    f"{row['value']}", 
                    (row['flops_per_token_readable'], row['accuracy']),
                    textcoords="offset points",
                    xytext=(0,10),
                    ha='center'
                )

    plt.title(f'Accuracy vs. FLOPs per Token by Strategy ({dataset})')
    plt.xlabel('FLOPs per Token (GFLOPs)')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/accuracy_vs_flops_{dataset}.png', dpi=300)

    # 2. Efficiency by strategy
    plt.figure(figsize=(12, 8))
    strategies = dataset_df['strategy'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(strategies)))

    for i, strategy in enumerate(strategies):
        strategy_df = dataset_df[dataset_df['strategy'] == strategy].sort_values('value')
        
        if strategy == 'autoregressive':
            plt.bar(
                strategy, 
                strategy_df['efficiency_readable'].iloc[0], 
                label=strategy,
                color=colors[i],
                alpha=0.7
            )
        else:
            x_pos = np.arange(len(strategy_df))
            values = strategy_df['value'].astype(str)
            if strategy == 'layerdrop':
                values = [f"dr={val}" for val in values]
            else:
                values = [f"ht={val}" for val in values]
                
            plt.bar(
                [f"{strategy}_{val}" for val in values], 
                strategy_df['efficiency_readable'], 
                label=strategy if i == 0 else "",
                color=colors[i],
                alpha=0.7
            )

    plt.title(f'Efficiency (Accuracy per FLOPs) by Strategy ({dataset})')
    plt.xlabel('Strategy and Parameter')
    plt.ylabel('Efficiency (Accuracy/FLOPs) × 10¹²')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.tight_layout()
    plt.savefig(f'plots/efficiency_by_strategy_{dataset}.png', dpi=300)

    # 3. Accuracy and Computational Cost by Strategy
    fig, ax1 = plt.subplots(figsize=(14, 8))

    bar_width = 0.35
    index = np.arange(len(dataset_df))

    # Plot accuracy on left y-axis
    ax1.bar(index - bar_width/2, dataset_df['accuracy'], bar_width, label='Accuracy', color='royalblue')
    ax1.set_ylabel('Accuracy', color='royalblue')
    ax1.tick_params(axis='y', labelcolor='royalblue')
    ax1.set_ylim(0, 0.6)

    # Create second y-axis for FLOPs
    ax2 = ax1.twinx()
    ax2.bar(index + bar_width/2, dataset_df['total_flops_readable'], bar_width, label='Total FLOPs', color='salmon')
    ax2.set_ylabel('Total FLOPs (TeraFLOPs)', color='salmon')
    ax2.tick_params(axis='y', labelcolor='salmon')

    # Configure x-axis with strategy labels
    labels = []
    for _, row in dataset_df.iterrows():
        if row['strategy'] == 'autoregressive':
            labels.append('autoregressive')
        else:
            param_val = str(row['value'])
            labels.append(f"{row['strategy']}\n{row['parameter']}={param_val}")

    plt.title(f'Accuracy and Computational Cost by Strategy ({dataset})')
    plt.xticks(index, labels, rotation=45, ha='right')
    plt.tight_layout()

    # Create legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center')

    plt.savefig(f'plots/accuracy_and_flops_{dataset}.png', dpi=300)

    # 4. Latency comparison
    plt.figure(figsize=(12, 8))
    plt.bar(labels, dataset_df['time_per_token'], alpha=0.7, color='teal')
    plt.title(f'Time per Token by Strategy ({dataset})')
    plt.xlabel('Strategy')
    plt.ylabel('Time per Token (seconds)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.tight_layout()
    plt.savefig(f'plots/latency_comparison_{dataset}.png', dpi=300)

print("Separate plots for each dataset generated successfully in 'plots' directory.") 