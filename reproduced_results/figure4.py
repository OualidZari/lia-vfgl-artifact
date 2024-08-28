import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import argparse

def safe_float(x):
    try:
        return float(x)
    except ValueError:
        return np.nan

def parse_results(experiment_dir):
    results = []
    for subdir in os.listdir(experiment_dir):
        csv_path = os.path.join(experiment_dir, subdir, 'attack_results.csv')
        config_path = os.path.join(experiment_dir, subdir, 'config.json')
        
        if os.path.exists(csv_path) and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                fraction_data_gcn = config.get('fraction_data_gcn', None)
            
            if fraction_data_gcn is None:
                continue
            
            df = pd.read_csv(csv_path)
            auc_df = df[df['metric'].str.startswith('AUC-')].copy()
            auc_df['value'] = auc_df['value'].apply(safe_float)
            
            for attack in ['gradients', 'forward_values', 'features']:
                attack_results = auc_df[auc_df['metric'] == f'AUC-{attack}']
                if not attack_results.empty:
                    max_auc = attack_results['value'].max()
                    attack_name = 'Inter-Reps' if attack == 'forward_values' else attack.capitalize()
                    results.append({
                        'attack': attack_name,
                        'max_auc': max_auc,
                        'seed': subdir,
                        'fraction_data_gcn': fraction_data_gcn
                    })
    
    return pd.DataFrame(results)

def custom_formatter(x, _):
    if int(x) == x:
        return str(int(x))
    else:
        return str(round(x, 1))

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Plot AUC vs. Feature Ratio for Gradient, Inter-Reps, and Feature Attacks')
    parser.add_argument('--experiment_name', type=str, default="Figure 4 results Cora",
                        help='Name of the experiment (e.g., "Figure 4 results Cora")')
    args = parser.parse_args()

    # Construct the experiment directory path
    experiment_dir = os.path.join('..', 'logs', args.experiment_name)

    # Check if the experiment directory exists
    if not os.path.exists(experiment_dir):
        print(f"Error: The experiment directory '{experiment_dir}' does not exist.")
        print(f"Please run the experiment first using scripts/run_figure4.sh")
        return

    # Parse results
    results_df = parse_results(experiment_dir)

    # Check if results are empty
    if results_df.empty:
        print(f"Error: No valid results found in '{experiment_dir}'.")
        return

    grouped_results = results_df.groupby(['attack', 'fraction_data_gcn'])['max_auc'].agg(['mean', 'std']).reset_index()

    fig, ax = plt.subplots(figsize=(6, 4))
    width = 0.25
    feature_ratios = (1 - grouped_results['fraction_data_gcn'].unique()) * 100
    x = np.arange(len(feature_ratios))

    colors = {'Gradients': 'blue', 'Inter-Reps': 'red', 'Features': 'orange'}
    for i, attack in enumerate(['Gradients', 'Inter-Reps', 'Features']):
        data = grouped_results[grouped_results['attack'] == attack].sort_values('fraction_data_gcn', ascending=False)
        ax.bar(x + i*width, data['mean'], width, yerr=data['std'], 
               label=attack, color=colors[attack], capsize=3)

    ax.set_xlabel('Feature ratio (%)')
    ax.set_ylabel('AUC')
    dataset = args.experiment_name.split()[-1]  # Get the last word of experiment_name
    ax.set_title(dataset, fontsize=14, y=1.12)
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'{ratio:.0f}%' for ratio in sorted(feature_ratios)])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)

    ax.set_ylim(0.5, 0.9)
    ax.grid(which='both', linestyle=':', linewidth=0.5)
    ax.yaxis.set_major_formatter(FuncFormatter(custom_formatter))
    plt.tight_layout()
    plt.savefig(f'feature_ratio_comparison_{dataset.lower()}.pdf', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
