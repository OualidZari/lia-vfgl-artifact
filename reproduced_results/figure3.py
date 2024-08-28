import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            
            # Filter for AUC metrics only
            auc_df = df[df['metric'].str.startswith('AUC-')].copy()
            
            # Convert 'value' column to numeric
            auc_df['value'] = auc_df['value'].apply(safe_float)
            
            # Get results for gradients and forward_values (Inter-Reps) for all epochs
            for attack in ['gradients', 'forward_values']:
                attack_results = auc_df[auc_df['metric'] == f'AUC-{attack}']
                for _, row in attack_results.iterrows():
                    attack_name = 'Inter-Reps' if attack == 'forward_values' else 'Gradient'
                    results.append({
                        'attack': attack_name,
                        'epoch': row['epoch'],
                        'auc': row['value'],
                        'seed': subdir
                    })
    
    return pd.DataFrame(results)

def plot_data(data, color, label_prefix, plot_max_auc=False):
    mean_auc = data.groupby('epoch')['auc'].mean()
    sem_auc = data.groupby('epoch')['auc'].sem()
    epochs = mean_auc.index

    plt.plot(epochs, mean_auc, label=f'{label_prefix}', color=color)
    plt.fill_between(epochs, mean_auc - sem_auc, mean_auc + sem_auc, color=color, alpha=0.2)

    if plot_max_auc:
        max_auc = mean_auc.max()
        plt.axhline(y=max_auc, color=color, linestyle='--')

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Plot AUC vs. Epoch for Gradient and Inter-Reps Attacks')
    parser.add_argument('--experiment_name', type=str, default="Figure 3 results Cora",
                        help='Name of the experiment (e.g., "Figure 3 results Cora")')
    args = parser.parse_args()

    # Construct the experiment directory path
    experiment_dir = os.path.join('..', 'logs', args.experiment_name)

    # Check if the experiment directory exists
    if not os.path.exists(experiment_dir):
        print(f"Error: The experiment directory '{experiment_dir}' does not exist.")
        print(f"Please run the experiment first using scripts/run_figure3.sh")
        return

    # Parse results
    results_df = parse_results(experiment_dir)

    # Check if results are empty
    if results_df.empty:
        print(f"Error: No valid results found in '{experiment_dir}'.")
        return

    # Get the dataset name from the last word of the experiment name
    dataset_name = args.experiment_name.split()[-1]

    # Plot the results
    fig, ax = plt.subplots(figsize=(6, 4))
    for attack, color in [('Gradient', 'blue'), ('Inter-Reps', 'red')]:
        attack_data = results_df[results_df['attack'] == attack]
        plot_data(attack_data, color, attack, plot_max_auc=True)

    ax.set_xlabel('Training epoch')
    ax.set_ylabel('AUC')
    ax.set_title(f'AUC vs. Epoch for Gradient and Inter-Reps Attacks ({dataset_name})')
    ax.set_ylim(0.5, 0.9)
    ax.set_xlim(0, results_df['epoch'].max())  # Set x-axis limit to max epoch
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'feature_ratio_comparison_{dataset_name.lower()}.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Figure saved as 'feature_ratio_comparison_{dataset_name.lower()}.pdf'")

if __name__ == "__main__":
    main()