import os
import pandas as pd
import numpy as np
import argparse
from tabulate import tabulate

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
            
            # Filter for Accuracy metrics only
            accuracy_df = df[df['metric'].str.startswith('Accuracy-')].copy()
            
            # Convert 'value' column to numeric
            accuracy_df['value'] = accuracy_df['value'].apply(safe_float)
            
            # Get results for gradients, features, labels at epoch 0
            epoch0_results = accuracy_df[accuracy_df['epoch'] == 0]
            for attack in ['gradients', 'features', 'labels']:
                accuracy = epoch0_results[epoch0_results['metric'] == f'Accuracy-{attack}']['value'].values
                if len(accuracy) > 0:
                    results.append({'attack': attack, 'accuracy': accuracy[0], 'seed': subdir})
            
            # Get results for output_server and forward_values at last epoch
            last_epoch = accuracy_df['epoch'].max()
            last_epoch_results = accuracy_df[accuracy_df['epoch'] == last_epoch]
            for attack in ['output_server', 'forward_values']:
                accuracy = last_epoch_results[last_epoch_results['metric'] == f'Accuracy-{attack}']['value'].values
                if len(accuracy) > 0:
                    attack_name = 'Inter-Reps' if attack == 'forward_values' else 'Prediction output'
                    results.append({'attack': attack_name, 'accuracy': accuracy[0], 'seed': subdir})
    
    return pd.DataFrame(results)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate Table 4 results')
    parser.add_argument('--experiment_name', type=str, default="Table 4 results Cora",
                        help='Name of the experiment (e.g., "Table 4 results Cora")')
    args = parser.parse_args()

    # Construct the experiment directory path
    experiment_dir = os.path.join('..', 'logs', args.experiment_name)

    # Check if the experiment directory exists
    if not os.path.exists(experiment_dir):
        print(f"Error: The experiment directory '{experiment_dir}' does not exist.")
        print(f"Please run the experiment first using scripts/run_table4.sh")
        return

    # Parse results
    results_df = parse_results(experiment_dir)

    # Check if results are empty
    if results_df.empty:
        print(f"Error: No valid results found in '{experiment_dir}'.")
        return

    # Compute mean and std
    summary = results_df.groupby('attack')['accuracy'].agg(['mean', 'std']).reset_index()

    # Format the results
    summary['result'] = summary.apply(lambda row: f"{row['mean']*100:.2f} ± {row['std']*100:.2f}", axis=1)

    # Define the desired order
    order = ['gradients', 'Inter-Reps', 'features', 'labels', 'Prediction output']

    # Reorder the rows
    summary = summary.set_index('attack').loc[order].reset_index()

    # Get the dataset name from the last word of the experiment name
    dataset_name = args.experiment_name.split()[-1]

    # Display the results
    print(f"---------------- {dataset_name} ----------------")
    table = tabulate(summary[['attack', 'result']], headers=['Attack', 'Accuracy (%)'], tablefmt='pipe')
    print(table)

    # Save the results to a file
    output_file = f'table4_{dataset_name.lower()}.txt'
    with open(output_file, 'w') as f:
        f.write(f"---------------- {dataset_name} ----------------\n")
        f.write(table)
    print(f"\nResults saved to '{output_file}'")

if __name__ == "__main__":
    main()
