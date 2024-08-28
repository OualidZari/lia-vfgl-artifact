import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from tabulate import tabulate
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
        metrics_path = os.path.join(experiment_dir, subdir, 'metrics.csv')
        
        if os.path.exists(csv_path) and os.path.exists(config_path) and os.path.exists(metrics_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                label_defense_budget = config.get('label_defense_budget', None)
            
            if label_defense_budget is None:
                continue
            
            df = pd.read_csv(csv_path)
            accuracy_df = df[df['metric'].str.startswith('Accuracy-')].copy()
            accuracy_df['value'] = accuracy_df['value'].apply(safe_float)
            
            metrics_df = pd.read_csv(metrics_path)
            last_epoch = metrics_df['epoch'].max()
            test_accuracy_row = metrics_df[(metrics_df['epoch'] == last_epoch) & (metrics_df['metric'] == 'accuracy_test')]
            
            if test_accuracy_row.empty:
                print(f"Warning: No test accuracy found for {subdir}")
                continue
            
            test_accuracy = test_accuracy_row['value'].iloc[0]
            
            run_result = {
                'label_defense_budget': label_defense_budget,
                'seed': subdir,
                'test_accuracy': test_accuracy
            }
            
            for attack in ['gradients', 'forward_values', 'output_server', 'labels']:
                attack_results = accuracy_df[accuracy_df['metric'] == f'Accuracy-{attack}']
                if not attack_results.empty:
                    if attack == 'gradients':
                        accuracy = attack_results['value'].iloc[0]  # First epoch for gradient attack
                    else:
                        accuracy = attack_results['value'].iloc[-1]  # Last epoch for other attacks
                    attack_name = 'Inter-Reps' if attack == 'forward_values' else 'Gradients' if attack == 'gradients' else 'Prediction Output' if attack == 'output_server' else 'Label'
                    run_result[f'{attack_name}_accuracy'] = accuracy
            
            results.append(run_result)
    
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description='Generate plots and tables for label randomization defense results')
    parser.add_argument('--experiment_name', type=str, default="Defense Label Randomization results Cora",
                        help='Name of the experiment (e.g., "Defense Label Randomization results Cora")')
    args = parser.parse_args()

    experiment_dir = os.path.join('..', 'logs', args.experiment_name)

    if not os.path.exists(experiment_dir):
        print(f"Error: The experiment directory '{experiment_dir}' does not exist.")
        print(f"Please run the experiment first using scripts/run_defense_label_rand.sh")
        return

    results_df = parse_results(experiment_dir)

    if results_df.empty:
        print("No valid results found. Please check the experiment directory and data.")
        print(f"Please run the experiment first using scripts/run_defense_label_rand.sh")
        return

    # Group results by label_defense_budget and calculate mean and sem for all metrics
    grouped_results = results_df.groupby('label_defense_budget').agg({
        'Gradients_accuracy': ['mean', 'sem'],
        'Inter-Reps_accuracy': ['mean', 'sem'],
        'Prediction Output_accuracy': ['mean', 'sem'],
        'Label_accuracy': ['mean', 'sem'],
        'test_accuracy': ['mean', 'sem']
    }).reset_index()

    # Flatten column names
    grouped_results.columns = ['_'.join(col).strip() for col in grouped_results.columns.values]

    # Rename columns for clarity
    grouped_results = grouped_results.rename(columns={
        'label_defense_budget_': 'label_defense_budget',
        'Gradients_accuracy_mean': 'Gradients_mean',
        'Gradients_accuracy_sem': 'Gradients_sem',
        'Inter-Reps_accuracy_mean': 'Inter-Reps_mean',
        'Inter-Reps_accuracy_sem': 'Inter-Reps_sem',
        'Prediction Output_accuracy_mean': 'Prediction Output_mean',
        'Prediction Output_accuracy_sem': 'Prediction Output_sem',
        'Label_accuracy_mean': 'Label_mean',
        'Label_accuracy_sem': 'Label_sem',
        'test_accuracy_mean': 'test_accuracy_mean',
        'test_accuracy_sem': 'test_accuracy_sem'
    })

    # Sort by label_defense_budget in ascending order
    grouped_results = grouped_results.sort_values('label_defense_budget', ascending=True)

    # Add data for no defense (Budget = 0)
    no_defense_data = {
        'label_defense_budget': 0,
        'Gradients_mean': 0.8171,
        'Gradients_sem': 0.0021,
        'Inter-Reps_mean': 0.6577,
        'Inter-Reps_sem': 0.0119,
        'Prediction Output_mean': 0.8014,
        'Prediction Output_sem': 0.0050,
        'Label_mean': 0.8174,
        'Label_sem': 0.0015,
        'test_accuracy_mean': 0.8397,
        'test_accuracy_sem': 0.0075
    }

    # Add no defense data to grouped_results
    grouped_results = pd.concat([pd.DataFrame([no_defense_data]), grouped_results], ignore_index=True)
    grouped_results = grouped_results.sort_values('label_defense_budget', ascending=True)

    # Format the table
    table_data = []
    for _, row in grouped_results.iterrows():
        table_data.append([
            f"{row['label_defense_budget']:.2f}",
            f"{row['Gradients_mean']*100:.2f} ± {row['Gradients_sem']*100:.2f}",
            f"{row['Inter-Reps_mean']*100:.2f} ± {row['Inter-Reps_sem']*100:.2f}",
            f"{row['Prediction Output_mean']*100:.2f} ± {row['Prediction Output_sem']*100:.2f}",
            f"{row['Label_mean']*100:.2f} ± {row['Label_sem']*100:.2f}",
            f"{row['test_accuracy_mean']*100:.2f} ± {row['test_accuracy_sem']*100:.2f}"
        ])

    # Create the table
    table = tabulate(table_data, headers=["Label Defense Budget", "Gradients", "Inter-Reps", "Prediction Output", "Label", "Test Accuracy"], 
                     tablefmt="pipe", floatfmt=".2f")

    print(table)

    # Plotting
    plt.figure(figsize=(7, 6))
    markers = ['o', 's', 'D', '^', 'v']  # Different marker shapes
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Different colors for each line
    for i, attack in enumerate(['Gradients', 'Inter-Reps', 'Prediction Output', 'Label', 'test_accuracy']):
        mean_col = f'{attack}_mean'
        sem_col = f'{attack}_sem'
        label = 'Test Accuracy' if attack == 'test_accuracy' else attack
        plt.errorbar(grouped_results['label_defense_budget'], grouped_results[mean_col]*100, yerr=grouped_results[sem_col]*100, 
                     label=label, capsize=5, marker=markers[i], color=colors[i])

    # Add horizontal line for random guessing baseline of test accuracy
    plt.axhline(y=14.28, color='#9467bd', linestyle='--', label='Random Guessing')

    plt.xlabel(r'Label Defense Budget ($B$)', fontsize=18)
    plt.ylabel('Accuracy (%)', fontsize=18)
    dataset = args.experiment_name.split()[-1]
    plt.title(dataset, fontsize=20)
    plt.legend(fontsize=16)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Set x-axis ticks to show all budget values
    budget_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    plt.xticks(budget_values)

    # Increase tick font size and add more contrast
    plt.tick_params(axis='both', which='major', labelsize=16, width=2, length=6)
    plt.tick_params(axis='both', which='minor', labelsize=10, width=1, length=4)

    # Add border to the figure
    plt.gca().spines['top'].set_linewidth(1.5)
    plt.gca().spines['right'].set_linewidth(1.5)
    plt.gca().spines['bottom'].set_linewidth(1.5)
    plt.gca().spines['left'].set_linewidth(1.5)

    plt.tight_layout()
    # save the figure pdf
    plt.savefig(f'label_rand_defense_{dataset.lower()}.pdf', dpi=150)
    plt.close()

if __name__ == "__main__":
    main()