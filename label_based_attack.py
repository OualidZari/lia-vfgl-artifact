import pickle

import torch

from torch_geometric.datasets import Planetoid, Twitch
import matplotlib.pyplot as plt
import os
import argparse
import pandas as pd
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from sklearn.metrics import roc_curve, confusion_matrix, f1_score
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import roc_curve, auc
from utils import get_args, load_data
from argparse import Namespace
from torch.nn.functional import softmax
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, accuracy_score
import numpy as np

def load_attack_data(run_id):

    with open(f'scores_log/{run_id}_gradients.pkl', 'rb') as f:
        loaded_gradients = pickle.load(f)
        loaded_gradients = [tensor.cpu() for tensor in loaded_gradients]
    with open(f'scores_log/{run_id}_train_mask.pkl', 'rb') as f:
        loaded_train_mask = pickle.load(f)
    with open(f'scores_log/{run_id}_forward_mlp.pkl', 'rb') as f:
        loaded_forward_mlp = pickle.load(f)
    if os.path.exists(f'scores_log/{run_id}_output_server.pkl'):
        with open(f'scores_log/{run_id}_output_server.pkl', 'rb') as f:
            loaded_output_server = pickle.load(f)
    else:
        loaded_output_server = None
        
    return loaded_gradients, loaded_train_mask, loaded_forward_mlp, loaded_output_server


def get_training_graph(train_mask, args):
    
    dataset, data, _, _, _ = load_data(args)
    edge_index_np = data.edge_index.cpu().numpy()
    train_mask_np = np.array(train_mask)
    node_index_mapping = {j: i for i, j in enumerate(np.where(train_mask_np)[0])}
    train_labels = data.y[train_mask_np]
    train_features = data.x[train_mask_np]
    data.x = train_features
    data.y = train_labels
    adj = to_dense_adj(data.edge_index).squeeze()
    adj_train = adj[train_mask_np][:, train_mask_np].cpu().numpy()
    edge_index_train = dense_to_sparse(torch.Tensor(adj_train))[0]
    data.edge_index = edge_index_train
    data.train_mask = torch.Tensor(train_mask)
    data.adj = to_dense_adj(data.edge_index).squeeze()


    return data, node_index_mapping


def get_attack_data(train_mask, args):
    dataset, data, _, _, _ = load_data(args)
    edge_index_np = data.edge_index.cpu().numpy()
    train_mask_np = np.array(train_mask)
    pass

def process_attack_data(attack_methods, attack_time, args):
    attack_data = {}
    
    # Load necessary data
    loaded_gradients, loaded_train_mask, loaded_forward_mlp, loaded_output_server = load_attack_data(args.run_id)
    data_train, node_index_mapping = get_training_graph(loaded_train_mask, args)
    # time_index = args.attack_epochs.index(attack_time)
    time_index = list(range(1, 301)).index(attack_time)
    # Process data based on attack methods
    if 'gradients' in attack_methods:
        gradients = loaded_gradients[time_index][loaded_train_mask].cpu().numpy()
        attack_data['gradients'] = gradients
    
    if 'forward_values' in attack_methods:
        print('the attack time index is', time_index)
        forward_values = loaded_forward_mlp[time_index][loaded_train_mask]
        attack_data['forward_values'] = forward_values

    if 'output_server' in attack_methods:
        output_values = loaded_output_server[time_index][loaded_train_mask]
        attack_data['output_server'] = output_values
    
    if 'features' in attack_methods:
        features = data_train.x.cpu().numpy()
        attack_data['features'] = features
        
    if 'labels' in attack_methods:
        labels = data_train.y.cpu().numpy()
        attack_data['labels'] = labels
    return attack_data


def compute_performance_metrics(args, inputs, labels, attack_method, indices, **kwargs):
    
    if attack_method == 'labels':
        predictions = (inputs[:, None] == inputs).astype(int)
        predictions = predictions[np.triu_indices_from(predictions, k=1)]
        cos_sim_upper = predictions
    else:
        inputs = inputs

        cos_sim = cosine_similarity(inputs)
        cos_sim_upper = cos_sim[np.triu_indices_from(cos_sim, k=1)]
            
    labels_upper = labels[np.triu_indices_from(labels, k=1)]
    
    if indices:
        cos_sim_upper = cos_sim_upper[indices]
        labels_upper = labels_upper[indices]
        
        
        
    if attack_method == 'labels':
        accuracy = np.mean(cos_sim_upper == labels_upper)
        area_under_curve = None
        precision = precision_score(labels_upper, cos_sim_upper)
        recall = recall_score(labels_upper, cos_sim_upper)
        f1 = f1_score(labels_upper, cos_sim_upper)
        
    else:
        
        fpr, tpr, thresholds_temp = roc_curve(labels_upper, cos_sim_upper)
        f1_scores_temp = 2 * (tpr * (1 - fpr)) / (tpr + (1 - fpr))
        optimal_threshold = thresholds_temp[f1_scores_temp.argmax()]
        
        predictions = (cos_sim_upper > optimal_threshold).astype(int)
        accuracy = np.mean(predictions == labels_upper)

        area_under_curve = auc(fpr, tpr)

        
    if args.save_predictions:
        np.save(f'predictions/{kwargs["ID"]}_{attack_method}_{kwargs["attack_time"]}_predictions.npy', predictions)
        np.save(f'predictions/{kwargs["ID"]}_{attack_method}_{kwargs["attack_time"]}_ground_truth.npy', labels_upper)
        np.save(f'predictions/{kwargs["ID"]}_{attack_method}_{kwargs["attack_time"]}_cos_sim_upper.npy', cos_sim_upper)
        
    
    return {
        'Accuracy': accuracy,
        'AUC': area_under_curve,
    }
    
def get_attacked_nodes(labels, num_positives=None, num_negatives=None):
    
    labels_upper = labels[np.triu_indices_from(labels, k=1)].cpu().numpy()
    # Calibrate the number of positive and negative examples
    if num_positives or num_negatives:
        pos_indices = np.where(labels_upper == 1)[0]
        neg_indices = np.where(labels_upper == 0)[0]


        if num_positives:
            num_positives = min(num_positives, len(pos_indices))  # Take the minimum of the desired and available positives
            pos_indices = np.random.choice(pos_indices, num_positives, replace=False)

        if num_negatives:
            num_negatives = min(num_negatives, len(neg_indices))  # Take the minimum of the desired and available negatives
            neg_indices = np.random.choice(neg_indices, num_negatives, replace=False)

        indices = np.concatenate((pos_indices, neg_indices))
        print('Number of positives:', len(pos_indices))
        print('Number of negatives:', len(neg_indices))
        
    return indices


def main(args):
    
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    # Load attack data
    loaded_gradients, loaded_train_mask, loaded_forward_mlp, loaded_output_server = load_attack_data(args.run_id)
    
    # Get training graph
    data_train, node_index_mapping = get_training_graph(loaded_train_mask, args)
    
    # Process attack data
    attack_methods = args.attack_methods.split(',')
    if '-' in args.attack_epochs:
        attack_epochs = args.attack_epochs
        attack_epochs = attack_epochs.split('-')
        attack_epochs = list(range(int(attack_epochs[0]), int(attack_epochs[1])+1))
        args.attack_epochs = attack_epochs
    else:
        args.attack_epochs = [int(epoch) for epoch in args.attack_epochs.split(',')]
        
    attack_times = args.attack_epochs
    
    for attack_time in attack_times:
        args.attack_time = attack_time
        attack_data = process_attack_data(attack_methods, args.attack_time, args)
        
        # Store results for each method
        results = {}
        for attack_method in attack_methods:
            print(f"Evaluating attack method: {attack_method}")
            attack_results = compute_performance_metrics(
                inputs=attack_data[attack_method],
                attack_method=attack_method,
                labels=data_train.adj.cpu().numpy(),
                indices=None,
            )
            
            results[attack_method] = attack_results
            results[attack_method]['Attack Time'] = args.attack_time
            results[attack_method]['Seed'] = args.seed
            results[attack_method]['Attack Method'] = attack_method
            results[attack_method]['ID'] = args.run_id
        
        df = pd.DataFrame(results).T
        output_file = f'label_based_attack_logs/attack_results_{args.run_id}.csv'
        if os.path.isfile(output_file):
            df.to_csv(output_file, mode='a', header=False, index=False)
        else:
            df.to_csv(output_file, index=False)
        
    # print("\nComparative Performance of Attack Methods:")
    # print(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Attack performance evaluator')
    
    parser.add_argument('--dataset', default="Cora", type=str, help='Dataset to use')
    parser.add_argument('--train_ratio', default=0.5, type=float, help='Training set ratio')
    parser.add_argument('--val_ratio', default=0.1, type=float, help='Validation set ratio')
    parser.add_argument('--test_ratio', default=0.4, type=float, help='Test set ratio')
    parser.add_argument('--attack_epochs', default="1,100,200,300", type=str, help='Comma-separated list of attack epochs')
    parser.add_argument('--run_id', required=True, type=str, help='Run ID to load data from')
    parser.add_argument('--attack_methods', default='gradients,labels', type=str, help='Comma-separated list of attack methods to evaluate')
    parser.add_argument('--attack_time', default=1, type=int, help='Time to evaluate the attack')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--num_positives', type=int, help='Number of positive examples to use')
    parser.add_argument('--num_negatives', type=int, help='Number of negative examples to use')
    parser.add_argument('--save_predictions', action='store_true', help='Whether to save predictions')
    
    args = parser.parse_args()
    
    # Convert some comma-separated arguments to lists
    # if '-' in args.attack_epochs:
    #     attack_epochs = args.attack_epochs
    #     attack_epochs = attack_epochs.split('-')
    #     attack_epochs = list(range(int(attack_epochs[0]), int(attack_epochs[1])+1))
    #     args.attack_epochs = attack_epochs
    # else:
    #     args.attack_epochs = [int(epoch) for epoch in args.attack_epochs.split(',')]
    
    main(args)

# python label_based_attack.py --dataset Cora --run_id 399vcr2y --attack_methods gradients,labels,features,output_server,forward_values
#python label_based_attack.py --dataset Cora --run_id 28g7phwz --attack_methods forward_values --attack_time 300