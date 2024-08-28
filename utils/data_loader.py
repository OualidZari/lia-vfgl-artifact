from defense.label_defense import labels_defense
from torch_geometric.datasets import Planetoid, Twitch, TUDataset, Amazon, WebKB
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import numpy as np
import torch
import random
import os

def split_indices(n, train_ratio=0.5, val_ratio=0.1):
    shuffled_indices = np.random.permutation(n)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)

    train_indices = shuffled_indices[:train_size]
    val_indices = shuffled_indices[train_size:train_size+val_size]
    test_indices = shuffled_indices[train_size+val_size:]

    # Create boolean masks
    train_mask = np.zeros(n, dtype=bool)
    train_mask[train_indices] = True

    val_mask = np.zeros(n, dtype=bool)
    val_mask[val_indices] = True

    test_mask = np.zeros(n, dtype=bool)
    test_mask[test_indices] = True

    return train_mask, val_mask, test_mask

def load_data(args):
    # Load dataset
    if args.dataset in ['Texas', 'Wisconsin', 'Cornell']:
        dataset = WebKB(root='datasets', name=args.dataset)
        data = dataset[0]
    elif args.dataset in ['Cora', 'Citeseer', 'Pubmed']:
        dataset = Planetoid(root='datasets', name=args.dataset)
        data = dataset[0]
    elif args.dataset.startswith('Twitch'):
        name = args.dataset[-2:]
        dataset = Twitch(root='datasets', name=name)
        data = dataset[0]
    elif args.dataset == 'PROTEINS':
        dataset = TUDataset(root='datasets', name='PROTEINS')
        data = dataset[0]
    elif args.dataset == 'amazon_computer':
        dataset = Amazon(root='datasets', name='Computers')
        data = dataset[0]
    elif args.dataset == 'amazon_photo':
        dataset = Amazon(root='datasets', name='Photo')
        data = dataset[0]
    else:
        print('Dataset not found.')
        return None, None, None, None, None

    # Apply split_indices function
    train_mask, val_mask, test_mask = split_indices(n=data.num_nodes, train_ratio=args.train_ratio, val_ratio=args.val_ratio)

    return dataset, data, train_mask, val_mask, test_mask

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        

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


def process_attack_data_online(gradients, forward_values, output_values,
                               data_attack, attack_methods, args):
    attack_data = {}
    if 'gradients' in attack_methods:
        attack_data['gradients'] = gradients
    
    if 'forward_values' in attack_methods:
        forward_values = forward_values[data_attack.attacked_indices].cpu().numpy()
        attack_data['forward_values'] = forward_values

    if 'output_server' in attack_methods:
        output_values = output_values[data_attack.attacked_indices].cpu().numpy()
        attack_data['output_server'] = output_values
    
    if 'features' in attack_methods:
        features = data_attack.x.cpu().numpy()
        attack_data['features'] = features
        
    if 'labels' in attack_methods:
        labels = data_attack.y.cpu().numpy()
        attack_data['labels'] = labels
    return attack_data
    
    
def get_attacked_nodes_data(features_mlp, train_mask, args):

    dataset, data, _, _, _ = load_data(args)
    edge_index_np = data.edge_index.cpu().numpy()
    indices = np.arange(data.num_nodes)

    if args.label_defense:
        new_labels = labels_defense(data.y, args.label_defense_budget)
        data.y = new_labels
    if args.sampling_strategy == 'all_nodes':
        attacked_nodes = np.where(train_mask)[0]
    if args.sampling_strategy == 'random':
        attacked_nodes = np.random.choice(np.where(train_mask)[0], args.n_attacked_nodes, replace=False)
        
    print('Sampling is done using', args.sampling_strategy)
    attacked_nodes_mask = np.zeros(data.num_nodes, dtype=bool)
    node_index_mapping = {j: i for i, j in enumerate(np.where(attacked_nodes_mask)[0])}
    attacked_nodes_mask[attacked_nodes] = True
    adj = to_dense_adj(data.edge_index).squeeze()
    adj_attacked = adj[attacked_nodes_mask][:, attacked_nodes_mask].cpu().numpy()
    edge_index_attacked = dense_to_sparse(torch.Tensor(adj_attacked))[0]
    data.edge_index = edge_index_attacked
    data.attacked_indices = attacked_nodes
    data.attacked_nodes_mask = attacked_nodes_mask
    data.attacked_edge_index = edge_index_attacked
    #data.x = data.x[attacked_nodes_mask]
    data.x = features_mlp[attacked_nodes_mask]
    data.y = data.y[attacked_nodes_mask]
    data.adj = adj_attacked
    data.edge_index = edge_index_attacked
    adj = adj_attacked

    return data, node_index_mapping