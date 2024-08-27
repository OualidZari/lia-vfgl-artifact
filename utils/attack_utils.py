import torch
import numpy as np
from torch_geometric.utils import negative_sampling



def sample_pairs(args, edge_index, train_mask, num_pos_samples=None, num_neg_samples=None):
    # Convert numpy train_mask to torch tensor and move to GPU
    train_mask = torch.from_numpy(train_mask).bool().to(edge_index.device)

    # Filter edge_index to contain only edges that are part of the training set
    train_nodes = torch.where(train_mask)[0]
    train_edges = edge_index[:, train_mask[edge_index[0]] & train_mask[edge_index[1]]]

    # Positive samples
    if args.attack_all_nodes:
        pos_samples = train_edges.t().cpu().numpy().tolist()
        # If num_pos_samples is provided and is less than all training edges, raise an error
        if num_pos_samples and num_pos_samples < len(pos_samples):
            raise ValueError("num_pos_samples provided is less than total training edges when attack_all_nodes=True.")
    else:
        # Check if num_pos_samples is provided, otherwise default to the length of train_edges
        if not num_pos_samples:
            num_pos_samples = len(train_edges.t())
        
        # Check if the requested num_pos_samples is valid
        if num_pos_samples > train_edges.size(1):
            raise ValueError("Requested more positive samples than available training edges.")
        
        idx = torch.randint(train_edges.size(1), (num_pos_samples,), device=edge_index.device)
        pos_samples = train_edges[:, idx].t().cpu().numpy().tolist()

    # If num_neg_samples is not explicitly provided, set it to match num_pos_samples
    if not num_neg_samples:
        num_neg_samples = len(pos_samples)

    # Negative sampling
    neg_samples = negative_sampling(train_edges, num_nodes=train_nodes.size(0), num_neg_samples=num_neg_samples)

    # Convert the edge indices of the negative samples to the original node indices
    neg_samples = train_nodes[neg_samples]
    neg_samples = neg_samples.t().cpu().numpy().tolist()

    return pos_samples, neg_samples


def sample_pairs(args, edge_index, train_mask, num_neg_samples=None, num_pos_samples=None):
    # Convert numpy train_mask to torch tensor and move to GPU
    train_mask = torch.from_numpy(train_mask).to(edge_index.device)

    # Filter edge_index to contain only edges that are part of the training set
    train_nodes = torch.where(train_mask)[0]
    train_edges = edge_index[:, train_mask[edge_index[0]] & train_mask[edge_index[1]]]

    # Positive samples
    if args.attack_all_nodes:
        idx = torch.randint(train_edges.size(1), (num_pos_samples,), device=edge_index.device)
        pos_samples = train_edges[:, idx].t().cpu().numpy().tolist()
    else:
        pos_samples = train_edges.t().cpu().numpy().tolist()

    # If num_neg_samples is not provided, use as many negative samples as positive
    if args.attack_all_nodes:
        num_neg_samples = len(pos_samples)

    # Negative sampling
    neg_samples = negative_sampling(train_edges, num_nodes=train_nodes.size(0), num_neg_samples=num_neg_samples)

    # Convert the edge indices of the negative samples to the original node indices
    neg_samples = train_nodes[neg_samples]
    neg_samples = neg_samples.t().cpu().numpy().tolist()

    return pos_samples, neg_samples