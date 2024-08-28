import torch
from torch.distributions import Laplace
import numpy as np

def get_perturbed_adj_binarized(adj, epsilon, sensitivity, device, eps_factor):
    eps1 = epsilon * eps_factor
    eps2 = epsilon - eps1
    adj = adj.to_dense()

    n_edges = int(adj.sum() // 2)
    n_edges_keep = max(0, int(n_edges + np.random.laplace(0, 1 / eps1)))

    adj = torch.tril(adj, diagonal=-1).to(device)
    noise = generate_lap_noise(adj, eps2, sensitivity, device)
    adj = adj + noise
 
    val, indx = torch.topk(torch.flatten(adj), n_edges_keep)
    binary_adj = torch.zeros_like(torch.flatten(adj), device=device)
    binary_adj[indx] = 1
    adj_dp = binary_adj.reshape(adj.shape)
    return adj_dp + adj_dp.T

def LapGraph(adj, epsilon, device, eps_factor=0.01):
    device = torch.device(device)
    adj = adj.to(device)
    sensitvity = 1.0
    dp_adj = get_perturbed_adj_binarized(adj, epsilon, sensitvity, device, eps_factor=eps_factor)
    return dp_adj.cpu()

def generate_lap_noise(adj, epsilon, sensitivity, device):
    N = adj.shape[0]
    laplace = Laplace(loc=torch.tensor(0.0, requires_grad=False, device=device),
              scale=torch.tensor(sensitivity / epsilon, requires_grad=False, device=device))
    noise = laplace.sample((N, N))
    return torch.tril(noise, diagonal=-1)
