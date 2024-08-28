import torch
from torch.distributions import Laplace
import numpy as np

def get_perturbed_adj_binarized(adj, epsilon, sensitivity, device, eps_factor):
    """
    Generate a perturbed adjacency matrix using the Laplace mechanism and binarization.

    Args:
        adj (torch.Tensor): Input adjacency matrix.
        epsilon (float): Privacy budget.
        sensitivity (float): Sensitivity of the adjacency matrix.
        device (torch.device): Device to perform computations on.
        eps_factor (float): Factor to split epsilon between edge count and noise addition.

    Returns:
        torch.Tensor: Perturbed and binarized adjacency matrix.
    """
    # Split epsilon into two parts
    eps1 = epsilon * eps_factor
    eps2 = epsilon - eps1

    # Convert sparse adjacency matrix to dense
    adj = adj.to_dense()

    # Add Laplace noise to edge count
    n_edges = int(adj.sum() // 2)
    n_edges_keep = max(0, int(n_edges + np.random.laplace(0, 1 / eps1)))

    # Get lower triangular part of adjacency matrix
    adj = torch.tril(adj, diagonal=-1).to(device)

    # Generate and add Laplace noise to adjacency matrix
    noise = generate_lap_noise(adj, eps2, sensitivity, device)
    adj = adj + noise
 
    # Select top k edges based on noisy weights
    val, indx = torch.topk(torch.flatten(adj), n_edges_keep)
    binary_adj = torch.zeros_like(torch.flatten(adj), device=device)
    binary_adj[indx] = 1
    adj_dp = binary_adj.reshape(adj.shape)

    # Return symmetric adjacency matrix
    return adj_dp + adj_dp.T

def LapGraph(adj, epsilon, device, eps_factor=0.01):
    """
    Apply Laplace mechanism to generate a differentially private graph.

    Args:
        adj (torch.Tensor): Input adjacency matrix.
        epsilon (float): Privacy budget.
        device (str): Device to perform computations on.
        eps_factor (float, optional): Factor to split epsilon. Defaults to 0.01.

    Returns:
        torch.Tensor: Differentially private adjacency matrix.
    """
    device = torch.device(device)
    adj = adj.to(device)
    sensitivity = 1.0
    dp_adj = get_perturbed_adj_binarized(adj, epsilon, sensitivity, device, eps_factor=eps_factor)
    return dp_adj.cpu()

def generate_lap_noise(adj, epsilon, sensitivity, device):
    """
    Generate Laplace noise for the adjacency matrix.

    Args:
        adj (torch.Tensor): Input adjacency matrix.
        epsilon (float): Privacy budget.
        sensitivity (float): Sensitivity of the adjacency matrix.
        device (torch.device): Device to perform computations on.

    Returns:
        torch.Tensor: Lower triangular Laplace noise matrix.
    """
    N = adj.shape[0]
    laplace = Laplace(loc=torch.tensor(0.0, requires_grad=False, device=device),
              scale=torch.tensor(sensitivity / epsilon, requires_grad=False, device=device))
    noise = laplace.sample((N, N))
    return torch.tril(noise, diagonal=-1)
