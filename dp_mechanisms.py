import torch
from torch.distributions import Laplace
import numpy as np

# def generate_lap_noise(adj, epsilon, sensitivity, device):
#     N = adj.shape[0]
#     laplace = Laplace(loc=torch.tensor(0.0, requires_grad=False, device=device),
#               scale=torch.tensor(sensitivity / epsilon, requires_grad=False, device=device))
#     noise = laplace.sample((N, N))
#     return torch.tril(noise, diagonal=-1)

def get_perturbed_adj_binarized(adj, epsilon, sensitivity, device, eps_factor):
    eps1 = epsilon * eps_factor
    eps2 = epsilon - eps1
    adj = adj.to_dense()

    n_edges = int(adj.sum() // 2)
    n_edges_keep = max(0, int(n_edges + np.random.laplace(0, 1 / eps1)))
    # print('n_edges_keep:', n_edges_keep)
    # print('n_edges:', n_edges)

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
    dp_adj = get_perturbed_adj_binarized(adj, epsilon,sensitvity, device, eps_factor=eps_factor)
    return dp_adj.cpu()


def LapGraphNorm(adj, epsilon, device, sensitivity):
    """
    Function to add Laplace noise to the normalized adjacency matrix.
    
    Args:
    adj (torch.Tensor): The input adjacency matrix, expected to be FirstOrderGCN normalized.
    epsilon (float): The privacy budget to be used for generating Laplace noise.
    device (str): The device (e.g. 'cuda' or 'cpu') on which the tensors will be processed.
    sensitivity (float): The sensitivity parameter for generating Laplace noise.

    Returns:
    torch.Tensor: The perturbed adjacency matrix with Laplace noise added.
    """

    # Keep only the lower triangle of the adjacency matrix
    adj = torch.tril(adj, diagonal=-1).to(device)

    # Generate Laplace noise and add it to the adjacency matrix
    noise = generate_lap_noise(adj, epsilon, sensitivity, device)
    adj = adj + noise

    # Return the sum of the perturbed adjacency matrix and its transpose
    return adj + adj.T


def sensitivity_LapGraphNorm(d_min, d_max):
    term1 = 1 / d_min
    term2 = 2 * d_max * (1 / (np.sqrt((d_min - 1) * d_min)) - 1 / d_min)

    S_rem = term1 + term2
    
    term1 = 1 / d_min
    term2 = 2 * d_max * ((1 / d_min) - 1 / (np.sqrt(d_min) * np.sqrt(d_min + 1)))

    S_add = term1 + term2
    return max(S_add, S_rem)
    
    



import math
import numpy as np
import random
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
import time
from tqdm import tqdm

def torch_sparse_to_scipy(tensor):
    tensor = tensor.coalesce()
    values = tensor._values().numpy()
    indices = tensor._indices().numpy()
    shape = tensor.shape

    return coo_matrix((values, indices), shape=shape)


def scipy_to_torch_sparse(sparse_matrix):
    """
    Convert a SciPy sparse matrix to a PyTorch sparse tensor.

    Args:
        sparse_matrix (scipy.sparse): A SciPy sparse matrix.

    Returns:
        torch.sparse: A PyTorch sparse tensor with the same data.
    """
    sparse_matrix = sparse_matrix.tocoo()
    indices = torch.LongTensor([sparse_matrix.row, sparse_matrix.col])
    values = torch.FloatTensor(sparse_matrix.data)
    shape = torch.Size(sparse_matrix.shape)

    return torch.sparse.FloatTensor(indices, values, shape)

def perturb_adj_continuous(adj, args):
    adj = torch_sparse_to_scipy(adj)
    n_nodes = adj.shape[0]
    n_edges = len(adj.data) // 2

    N = n_nodes
    t = time.time()

    A = sp.tril(adj, k=-1)
    print('getting the lower triangle of adj matrix done!')

    eps_1 = args.epsilon * args.epsilon_factor
    eps_2 = args.epsilon - eps_1
    noise = get_noise(noise_type=args.noise_type, size=(N, N), seed=args.seed, 
                    eps=eps_2, delta=1e-5, sensitivity=1)
    noise *= np.tri(*noise.shape, k=-1, dtype=np.bool)
    print(f'generating noise done using {time.time() - t} secs!')

    A += noise
    print(f'adding noise to the adj matrix done!')

    t = time.time()
    n_edges_keep = n_edges + int(
        get_noise(noise_type='laplace', size=1, seed=args.seed, 
                eps=eps_1, delta=1e-5, sensitivity=1)[0])
    print(f'edge number from {n_edges} to {n_edges_keep}')

    t = time.time()
    a_r = A.A.ravel()

    n_splits = 50
    len_h = len(a_r) // n_splits
    ind_list = []
    for i in tqdm(range(n_splits - 1)):
        ind = np.argpartition(a_r[len_h*i:len_h*(i+1)], -n_edges_keep)[-n_edges_keep:]
        ind_list.append(ind + len_h * i)

    ind = np.argpartition(a_r[len_h*(n_splits-1):], -n_edges_keep)[-n_edges_keep:]
    ind_list.append(ind + len_h * (n_splits - 1))

    ind_subset = np.hstack(ind_list)
    a_subset = a_r[ind_subset]
    ind = np.argpartition(a_subset, -n_edges_keep)[-n_edges_keep:]

    row_idx = []
    col_idx = []
    for idx in ind:
        idx = ind_subset[idx]
        row_idx.append(idx // N)
        col_idx.append(idx % N)
        assert(col_idx < row_idx)
    data_idx = np.ones(n_edges_keep, dtype=np.int32)
    print(f'data preparation done using {time.time() - t} secs!')

    mat = sp.csr_matrix((data_idx, (row_idx, col_idx)), shape=(N, N))
    dp_adj =  mat + mat.T
    dp_adj_ = scipy_to_torch_sparse(dp_adj)
    return dp_adj_


def get_noise(noise_type, size, seed, eps=10, delta=1e-5, sensitivity=2):
    np.random.seed(seed)

    if noise_type == 'laplace':
        noise = np.random.laplace(0, sensitivity/eps, size)
    elif noise_type == 'gaussian':
        c = np.sqrt(2*np.log(1.25/delta))
        stddev = c * sensitivity / eps
        noise = np.random.normal(0, stddev, size)
    elif noise_type == 'staircase':
        print('getting Staircase noise')
        if isinstance(size, int):
            size = (size, size)

        staircase_dist = Staircase(epsilon=torch.tensor(eps).cuda(),
                                sensitivity=torch.tensor(1.0).cuda())
        noise = staircase_dist.sample(size).cpu()
        noise = noise.numpy()
    else:
        raise NotImplementedError('noise {} not implemented!'.format(noise_type))

    return noise


import torch
from torch.distributions import Laplace
import numpy as np

# Other functions remain unchanged

def generate_lap_noise(adj, epsilon, sensitivity, device):
    N = adj.shape[0]
    laplace = Laplace(loc=torch.tensor(0.0, requires_grad=False, device=device),
              scale=torch.tensor(sensitivity / epsilon, requires_grad=False, device=device))
    noise = laplace.sample((N, N))
    return torch.tril(noise, diagonal=-1)

def distributed_LapGraph(adj, epsilon, device_ids, sensitivity=1.0):
    n_devices = len(device_ids)
    N = adj.shape[0]
    
    # Convert the sparse adjacency matrix to a dense matrix
    adj_dense = adj.to_dense()

    # Initialize the resulting perturbed adjacency matrix on CPU
    dp_adj = torch.zeros(N, N, dtype=adj.dtype)

    # Calculate the number of non-zero elements in the lower triangle
    adj_lower = torch.tril(adj_dense, diagonal=-1)
    n_edges_keep = int(adj_lower.sum().item())

    # Process adjacency matrix in parallel on different GPUs
    for k, device_id in enumerate(device_ids):
        device = torch.device(f'cuda:{device_id}')

        # Divide the adjacency matrix into submatrices along the diagonal
        start_row = k * N // n_devices
        end_row = (k + 1) * N // n_devices

        if start_row < end_row:
            adj_part = adj_dense[start_row:end_row, :start_row].to(device)

            # Generate Laplace noise and add it to the adjacency matrix
            noise = generate_lap_noise(adj_part, epsilon, sensitivity, device)
            adj_part_noisy = adj_part + noise

            # Get the top k values and their indices in each submatrix
            val, indx = torch.topk(torch.flatten(adj_part_noisy), n_edges_keep)
            binary_adj = torch.zeros_like(torch.flatten(adj_part_noisy), device=device)
            binary_adj[indx] = 1
            adj_part_dp = binary_adj.reshape(adj_part_noisy.shape)

            # Combine the results on CPU
            dp_adj[start_row:end_row, :start_row] = adj_part_dp.cpu()
            dp_adj[:start_row, start_row:end_row] = adj_part_dp.t().cpu()

    return dp_adj

########################################## Staircase mechanism #############################################

import numpy as np
import diffprivlib
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj
import torch
import math
from dp_mechanisms import LapGraph

import torch
from torch.distributions import Distribution, constraints
from torch.distributions.utils import _standard_normal, broadcast_all

class Staircase(Distribution):
    arg_constraints = {
        'epsilon': constraints.positive,
        'sensitivity': constraints.positive,
        'gamma': constraints.interval(0, 1)
    }
    support = constraints.real
    has_rsample = True

    def __init__(self, epsilon, sensitivity, gamma=None, validate_args=None):
        gamma = self._check_gamma(gamma, epsilon)
        self.epsilon, self.sensitivity, self.gamma = broadcast_all(epsilon, sensitivity, gamma)
        #self.epsilon, self.sensitivity, self.gamma = epsilon, sensitivity, gamma
        b = torch.exp(-self.epsilon)
        self.b = b
        super(Staircase, self).__init__(batch_shape=self.epsilon.size(), validate_args=validate_args)

    @classmethod
    def _check_gamma(cls, gamma, epsilon=None):
        if gamma is None and epsilon is not None:
            gamma = 1 / (1 + torch.exp(epsilon / 2))

        if not isinstance(gamma, (torch.Tensor, float, int)):
            raise TypeError("Gamma must be numeric")

        if not isinstance(gamma, torch.Tensor):
            gamma = torch.tensor(gamma).clone().detach()

        if not (0.0 <= gamma <= 1.0).all():
            raise ValueError("Gamma must be in [0,1]")

        return gamma

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        # Generate r.v. S
        S = torch.where(torch.rand(shape, device=self.epsilon.device) < 0.5, 1.0, -1.0)

        # Generate geometric r.v. G
        G = torch.distributions.Geometric(1 - self.b).sample(shape).to(self.epsilon.device)

        # Generate r.v. U
        U = torch.rand(shape, device=self.epsilon.device)

        # Generate binary r.v. B
        prob_B_zero = self.gamma / (self.gamma + (1 - self.gamma) * self.b)
        B = torch.where(torch.rand(shape, device=self.epsilon.device) < prob_B_zero, 0.0, 1.0)

        # Compute X
        X = S * ((1 - B) * ((G + self.gamma * U) * self.sensitivity) + B * 
                 ((G + self.gamma + (1 - self.gamma) * U) * self.sensitivity))
        return X

    def rsample(self, sample_shape=torch.Size()):
        # Reparameterization is generally complex for custom distributions
        # For now, just use regular sampling method
        return self.sample(sample_shape)

    def log_prob(self, value):
        # TODO: Implement the log probability if required
        raise NotImplementedError
    
    
def pick_largest_from_upper(adj, n_edges_keep):
    # Extract the upper triangle (excluding diagonal) of the adjacency matrix
    upper_triangle = torch.triu(adj, diagonal=1)
    
    # Flatten and pick the largest n_edges_keep values
    upper_flat = torch.flatten(upper_triangle)
    _, indices = torch.topk(upper_flat, n_edges_keep)
    
    # Create a new matrix with zeros of the same shape as adj
    result = torch.zeros_like(adj)
    
    # Set the chosen positions to 1
    result.view(-1)[indices] = 1
    
    # Make the result symmetric by adding its transpose
    result = result + result.T
    
    return result

def staircase_mechanism(adj, args):
    # eps = 1.0
    # eps_factor = 0.05
    adj = adj.to(args.device)
    eps1 = args.epsilon_factor * args.epsilon
    eps2 = args.epsilon - eps1

    staircase_dist = Staircase(epsilon=torch.tensor(eps2).cuda(),
                            sensitivity=torch.tensor(1.0).cuda())

    staircase_dist_T = Staircase(epsilon=torch.tensor(eps1).cuda(),
                                sensitivity=torch.tensor(1.0).cuda())
    # estimated number of edges with DP staircase mechanism
    T = math.floor(adj.sum() / 2 + staircase_dist_T.sample((1,)))
    perturbed_adj = adj + staircase_dist.sample((adj.shape))
    binarized_adj = pick_largest_from_upper(adj, T)
    return binarized_adj