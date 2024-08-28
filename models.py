import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
import torch


class MLPModel(nn.Module):
    def __init__(self, args, input_size, hidden_size, output_size):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        #self.fc2.register_full_backward_hook(self.save_grad)
        self.gradient_list = []
        self.args = args
        if args.initialize:
            weight_init_factor = args.weight_init_factor
            self.fc1.weight.data.fill_(1) * weight_init_factor
            self.fc2.weight.data.fill_(1) * weight_init_factor
            self.fc1.bias.data.fill_(0)
            self.fc2.bias.data.fill_(0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if self.args.perform_attack or self.args.perform_attack_all_methods:
            if x.requires_grad:
                def save_grad(grad):
                    self.gradient_list.append(grad)
                x.register_hook(save_grad)  # save the gradient
        return x
    


class GCN(torch.nn.Module):
    def __init__(self, args, input_size, output_size):
        super().__init__()
        self.conv1 = GCNConv(input_size, 16)
        self.conv2 = GCNConv(16, output_size)
        #self.conv2_grads = []
        self.args = args
        if args.initialize:
            weight_init_factor = args.weight_init_factor
            self.conv1.lin.weight.data.fill_(1) * weight_init_factor
            self.conv2.lin.weight.data.fill_(1) * weight_init_factor
            # self.conv1.lin.bias.data.fill_(0)
            # self.conv2.lin.bias.data.fill_(0)

    def forward(self, x, edge_index):
       # x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        #x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        # if self.args.perform_attack:
        #     if x.requires_grad:
        #         def save_grad(grad):
        #             self.conv2_grads.append(grad)
        #         x.register_hook(save_grad)  # save the gradient

        return x#F.log_softmax(x, dim=1)
    
class GAT(torch.nn.Module):
    def __init__(self, args, input_size, output_size):
         super().__init__()
         self.conv1 = GATConv(in_channels=-1, out_channels=output_size)
         self.conv2 = GATConv(in_channels=output_size, out_channels=output_size)
         
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return x
    
class SAGE(torch.nn.Module):
    def __init__(self, args, input_size, output_size):
         super().__init__()
         self.conv1 = SAGEConv(in_channels=-1, out_channels=output_size)
         self.conv2 = SAGEConv(in_channels=output_size, out_channels=output_size)
         
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return x


class ActiveModel(nn.Module):
    def __init__(self, args, input_size, output_size):
        super(ActiveModel, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, output_size)
        self.args = args
        if args.initialize:
            weight_init_factor = args.weight_init_factor
            self.fc1.weight.data.fill_(1) * weight_init_factor
            self.fc2.weight.data.fill_(1) * weight_init_factor
            self.fc1.bias.data.fill_(0)
            self.fc2.bias.data.fill_(0)

    def add_noise_to_grad(self, grad):
        if self.args.gradient_defense:
            noise = torch.randn_like(grad) * self.args.gradient_defense_noise_level
            return grad + noise
        return grad

    def forward(self, x):
        if self.training and self.args.gradient_defense:
            x.register_hook(self.add_noise_to_grad)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x



