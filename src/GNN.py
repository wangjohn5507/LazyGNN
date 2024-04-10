import argparse
import os.path as osp
import numpy as np
import time
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_sparse import SparseTensor
from torch_geometric.data import Data
from torch.nn import Linear
import scipy.sparse as sp

import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv

class Lazy_Prop(torch.autograd.Function):

    @staticmethod
    def forward(self, x: torch.Tensor, adj_matrix, id, size, K: int, alpha: float, beta: float, theta: float, 
                equ_preds: torch.FloatTensor, equ_grad: torch.FloatTensor, device, **kwargs):

        self.size = size
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.adj_matrix = adj_matrix
        self.device = device
        self.adj_matrix = self.adj_matrix.to(self.device)
        feature = torch.zeros_like(x)
        grad = torch.zeros_like(x)
        
        # use memory
        equ_preds = equ_preds.to(device)
        equ_grad = equ_grad.to(device)
        feature = equ_preds[id]
        grad = equ_grad[id]
        
        feature = feature.to(self.device)
        grad = grad.to(self.device)
       
        self.save_for_backward(feature, grad)
        
         # forward pass

        f = torch.zeros_like(x)
        
        ### forward lazy propagation & momentum connection

        if torch.equal(feature[:self.size], torch.zeros_like(feature)[:self.size].to(self.device)):
            f = x
        else:
            f = (1-self.beta)*feature + self.beta*x #target nodes

        ### aggragation
        
        for i in range(self.K):
            f = (1 - self.alpha) * (self.adj_matrix @ f) + self.alpha * x
        return f

    @staticmethod
    def backward(self, grad_output):
        feature, grad = self.saved_tensors
        
        g = torch.zeros_like(grad)

        ###backward lazy propagation & momentum connection
        
        if torch.equal(grad[:self.size], torch.zeros_like(grad)[:self.size].to(self.device)):
            g = grad_output
        else:
            g = (1-self.theta)*grad + self.theta*grad_output
   
        for j in range(self.K):
            g = (1 - self.alpha) * (self.adj_matrix @ g) + self.alpha * grad_output

        g[self.size:] = 0  ###use gradients of target nodes 
        return g, None, None, None, None, None, None, None, None, None, None


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
    
class Net(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, num_layers, num_nodes, dropout, **kwargs):
        super(Net, self).__init__()
        self.linears = torch.nn.ModuleList()
        self.linears.append(torch.nn.Linear(num_features, hidden_channels))
        for _ in range(num_layers - 2):
            self.linears.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.linears.append(torch.nn.Linear(hidden_channels, num_classes))
        self.prop1 = Lazy_Prop()
        self.grad_memory = torch.zeros(num_nodes, num_classes)
        self.feature_memory = torch.zeros(num_nodes, num_classes)
        self.num_classes = num_classes
        self.num_nodes = num_nodes
        self.dropout = dropout

    def reset_parameters(self):
        for linear in self.linears:
            linear.reset_parameters()


    def forward(self, x, adj, id, size, K, alpha, beta, theta, device):
        for i, linear in enumerate(self.linears[:-1]):
            x = linear(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linears[-1](x)
        def backward_hook(grad):   ### lazy backprop
            self.grad_memory = self.grad_memory.to(device)
            self.grad_memory[id[:size]] = grad[:size].clone().detach()
            return grad
        x.register_hook(backward_hook)
        z_out = self.prop1.apply(x, adj, id, size, K, alpha, beta, theta, self.feature_memory, self.grad_memory, device) ### lazy forward
        self.feature_memory = self.feature_memory.to(device)
        self.feature_memory[id[:size]] = z_out[:size].clone().detach()  ###cache into memory
        out = F.log_softmax(z_out, dim=1)
        return out
    

    @torch.no_grad()
    def inference(self, x, device):
        self.eval()
        x = x.to(device)
        for i, linear in enumerate(self.linears[:-1]):
            x = linear(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linears[-1](x)
        return x