import time

import argparse
import os.path as osp
import numpy as np
import time
import copy
import math

import torch

print('version', torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.data import Data
from torch.nn import Linear
from torch.utils.data import DataLoader

import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected
from torch_geometric.loader import DataLoader
from torch_geometric.loader import NeighborLoader
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric_autoscale import get_data
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from lazy_model import Net
from preprocessing import metis, permute, SubgraphLoader
from args import get_args
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn import GCNConv
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch.utils.tensorboard import SummaryWriter



device = 'cuda' if torch.cuda.is_available() else 'cpu'


dataset = PygNodePropPredDataset(name='ogbn-arxiv')
data = dataset[0].to(device)


split_idx = dataset.get_idx_split()
train_idx = split_idx['train'].to(device)
val_idx = split_idx['valid'].to(device)
test_idx = split_idx['test'].to(device)

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


def train(model, optimizer, scheduler, train_loader, device, args):
    model.train()
    for batch in train_loader:
        x = batch.x.to(device)
        adj_t = batch.adj_t.to(device)
        id = batch.n_id.to(device)
        optimizer.zero_grad()
        train_target = batch.train_mask[:batch.batch_size].to(device)  ###get target nodes
        batch.y = batch.y.to(device)
        y = batch.y[:batch.batch_size][train_target].to(device)
        out = model(x, adj_t, id, batch.batch_size, args.K_train, args.alpha, args.beta_train, args.theta, device)[:batch.batch_size][train_target]
        loss = F.nll_loss(out, y)
        loss.backward()
        optimizer.step()
        # scheduler.step()
    return float(loss)


@torch.no_grad()
def test(model, data, device, args):
    model.eval()
    mlp = model.inference(data.x, device).cpu()#.to(device)
    
    z_pred = mlp
    for i in range(args.K_val_test):
        #APPNP for inference
        data.adj_t = data.adj_t#.to(device)
        z_pred = (1 - args.alpha) * (data.adj_t @ z_pred) + args.alpha * mlp


    out = F.softmax(z_pred, dim=1)
    y_hat = out.argmax(dim=-1)
    y = data.y.to(y_hat.device)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((y_hat[mask] == y[mask]).sum()) / int(mask.sum()))
    return accs[0], accs[1], accs[2]



def main(args):
    # model = GCN(dataset.num_features, 256, dataset.num_classes).to(device)
    root='/tmp/datasets'
    data, in_channels, out_channels = get_data(root, args.dataset)
    print(in_channels, out_channels)
    print('data: ',data)
    perm, ptr = metis(data.adj_t, num_parts=args.num_parts, log=True)  #### clustering
    data = permute(data, perm, log=True)
    data.n_id = torch.arange(data.num_nodes) ### assign index to nodes
    
    data.adj_t = data.adj_t.set_diag()
    data.adj_t = gcn_norm(data.adj_t, add_self_loops=False)
    
    train_loader = SubgraphLoader(data, ptr, batch_size=args.batch_size,
                                  shuffle=True, num_workers=0,
                                  persistent_workers=False)
    print(len(train_loader))

    print('cuda: ', torch.cuda.is_available())
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print('device', device)
    writer = SummaryWriter('Tensorboard_record/')
    model = Net(in_channels, args.hidden, out_channels, args.num_layers, data.num_nodes, args.dropout)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    total_steps = args.epochs 
    warmup_steps = total_steps/10
    min_lr = 1e-5      
    max_lr = 1e-3        
    
    def cosine_warmup_scheduler(optimizer, warmup_steps, total_steps, min_lr, max_lr):
        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                warmup_factor = current_step / float(warmup_steps)
                return warmup_factor
            else:
                progress = float(current_step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
                progress_tensor = torch.tensor(progress)
                cosine_decay = 0.5 * (1.0 + torch.cos(torch.pi * progress_tensor))
                decayed = (1 - min_lr / max_lr) * cosine_decay + min_lr / max_lr
                return decayed

        return LambdaLR(optimizer, lr_lambda)

    scheduler = cosine_warmup_scheduler(optimizer, warmup_steps, total_steps, min_lr, max_lr)
    # scheduler = LambdaLR(optimizer, lr_lambda)
    
    best = 0
    best_val = 0
    model.reset_parameters()
    for epoch in range(args.epochs):
        scheduler.step()
        loss = train(model, optimizer, scheduler, train_loader, device, args)
        lr = scheduler.get_last_lr()
        writer.add_scalar('Loss/train', loss, epoch)
        writer.add_scalar('LR/leaning_rate', lr[0], epoch)
        train_acc, val_acc, test_acc = test(model, data, device, args)
        if val_acc > best_val:
            best_val = val_acc
            best = test_acc
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Accuracy/test', test_acc, epoch)
        print(f'Epoch: {epoch:02d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
            f'Test: {test_acc:.4f}, Best: {best:.4f}, Loss: {loss:.4f}, LR: {lr[0]:.4f}')
        
if __name__ == "__main__":
    args = get_args()
    main(args)