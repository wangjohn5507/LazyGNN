import time

import argparse
import os.path as osp
import numpy as np
import time
import copy

import torch

print('version', torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
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
        z_sam = torch.zeros_like(x)
        g_sam = torch.zeros_like(x)
        
        # use memory
        equ_preds = equ_preds.to(device)
        equ_grad = equ_grad.to(device)
        z_sam = equ_preds[id]
        g_sam = equ_grad[id]
        
        z_sam = z_sam.to(self.device)
        g_sam = g_sam.to(self.device)
       
        self.save_for_backward(z_sam, g_sam)
        
         # forward pass

        z = torch.zeros_like(x)
        
        ### forward lazy propagation & momentum connection

        if torch.equal(z_sam[:self.size], torch.zeros_like(z_sam)[:self.size].to(self.device)):
            z = x
        else:
            z = (1-self.beta)*z_sam + self.beta*x #target nodes

        ### aggragation
        
        for i in range(self.K):
            z = (1 - self.alpha) * (self.adj_matrix @ z) + self.alpha * x
        return z

    @staticmethod
    def backward(self, grad_output):
        z_sam, g_sam = self.saved_tensors
        
        g = torch.zeros_like(g_sam)

        ###backward lazy propagation & momentum connection
        
        if torch.equal(g_sam[:self.size], torch.zeros_like(g_sam)[:self.size].to(self.device)):
            g = grad_output
        else:
            g = (1-self.theta)*g_sam + self.theta*grad_output
   
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
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(num_features, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, num_classes))
        self.prop1 = Lazy_Prop()
        self.g_mem = torch.zeros(num_nodes, num_classes)
        self.z_mem_tr = torch.zeros(num_nodes, num_classes)
        self.num_classes = num_classes
        self.num_nodes = num_nodes
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()


    def forward(self, x, adj, id, size, K, alpha, beta, theta, device):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        def backward_hook(grad):   ### lazy backprop
            self.g_mem = self.g_mem.to(device)
            self.g_mem[id[:size]] = grad[:size].clone().detach()
            return grad
        x.register_hook(backward_hook)
        z_out = self.prop1.apply(x, adj, id, size, K, alpha, beta, theta, self.z_mem_tr, self.g_mem, device) ### lazy forward
        self.z_mem_tr = self.z_mem_tr.to(device)
        self.z_mem_tr[id[:size]] = z_out[:size].clone().detach()  ###cache into memory
        out = F.log_softmax(z_out, dim=1)
        return out
    

    @torch.no_grad()
    def inference(self, x, device):
        self.eval()
        x = x.to(device)
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


def train(model, optimizer, train_loader, device, args):
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
    print('cuda: ', torch.cuda.is_available())
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print('device', device)
    writer = SummaryWriter('Tensorboard_record/')
    model = Net(in_channels, args.hidden, out_channels, args.num_layers, data.num_nodes, args.dropout)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    best = 0
    best_val = 0
    model.reset_parameters()
    for epoch in range(args.epochs):
        loss = train(model, optimizer, train_loader, device, args)
        writer.add_scalar('Loss/train', loss, epoch)
        train_acc, val_acc, test_acc = test(model, data, device, args)
        if val_acc > best_val:
            best_val = val_acc
            best = test_acc
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Accuracy/test', test_acc, epoch)
        print(f'Epoch: {epoch:02d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
            f'Test: {test_acc:.4f}, Best: {best:.4f}')
        
if __name__ == "__main__":
    args = get_args()
    main(args)