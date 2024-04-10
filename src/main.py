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
from GNN import Net, GCN, Lazy_Prop
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



if __name__ == "__main__":
    args = get_args()
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