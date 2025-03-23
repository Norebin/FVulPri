import torch
# from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from sublayers.singleNodeAttention import SingleNodeAttentionLayer
# from layers.GAT_with_edge import GraphAttentionLayer
from sublayers.edge_pool_my import EdgePooling
from torch_geometric.nn import (global_mean_pool, JumpingKnowledge)
from torch_geometric.nn.glob import GlobalAttention
from torch_geometric.nn import GCNConv

device = "cuda" if torch.cuda.is_available() else "cpu"
# device='cpu'

class graphEmb(nn.Module):
    def __init__(self, num_layers, hidden, nheads, nclass, dropout, alpha, training):
        super(graphEmb, self).__init__()
        self.dropout = dropout
        self.num_classes = nclass
        self.training = training
        self.nheads = nheads

        self.h = SingleNodeAttentionLayer(hidden, hidden, dropout=dropout, alpha=alpha, concat=True)
        
        self.GCN1 = GCNConv(hidden, hidden)

        self.edge_pool1 = EdgePooling(hidden)

        self.GCN2 = GCNConv(hidden, hidden)

        self.edge_pool2 = EdgePooling(hidden)

        self.jump = JumpingKnowledge(mode='cat')

        self.mlp_gate1=nn.Sequential(nn.Linear(hidden,1),nn.Sigmoid())
        self.gpool1=GlobalAttention(gate_nn=self.mlp_gate1)

        self.mlp_gate2=nn.Sequential(nn.Linear(hidden,1),nn.Sigmoid())
        self.gpool2=GlobalAttention(gate_nn=self.mlp_gate2)

        self.mlp_gate3=nn.Sequential(nn.Linear(hidden,1),nn.Sigmoid())
        self.gpool3=GlobalAttention(gate_nn=self.mlp_gate3)

    def reset_parameters(self):
        self.h.reset_parameters()
        self.GAT.reset_parameters()
        self.out_att.reset_parameters()
        self.edge_pool1.reset_parameters()
        self.edge_pool2.reset_parameters()
        
    def forward(self, data):
        features1, edge_index1, edgesAttr1, adjacency1, node2node_features1= data

        #code patch 1
        h1 = self.h(features1)
        return h1
        #print()
        #print("input ",h1.shape)
        batch1 = torch.zeros(len(h1), dtype=torch.int64, device=device)

        hs1 = [self.gpool1(h1, batch1)]
        #h1 = F.dropout(h1, self.dropout, training=self.training)

        h1 = self.GCN1(h1, edge_index1)
        h1 = F.relu(h1)
        
        #print("layer1 ",h1.shape, edgesAttr1.shape)
        h1, edge_index1, edgesAttr1, batch1, _ = self.edge_pool1(h1, edge_index1, edgesAttr1, batch=batch1)
        batch1 = torch.zeros(len(h1), dtype=torch.int64, device=device)
        #print("h1",h1.shape)
        hs1 += [self.gpool2(h1, batch1)]
        #h1 = F.dropout(h1, self.dropout, training=self.training)

        h1 = self.GCN2(h1, edge_index1)
        h1 = F.relu(h1)

        #print("layer2 ",h1.shape, edgesAttr1.shape)
        h1, edge_index1, edgesAttr1, batch1, _ = self.edge_pool2(h1, edge_index1, edgesAttr1, batch=batch1)
        batch1 = torch.zeros(len(h1), dtype=torch.int64, device=device)

        hs1 += [self.gpool3(h1, batch1)]
        
        h1 = self.jump(hs1)

        return h1



    
        
    def __repr__(self):
        return self.__class__.__name__

