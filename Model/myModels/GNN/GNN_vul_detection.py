import os

import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch


from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from sublayers.singleNodeAttention import SingleNodeAttentionLayer
from sublayers.edge_pool_my import EdgePooling
from sublayers.global_self_att import GlobalSelfAttentionLayer
from torch_geometric.nn import (global_mean_pool, JumpingKnowledge)
from torch_geometric.nn.glob import GlobalAttention
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GatedGraphConv

device = "cuda" if torch.cuda.is_available() else "cpu"
# device="cpu"


class VulnerabilityDetection(nn.Module):
    def __init__(self, num_layers, hidden, nheads, nclass, dropout, alpha, training, opt):
        super(VulnerabilityDetection, self).__init__()
        self.hidden = hidden
        self.dropout = dropout
        self.num_classes = nclass
        self.training = training
        self.nheads = nheads

        self.h = SingleNodeAttentionLayer(hidden, hidden, dropout=dropout, alpha=alpha, concat=True)

        self.global_self_att = [GlobalSelfAttentionLayer(hidden, 2*hidden, dropout=dropout, alpha=alpha) for _ in range(self.nheads)]
        for i, global_self_att in enumerate(self.global_self_att):
            self.add_module('attention_{}'.format(i), global_self_att)

        self.gnn = GCNConv(hidden, hidden)

        self.out = nn.Linear(hidden, 2)



    def reset_parameters(self):
        self.h.reset_parameters()



    def forward(self, data):
        # features1, features2, edge_index1, edge_index2, edgesAttr1, edgesAttr2, adjacency1, adjacency2, node2node_features1, node2node_features2 = data
        features1,  edge_index1, edgesAttr1, adjacency1, node2node_features1 = data
        h1 = self.h(features1)

        res = self.gnn(h1, edge_index1)
        res = self.out(res)
        res = torch.mean(res, dim=(0, ), keepdim=True)
        out = F.softmax(res, dim=-1)

        return out

    
    def __repr__(self):
        return self.__class__.__name__

