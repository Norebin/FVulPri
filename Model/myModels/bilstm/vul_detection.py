import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch

from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from sublayers.singleNodeAttention import SingleNodeAttentionLayer
from sublayers.bi_lstm import LSTMModel
from sublayers.global_self_att import GlobalSelfAttentionLayer
from torch_geometric.nn import (global_mean_pool, JumpingKnowledge)

device = "cuda" if torch.cuda.is_available() else "cpu"
# device="cpu"


class VulnerabilityDetection(nn.Module):
    def __init__(self, num_layers, hidden, nheads, nclass, dropout, alpha, training):
        super(VulnerabilityDetection, self).__init__()
        self.dropout = dropout
        self.num_classes = nclass
        self.training = training
        self.nheads = nheads

        self.h = SingleNodeAttentionLayer(hidden, hidden, dropout=dropout, alpha=alpha, concat=True)

        self.global_self_att = [GlobalSelfAttentionLayer(hidden, 2*hidden, dropout=dropout, alpha=alpha) for _ in range(self.nheads)]
        for i, global_self_att in enumerate(self.global_self_att):
            self.add_module('attention_{}'.format(i), global_self_att)


        self.jump = JumpingKnowledge(mode='cat')
        self.lin1 = Linear((1+1+1)*hidden, hidden)


        self.bi_lstm = LSTMModel(hidden, 128, 2, self.num_classes)
        # self.ggnn = GGNN(opt)
        #self.lin2 = Linear(2*hidden, self.num_classes)




    def reset_parameters(self):
        self.h.reset_parameters()
        self.GAT.reset_parameters()
        self.out_att.reset_parameters()
        self.edge_pool1.reset_parameters()
        self.edge_pool2.reset_parameters()
        self.lin1.reset_parameters()
        self.bi_lstm.reset_parameters()
        # self.ggnn.reset_parameters()

    def forward(self, data):
        # features1, features2, edge_index1, edge_index2, edgesAttr1, edgesAttr2, adjacency1, adjacency2, node2node_features1, node2node_features2 = data
        features1,features2 = data
        h1 = features1
        h2 = features2
        #code patch 1
        # h1 = self.h(features1)

        # batch1 = torch.zeros(len(h1), dtype=torch.int64, device=device)
        # h1 = F.relu(h1)

        lstm_data = torch.stack([h1,h2],dim=1)
        res = self.bi_lstm(lstm_data)
        # res = torch.mean(out, dim=(0,), keepdim=True)
        out = F.softmax(res, dim=-1)

        return out

    
    def __repr__(self):
        return self.__class__.__name__

