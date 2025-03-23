import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import tensorflow as tf

from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, AvgPool2d
from torch.nn import RNN
from sublayers.singleNodeAttention import SingleNodeAttentionLayer
from sublayers.edge_pool_my import EdgePooling
from sublayers.bi_lstm import LSTMModel
from sublayers.ggnn import GGNN
from sublayers.global_self_att import GlobalSelfAttentionLayer
from torch_geometric.nn import (global_mean_pool, JumpingKnowledge)
from torch_geometric.nn.glob import GlobalAttention

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


        self.rnn = RNN(hidden, 128, 2, batch_first = True, bidirectional=True)
        self.pool = AvgPool2d((1, 2), stride=2)
        self.out = nn.Linear(128, 10)


    def reset_parameters(self):
        self.h.reset_parameters()

    def forward(self, data):
        features1 = data
        h1 = features1
        rnn_data = torch.stack([h1],dim=1)
        out, h0 = self.rnn(rnn_data)
        out = self.pool(out)
        # print(out,out.shape)
        res = self.out(out[:, -1, :])
        # res = torch.mean(out, dim=(0,), keepdim=True)
        out = F.softmax(res, dim=-1)

        # out = F.softmax(self.bi_lstm(lstm_data),dim=-1)
        # out = F.softmax(self.ggnn(ggnn_data, '', 2))
        return out

    
    def __repr__(self):
        return self.__class__.__name__

