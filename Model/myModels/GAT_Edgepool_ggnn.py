import torch
import torch.nn as nn
from torch_geometric.nn import GatedGraphConv
import torch.nn.functional as F
from sublayers.bi_lstm import LSTMModel
from sublayers.ggnn import GGNN

class ggnn_detect(nn.Module):
    def __init__(self, num_layers, hidden, nheads, nclass, dropout, alpha, training):
        super(ggnn_detect, self).__init__()
        self.hidden = hidden
        self.dropout = dropout
        self.num_classes = nclass
        self.training = training
        self.nheads = nheads

        self.ggnn = GatedGraphConv(out_channels=64, num_layers=2)
        self.out = nn.Linear(self.hidden * 4, 2)

    def reset_parameters(self):
        self.ggnn.reset_parameters()

    ''' 
    def forward(self, h1, h2):
        lstm_data = torch.stack([h1,h2],dim=1)
        out = F.softmax(self.bi_lstm(lstm_data),dim=-1)
        return out
    '''

    def forward(self, h1, edge_index1):

        res = self.ggnn(h1, edge_index1)
        res = self.out(res)
        a2 = torch.sum(res, dim=(0,), keepdim=True)
        out = F.softmax(a2, dim=-1)
        return out
    def __repr__(self):
        return self.__class__.__name__

