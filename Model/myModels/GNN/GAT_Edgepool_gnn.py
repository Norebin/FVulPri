import torch
import torch.nn as nn
from torch_geometric.nn import GatedGraphConv
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class gnn_detect(nn.Module):
    def __init__(self, num_layers, hidden, nheads, nclass, dropout, alpha, training):
        super(gnn_detect, self).__init__()
        self.hidden = hidden
        self.dropout = dropout
        self.num_classes = nclass
        self.training = training
        self.nheads = nheads

        self.gnn = GCNConv(hidden, hidden)
        self.out = nn.Linear(hidden, 2)

    def reset_parameters(self):
        self.gnn.reset_parameters()

    ''' 
    def forward(self, h1, h2):
        lstm_data = torch.stack([h1,h2],dim=1)
        out = F.softmax(self.bi_lstm(lstm_data),dim=-1)
        return out
    '''

    def forward(self, h1, edge_index1):
        res = self.gnn(h1, edge_index1)
        res = self.out(res)
        res = torch.mean(res, dim=(0,), keepdim=True)
        out = F.softmax(res, dim=-1)
        return res,out
    def __repr__(self):
        return self.__class__.__name__

