import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import RNN, AvgPool2d

class rnn_detect(nn.Module):
    def __init__(self, num_layers, hidden, nheads, nclass, dropout, alpha, training):
        super(rnn_detect, self).__init__()
        self.dropout = dropout
        self.num_classes = nclass
        self.training = training
        self.nheads = nheads

        self.rnn = RNN(hidden, 128, 2, batch_first=True, bidirectional=True)
        self.pool = AvgPool2d((1, 2), stride=2)
        self.out = nn.Linear(128, 10)

    def reset_parameters(self):
        self.rnn.reset_parameters()

    ''' 
    def forward(self, h1, h2):
        lstm_data = torch.stack([h1,h2],dim=1)
        out = F.softmax(self.bi_lstm(lstm_data),dim=-1)
        return out
    '''

    def forward(self, h1):
        rnn_data = torch.stack([h1], dim=1)
        out, h0 = self.rnn(rnn_data)
        out = self.pool(out)
        res = self.out(out[:, -1, :])
        # res = torch.mean(out, dim=(0,), keepdim=True)
        out = F.softmax(res, dim=-1)
        return out
    def __repr__(self):
        return self.__class__.__name__