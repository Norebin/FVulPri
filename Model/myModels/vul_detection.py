import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch

from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from sublayers.singleNodeAttention import SingleNodeAttentionLayer
from sublayers.edge_pool_my import EdgePooling
from sublayers.bi_lstm import LSTMModel
from sublayers.ggnn import GGNN
from sublayers.global_self_att import GlobalSelfAttentionLayer
from torch_geometric.nn import (global_mean_pool, JumpingKnowledge)
from torch_geometric.nn.glob import GlobalAttention
from torch_geometric.nn import GCNConv

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

        self.GCN1 = GCNConv(hidden, hidden)

        self.edge_pool1 = EdgePooling(hidden)

        self.GCN2 = GCNConv(hidden, hidden)

        self.edge_pool2 = EdgePooling(hidden)

        self.jump = JumpingKnowledge(mode='cat')
        self.lin1 = Linear((1+1+1)*hidden, hidden)


        self.bi_lstm = LSTMModel(hidden*(1+1+1), 128, 2, self.num_classes)
        # self.ggnn = GGNN(opt)
        #self.lin2 = Linear(2*hidden, self.num_classes)

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
        self.lin1.reset_parameters()
        self.bi_lstm.reset_parameters()
        # self.ggnn.reset_parameters()

    def forward(self, data):
        # features1, features2, edge_index1, edge_index2, edgesAttr1, edgesAttr2, adjacency1, adjacency2, node2node_features1, node2node_features2 = data
        features1,  edge_index1, edgesAttr1, adjacency1, node2node_features1 = data

        #code patch 1
        h1 = self.h(features1)
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

        #print("global_mean_pool(h1, batch1)",global_mean_pool(h1, batch1).shape)
        '''
        global_vec_tensor_list1 = []
        for global_att in self.global_self_att:
            global_vec_tensor1 = global_att(all_token_tensor1)
            global_vec_tensor_list1.append(global_vec_tensor1)
        global_self_att_cat1 = torch.cat(global_vec_tensor_list1, dim=1)

        hs1 += [global_self_att_cat1]
        '''
        h1 = self.jump(hs1)


        #code patch 2
        # h2 = self.h(features2)
        #print("input ",h2.shape)
        # batch2 = torch.zeros(len(h2), dtype=torch.int64, device='cuda')
        # hs2 = [self.gpool1(h2, batch2)]
        #h2 = F.dropout(h2, self.dropout, training=self.training)

        # h2 = self.GCN1(h2, edge_index2)
        # h2 = F.relu(h2)

        # print("layer1 ",h2.shape, edgesAttr2.shape)
        # h2, edge_index2, edgesAttr2, batch2, _ = self.edge_pool1(h2, edge_index2, edgesAttr2, batch=batch2)
        # batch2 = torch.zeros(len(h2), dtype=torch.int64, device='cuda')
        # hs2 += [self.gpool2(h2, batch2)]
        #h2 = F.dropout(h2, self.dropout, training=self.training)

        # h2 = self.GCN2(h2, edge_index2)
        # h2 = F.relu(h2)
        #print("layer2 ",h2.shape, edgesAttr2.shape)
        # h2, edge_index2, edgesAttr2, batch2, _ = self.edge_pool2(h2, edge_index2, edgesAttr2, batch=batch2)
        # batch2 = torch.zeros(len(h2), dtype=torch.int64, device='cuda')
        # hs2 += [self.gpool3(h2, batch2)]


        '''
        global_vec_tensor_list2 = []
        for global_att in self.global_self_att:
            global_vec_tensor2 = global_att(all_token_tensor2)
            global_vec_tensor_list2.append(global_vec_tensor2)
        global_self_att_cat2 = torch.cat(global_vec_tensor_list2, dim=1)
        hs2 += [global_self_att_cat2]
        '''
        # h2 = self.jump(hs2)

        # lstm_data = torch.stack([h1,h2],dim=1)
        # h1 = features1
        lstm_data = torch.stack([h1],dim=1)
        # ggnn_data = torch.stack([h1],dim=1)
        #print("lstm_data",lstm_data.shape)

        out = F.softmax(self.bi_lstm(lstm_data),dim=-1)
        # out = F.softmax(self.ggnn(ggnn_data, '', 2))
        return out

    
    def __repr__(self):
        return self.__class__.__name__

