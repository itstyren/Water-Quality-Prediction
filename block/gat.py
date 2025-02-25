# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_c, out_c, dropout):
        super(GraphAttentionLayer, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.dropout = dropout

        self.W = nn.Parameter(torch.zeros(size=(in_c, out_c)))
        nn.init.kaiming_uniform_(self.W)
        self.a = nn.Parameter(torch.zeros(size=(2*out_c, 1)))
        nn.init.kaiming_uniform_(self.a)

        self.leakyrelu = nn.LeakyReLU()
    
    def forward(self, inp, adj):
        B, N = inp.size(0), inp.size(1)
        h = torch.matmul(inp, self.W)

        a_input = torch.cat([h.repeat(1, 1, N).view(B, N * N, -1), h.repeat(1, N, 1)], dim=2).view(B, N, -1, 2 * self.out_c)

        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
        
        zero_vec = -1e12 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)   # [B,N, N]
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        return h_prime  
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GAT_model(nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        super(GAT_model, self).__init__()
        self.conv1 = GraphAttentionLayer(in_c, hid_c)
        self.conv2 = GraphAttentionLayer(hid_c, out_c)
        self.act = nn.ReLU()

    def forward(self, data, device):
        graph_data = data["graph"].to(device)[0]
        flow_x = data["flow_x"].to(device)
        B, N = flow_x.size(0), flow_x.size(1)
        flow_x = flow_x.view(B, N, -1)
        output_1 = self.act(self.conv1(flow_x, graph_data))
        output_2 = self.act(self.conv2(output_1, graph_data))

        return output_2.unsqueeze(2)
