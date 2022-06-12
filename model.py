import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import math
import scipy.sparse as sp
from scipy.sparse import coo_matrix
import numpy as np
import itertools


# Construct the adjacency matrix
def gen_adj(adj):

    D = torch.pow(adj.sum(1).float(), -0.5)
    D = torch.diag(D)
    adjacency = torch.matmul(torch.matmul(adj, D).t(), D)

    return adjacency


# Define graph convolution operation
class graphconvolution(nn.Module):

    def __init__(self, f_in, f_out, bias=True):
        super(graphconvolution, self).__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.weight = nn.Parameter(torch.FloatTensor(f_in, f_out))

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(f_out))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def forward(self, x, adj):
        s = torch.mm(x, self.weight)
        dim_s = s.size()
        e = dim_s[0]
        f = dim_s[1]
        adj = adj.resize_(f, e)
        output = torch.spmm(adj, s)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)


# Construt a gcn-based gnn
class cognn(nn.Module):

    def __init__(self, dim_in, dim_hi, dim_out):
        super(cognn, self).__init__()
        self.l1 = graphconvolution(dim_in, dim_hi)
        self.bn1 = nn.BatchNorm1d(dim_hi)
        self.relu = nn.LeakyReLU(0.2)
        self.l2 = graphconvolution(dim_hi, dim_out)
        self.bn2 = nn.BatchNorm1d(dim_out)

    def forward(self, x, adj):
        x = self.l1(x, adj)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.l2(x, adj)
        x = self.bn2(x)

        return F.log_softmax(x, dim=-1)


# Construct a Resnet-connection gnn
class Resgnn(nn.Module):

    def __init__(self, dim_in, dim_hi, dim_out, dropout, depth):

        super(Resgnn, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(graphconvolution(dim_in, dim_hi))

        for _ in range(depth - 2):
            self.convs.append(graphconvolution(dim_hi, dim_hi))

        self.convs.append(graphconvolution(dim_hi*2, dim_out))

        self.parallel = graphconvolution(dim_hi, dim_hi)
        self.linear1 = nn.Linear(dim_out, 111)
        self.dropout = dropout

    def reset_parameters(self):

        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj):

        x = self.convs[0](x, adj)
        shortcut = x

        for conv in self.convs[1:-1]:
            x = conv(x, adj)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        shortcut = self.parallel(shortcut, adj)
        x = torch.cat([shortcut, x], 1)
        x = self.convs[-1](x, adj)
        x = self.linear1(x)

        return F.log_softmax(x, dim=-1)
