import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

import numpy as np
import math
import time
import matplotlib.pyplot as plt
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import model as model
import datapre as datapre
import evaluation as evaluation


def train():

    loss_history = []
    valid_loss_history = []
    model.train()

    for epoch in range(epochs):
        train_logits = model(tensor_adjacency, x_train)
        loss = criterion(train_logits.double(), y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("Epoch {:03d}: Loss {:.4f}".format(
            epoch, loss.item()))

        if epoch % validationInterval == 0:
            valid_logits = model(tensor_adjacency, x_valid)
            valid_loss = criterion(valid_logits.double(), y_valid)
            print("Validation Loss {:.4f}".format(valid_loss))

    return loss_history, valid_loss


def test():

    model.eval()

    for epoch in range(epochs):
        output = model(tensor_adjacency, x_test)
        loss_test = criterion(output, y_test)

    print("Test set reults:",
          "loss={:.4f}".format(loss_test.item()))


start_time_tr = time.time()

test_size = 0.2
random_state = 42

dim_hi = 64
dim_in = 12
f_in = 4
f_out = 118
depth = 6

epochs = 500
lr = 0.001
dropout = 0.5

validationInterval = 4

weight_decay = 5e-4

x, y = datapre.load_data()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

node_feature = x / x.sum(1, keepdims=True)
num_nodes = node_feature.shape
classes = np.unique(y)
n_classes = 111 # 118

n_ts = x.shape[0]

x_train = torch.tensor(x_train)
y_train = torch.tensor(y_train)

n_va = int(np.floor(0.01*n_ts))
x_valid = x_train[0:n_va]
y_valid = y_train[0:n_va]
x_train = x_train[n_va:]
y_train = y_train[n_va:]

y_train = y_train.resize_(n_classes, n_classes)
x_train = x_train.float()

y_valid = y_valid.resize_(n_classes, n_classes)
x_valid = x_valid.float()


dim_out = n_classes

adj = np.array(
    [[0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
     [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
     [0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
     [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0],
     [0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1],
     [0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1],
     [1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
     [0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1],
     [0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0],
     [0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0],
     [0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1],
     [0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0]])

adj += sp.eye(adj.shape[0])

tensor_adj = torch.tensor(adj).float()
tensor_adjacency = model.gen_adj(tensor_adj)


model = model.Resgnn(dim_in, dim_hi, dim_out, dropout, depth)
# model = model.cognn(dim_in, dim_hi, dim_out)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(),
                       lr=lr,
                       weight_decay=weight_decay)


loss = train()

print("Time elapsed during training: {:.4f}s".format(time.time() - start_time_tr))


start_time_te = time.time()

x_test = torch.tensor(x_test)
y_test = torch.tensor(y_test)

x_test = x_test.float()
y_test = y_test.float()
y_test = y_test.resize_(n_classes, n_classes)

test()

print("Time elapsed during testing: {:.4f}s".format(time.time() - start_time_te))
