import torch
import numpy as np
import pandas as pd
import scipy.io as spio
import scipy.sparse as sp
import random
import time


def clean_data(data):
    data = np.nan_to_num(data)
    data = data - np.mean(data)
    data = data / np.std(data)

    return data


def load_data():
    features = spio.loadmat('feature.mat')
    labels = spio.loadmat('label.mat')

    a, b, c, d = list(features.keys())
    e, f, g, h = list(labels.keys())
    feature = features[d]
    label = labels[h]

    feature = clean_data(feature)
    label = clean_data(label)

    return feature, label


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def process_data():

    features, labels = load_data()
    features = features.flatten()
    labels  = labels.flatten()
    features = sp.csr_matrix(features[: -1], dtype=np.float32)
    labels = encode_onehot(labels[: -1])

    idx_train = range(140)
    idx_val = range(150, 180)
    idx_test = range(190, 250)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return features, labels, idx_train, idx_val, idx_test


