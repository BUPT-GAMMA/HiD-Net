import os
import torch
import random
import numpy as np
import scipy.sparse as sp
import pickle as pkl
import os
import sys
import networkx as nx
from sklearn.model_selection import ShuffleSplit
from torch_sparse import SparseTensor
from collections import Counter
from torch_geometric.datasets import Planetoid, WikipediaNetwork, Actor
from torch.utils.data import Dataset, DataLoader
from torch_geometric.utils.convert import to_networkx, from_networkx

def accuracy(output, label):
    """ Return accuracy of output compared to label.
    Parameters
    ----------
    output:
        output from model (torch.Tensor)
    label:
        node label (torch.Tensor)
    """
    preds = output.max(1)[1].type_as(label)
    correct = preds.eq(label).double()
    correct = correct.sum()
    return correct / len(label)


def sparse_mx_to_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a sparse tensor.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    rows = torch.from_numpy(sparse_mx.row).long()
    cols = torch.from_numpy(sparse_mx.col).long()
    values = torch.from_numpy(sparse_mx.data)
    return SparseTensor(row=rows, col=cols, value=values, sparse_sizes=torch.tensor(sparse_mx.shape))


def parse_index_f(path):
    """Parse the index file.
    Parameters
    ----------
    path:
        directory of index file (str)
    """
    index = []
    for line in open(path):
        index.append(int(line.strip()))
    return index


def get_mask(idx, l):
    """Create mask.
    """
    mask = torch.zeros(l, dtype=torch.bool)
    mask[idx] = 1
    return mask


def normalize(mx):
    """Row-normalize sparse matrix.
    """
    r_sum = np.array(mx.sum(1))
    r_inv = np.power(r_sum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def get_homophily(label, adj):
    num_node = len(label)
    label = label.repeat(num_node).reshape(num_node, -1)
    n = np.triu((label==label.T) & (adj==1)).sum(axis=0)
    d = np.triu(adj).sum(axis=0)
    homos = []
    for i in range(num_node):
        if d[i] > 0:
            homos.append(n[i] * 1./ d[i])
    return np.mean(homos)

def load_data(path, name):
    ROOT_DIR = os.getcwd()


    if name in ['cora', 'citeseer', 'pubmed']:
        if name == 'cora':
            dataset = Planetoid(root=f'{ROOT_DIR}/data', name='Cora')
        elif name == 'citeseer':
            dataset = Planetoid(root=f'{ROOT_DIR}/data', name='CiteSeer')
        elif name == 'pubmed':
            dataset = Planetoid(root=f'{ROOT_DIR}/data', name='PubMed')
        else:
            return
        return dataset

    elif name == 'sbm':
        with open("{}/{}.p".format(f'{ROOT_DIR}/data/hetero', name), 'rb') as f:
            (G, feature, label) = pkl.load(f)
        f.close()

        feature = normalize(feature)
        feature = torch.from_numpy(feature).float()

        adj = nx.adjacency_matrix(G).tolil()
        #v adj = sparse_mx_to_sparse_tensor(adj)

        num_class = len(set(label))
        num_node = len(label)
        idx_train = []
        idx_val = []
        idx_test = []
        for j in range(num_class):
            idx_train.extend([i for i, x in enumerate(label) if x == j][:5])
            idx_val.extend([i for i, x in enumerate(label) if x == j][5:10])
            idx_test.extend([i for i, x in enumerate(label) if x == j][10:20])

        label = torch.LongTensor(label)

        # homophily = get_homophily(label.cpu().numpy(), adj.to_dense().cpu().numpy())

        mask_train = get_mask(idx_train, label.size(0))
        mask_val = get_mask(idx_val, label.size(0))
        mask_test = get_mask(idx_test, label.size(0))
        pyg_graph = from_networkx(G)
        return DataSet(x=feature, y=label,  edge_index=pyg_graph.edge_index, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test,
                       mask_train=mask_train, mask_val=mask_val, mask_test=mask_test, )#homophily=homophily)

    elif name in ['chameleon', 'squirrel']:
        dataset = WikipediaNetwork(root=f'{ROOT_DIR}/data', name=name)
        return dataset
    elif name == 'actor':
        dataset = Actor(root=f'{ROOT_DIR}/data/actor')
        return dataset
    else:
        return

class DataSet(Dataset):
    def __init__(self, x, y, edge_index, idx_train, idx_val, idx_test,
                mask_train, mask_val, mask_test,):
        self.x = x
        self.y = y
        # self.adj = adj
        self.edge_index = edge_index
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test
        self.train_mask = mask_train
        self.val_mask = mask_val
        self.test_mask = mask_test
        self.num_nodes = x.size(0)
        self.num_features = x.size(1)
        self.num_classes = int(torch.max(y)) + 1
        # self.homophily = homophily

    def to(self, device):
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        self.edge_index = self.edge_index.to(device)
        self.train_mask = self.train_mask.to(device)
        self.val_mask = self.val_mask.to(device)
        self.test_mask = self.test_mask.to(device)
        return self
