from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor, PairTensor

import torch
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
from torch_sparse import spspmm
from torch_geometric.nn.inits import zeros
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import gather_csr, scatter
from torch_geometric.nn.conv import GATConv
import torch.nn.functional as F
import numpy as np
import os
import yaml
#from memory_profiler import profile




def compute_D(a, b):
    t1 = a.unsqueeze(1).expand(len(a), len(a), a.shape[1])
    t2 = b.unsqueeze(0).expand(len(b), len(b), b.shape[1])
    d = (t1 - t2).pow(2).sum(2)
    return d

# def calculate_P(edge_index, x, edge_weight=None, num_nodes=None, improved=False,
#              add_self_loops=True, dtype=None):
#     if edge_weight is None:
#         edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
#                                  device=edge_index.device)
#     num_nodes = maybe_num_nodes(edge_index, num_nodes)
#     row, col = edge_index[0], edge_index[1]
#     deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
#     deg_inv_sqrt = deg.pow_(-0.5)
#     deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
#     absx = torch.norm(x, p=2, dim=1)
#     return ((torch.sum(x[row] * x[col], dim=1) / (absx[row] * absx[col])) * deg_inv_sqrt[col] * deg_inv_sqrt[row]).view(-1, 1) * (x[col] - x[row])
#
# def calculate_Q(edge_index, x, edge_weight=None, num_nodes=None, improved=False,
#              add_self_loops=True, dtype=None):
#     if edge_weight is None:
#         edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
#                                  device=edge_index.device)
#     num_nodes = maybe_num_nodes(edge_index, num_nodes)
#     row, col = edge_index[0], edge_index[1]
#     deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
#     deg_inv_sqrt = deg.pow_(-0.5)
#     deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
#     # absx = torch.norm(x, p=2, dim=1)
#     dx = x[col] - x[row]
#     absdx = torch.norm(dx, p=2, dim=1)
#     return ((torch.sum(dx * dx, dim=1) / (absdx + 1e-5)) * deg_inv_sqrt[col] * deg_inv_sqrt[row]).view(-1, 1) * (x[col] - x[row])
#

# 第一步用均值，第二部用s
def cal_g_gradient1(edge_index, x, edge_weight=None, sigma1=None, sigma2=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    row, col = edge_index[0], edge_index[1]
    ones = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    deg = scatter_add(ones, col, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    deg_inv = deg.pow(-1)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)
    # caculate gradient
    gra = deg_inv[row].view(-1, 1) * (x[col] - x[row])
    avg_gra = scatter(gra, row, dim=-2, dim_size=x.size(0), reduce='add')

    # calculate similarity
    dx = x[row] - x[col]
    s = torch.norm(dx, p=2, dim=1)
    # sigma2 = torch.var(s)
    s = torch.exp(- (s * s) / (2 * sigma2 * sigma2))
    r = scatter(s.view(-1, 1), row, dim=-2, dim_size=x.size(0), reduce='add')
    coe = s.view(-1, 1) / (r[row] + 1e-12)
    result = scatter(avg_gra[row] * coe, col, dim=-2, dim_size=x.size(0), reduce='add')
    # result = scatter(avg_gra[row] * (deg_inv_sqrt[col] * deg_inv_sqrt[row]).view(-1, 1), col, dim=-2, dim_size=x.size(0), reduce='sum')
    return result

# 第一步用ew，第二部用s+ew
def cal_g_gradient2(edge_index, x, edge_weight=None, sigma1=None, sigma2=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    deg_inv = deg.pow_(-1)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)
    # caculate gradient
    gra = deg_inv[row].view(-1, 1) * (x[col] - x[row])
    avg_gra = scatter(gra, row, dim=-2, dim_size=x.size(0), reduce='add')

    # calculate similarity
    dx = x[row] - x[col]
    s = torch.norm(dx, p=2, dim=1)
    s = (s * s) / (2 * sigma2 * sigma2)
    r = scatter(s.view(-1, 1), row, dim=-2, dim_size=x.size(0), reduce='add')
    coe = s.view(-1, 1) / (r[row] + 1e-6)
    result = scatter(avg_gra[row] * coe * edge_weight, col, dim=-2, dim_size=x.size(0), reduce='sum')
    # result = scatter(avg_gra[row] * (deg_inv_sqrt[col] * deg_inv_sqrt[row]).view(-1, 1), col, dim=-2, dim_size=x.size(0), reduce='sum')
    return result

#@profile(precision=4, stream=open('g2.log','w+'))
def cal_g_gradient2(edge_index, x, edge_weight=None, sigma1=None, sigma2=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    row, col = edge_index[0], edge_index[1]

    onestep = scatter(x[col] * edge_weight, row, dim=-2, dim_size=x.size(0), reduce='sum')
    twostep = scatter(onestep[col] * edge_weight, row, dim=-2, dim_size=x.size(0), reduce='sum')

    return twostep

#@profile(precision=4, stream=open('g3.log','w+'))
def cal_g_gradient3(edge_index, x, edge_weight=None, sigma1=None, sigma2=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    row, col = edge_index[0], edge_index[1]

    onestep = scatter((x[col] - x[row]) * edge_weight, row, dim=-2, dim_size=x.size(0), reduce='add')
    twostep = scatter(onestep[col] * edge_weight, row, dim=-2, dim_size=x.size(0), reduce='add')

    # onestep = scatter((x[col] - x[row]) * edge_weight, col, dim=-2, dim_size=x.size(0), reduce='add')
    # twostep = scatter(onestep[col] * edge_weight, col, dim=-2, dim_size=x.size(0), reduce='add')
    twostep = feature_norm(twostep)
    return twostep

def cal_g_gradient6(edge_index, x, edge_weight=None, sigma1=None, sigma2=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    row, col = edge_index[0], edge_index[1]

    onestep = scatter((x[col] - x[row]) * edge_weight, row, dim=-2, dim_size=x.size(0), reduce='add')
    # twostep = scatter(onestep[col] * edge_weight, row, dim=-2, dim_size=x.size(0), reduce='add')

    # onestep = scatter((x[col] - x[row]) * edge_weight, col, dim=-2, dim_size=x.size(0), reduce='add')
    # twostep = scatter(onestep[col] * edge_weight, col, dim=-2, dim_size=x.size(0), reduce='add')
    onestep = feature_norm(onestep)
    return onestep

def calAx(edge_index, x, edge_weight=None, sigma=0):
    row, col = edge_index[0], edge_index[1]

    d = x[col] - x[row]
    d2 = torch.sum(d * d, dim=1)

    coe = torch.exp(- d2 / 2) * (1 / (torch.sqrt(2 * 3.141592) * sigma))
    result = scatter(x[col] * coe, row, dim=-2,
                     dim_size=x.size(0), reduce='sum')
    return result


def cal_g_gradient4(edge_index, x, edge_weight=None, sigma1=None, sigma2=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    ones = torch.ones((edge_index.size(1),), dtype=dtype,
                                 device=edge_index.device)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(ones, col, dim=0, dim_size=num_nodes)
    deg_inv = deg.pow(-1)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)
    # caculate gradient
    onestep = scatter(deg_inv[row].view(-1, 1) * (x[col] - x[row]), row, dim=-2, dim_size=x.size(0), reduce='add')
    twostep = scatter(onestep[col] * edge_weight, row, dim=-2, dim_size=x.size(0), reduce='add')
    twostep = feature_norm(twostep)
    return twostep

# 正态分布计算系数
def cal_g_gradient5(edge_index, x, edge_weight=None, sigma1=None, sigma2=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    deg_inv = deg.pow_(-1)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)
    # caculate gradient
    gra = deg_inv[row].view(-1, 1) * (x[col] - x[row])
    avg_gra = scatter(gra, row, dim=-2, dim_size=x.size(0), reduce='add')
    abs_agra = torch.norm(avg_gra, p=2, dim=1)
    s = compute_D(x[row], x[col])
    s = (torch.sum(avg_gra[row] * avg_gra[col], dim=1) / (abs_agra[row] * abs_agra[col] + 1e-6))
    r = scatter(s.view(-1, 1), row, dim=-2, dim_size=x.size(0), reduce='add')
    coe = s.view(-1, 1) / (r[row] + 1e-6)
    result = scatter(avg_gra[row] * coe * (deg_inv_sqrt[col] * deg_inv_sqrt[row]).view(-1, 1), col, dim=-2, dim_size=x.size(0), reduce='sum')

    return result

#@profile(precision=4, stream=open('ggat.log','w+'))
def cal_g_gradient_gat(edge_index, x, gat, edge_weight=None, sigma1=None, sigma2=None, num_nodes=None, dropout=0.1, improved=False,
             add_self_loops=True, dtype=None):

    row, col = edge_index[0], edge_index[1]
    avg_gra = scatter((x[col] - x[row]) * edge_weight, row, dim=-2, dim_size=x.size(0), reduce='add')
    # result = gat(avg_gra, edge_index)
    # result = scatter(avg_gra[col] * (deg_inv_sqrt[col] * deg_inv_sqrt[row]).view(-1, 1), row, dim=-2,dim_size=x.size(0), reduce='sum')
    result = gat(avg_gra, edge_index)
    return result

# def cal_Bx(edge_index, x, g, gamma,  edge_weight=None, num_nodes=None, improved=False,
#              add_self_loops=True, dtype=None):
#     if edge_weight is None:
#         edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
#                                  device=edge_index.device)
#     num_nodes = maybe_num_nodes(edge_index, num_nodes)
#     row, col = edge_index[0], edge_index[1]
#     deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
#     deg_inv_sqrt = deg.pow_(-0.5)
#     deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
#     absx = torch.norm(x, p=2, dim=1)
#     s = torch.sum(x[row] * x[col], dim=1) / (absx[row] * absx[col] + 1e-6)
#     s = s * (deg_inv_sqrt[col] * deg_inv_sqrt[row])
#     r = scatter(s.view(-1, 1), row, dim=-2, dim_size=x.size(0), reduce='sum')
#     coe = s / (r[row] + 1e-6).view(-1)
#     # result = scatter((x[col] - x[row] - gamma * g[row]) * coe.view(-1,1), col, dim=-2, dim_size=g.size(0), reduce='sum')
#     result = scatter((x[col] - x[row] - gamma * g[row]) * coe.view(-1, 1), row, dim=-2, dim_size=g.size(0),
#                      reduce='sum')
#     return result
#
# def cal_Q(edge_index, x, edge_weight=None, num_nodes=None, improved=False,
#              add_self_loops=True, dtype=None):
#     if edge_weight is None:
#         edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
#                                  device=edge_index.device)
#     num_nodes = maybe_num_nodes(edge_index, num_nodes)
#     row, col = edge_index[0], edge_index[1]
#     deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
#     deg_inv_sqrt = deg.pow_(-0.5)
#     deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
#     dx = x[col] - x[row]
#     absdx = torch.norm(dx, p=2, dim=1)
#     return ((torch.sum(dx * dx, dim=1) / (absdx + 0.000001)) * deg_inv_sqrt[col] * deg_inv_sqrt[row])
#
#
#
# def calculate_PQ(edge_index, x, Q, edge_weight=None, num_nodes=None, improved=False,
#              add_self_loops=True, dtype=None):
#     if edge_weight is None:
#         edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
#                                  device=edge_index.device)
#     num_nodes = maybe_num_nodes(edge_index, num_nodes)
#     row, col = edge_index[0], edge_index[1]
#     deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
#     deg_inv_sqrt = deg.pow_(-0.5)
#     deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
#     absx = torch.norm(x, p=2, dim=1)
#     absdx = torch.norm(x[col] - x[row], p=2, dim=1)
#     return ((torch.sum(x[row] * x[col], dim=1) / (absx[row] * absx[col])) * deg_inv_sqrt[col]
#             * deg_inv_sqrt[row] / (absdx + 0.000001)).view(-1, 1) * (x[col] - x[row]) * Q


def read_config(args):
    # specify the model family

    fileNamePath = os.path.split(os.path.realpath(__file__))[0]
    yamlPath = os.path.join(fileNamePath, 'prediction/config/{}/{}.yaml'.format(args.configfile, args.times))
    print(yamlPath)
    with open(yamlPath, 'r', encoding='utf-8') as f:
        cont = f.read()
        # TODO
        config_dict = yaml.safe_load(cont)['g3'][args.dataset]

    if args.gpu == -1:
        device = torch.device('cpu')
    elif args.gpu >= 0:
        if torch.cuda.is_available():
            device = torch.device('cuda', int(args.gpu))
        else:
            print("cuda is not available, please set 'gpu' -1")
    for key, value in config_dict.items():
        args.__setattr__(key, value)

    return args

def feature_norm(fea):
    device = fea.device
    epsilon = 1e-12
    fea_sum = torch.norm(fea, p=1, dim=1)
    fea_inv = 1 / np.maximum(fea_sum.detach().cpu().numpy(), epsilon)
    fea_inv = torch.from_numpy(fea_inv).to(device)
    fea_norm = fea * fea_inv.view(-1, 1)

    return fea_norm

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

def prob_to_adj(mx, threshold):
    mx = np.triu(mx, 1)
    mx += mx.T
    (row, col) = np.where(mx > threshold)
    adj = sp.coo_matrix((np.ones(row.shape[0]), (row,col)), shape=(mx.shape[0], mx.shape[0]), dtype=np.int64)
    adj = sparse_mx_to_sparse_tensor(adj)
    return adj

