import torch
import torch
from torch.nn import Parameter, ReLU
from torch_geometric.nn.inits import zeros
import torch.nn.functional as F
from torch import Tensor
from torch_sparse import SparseTensor, matmul
from torch_scatter import scatter_add
from torch_scatter import gather_csr, scatter
from utils import cal_g_gradient1, cal_g_gradient2, cal_g_gradient3, cal_g_gradient4, cal_g_gradient5, cal_g_gradient_gat
from torch_geometric.nn.conv import MessagePassing, GATConv, GCNConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.nn.dense.linear import Linear
from utils import feature_norm

from typing import Optional, Tuple
import numpy as np

class ReactionNet(MessagePassing):

    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, args, in_channels: int, out_channels: int, bias: bool = False,
                cached: bool = False, add_self_loops: bool = True,
                 normalize: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.k = args.k
        self.alpha = args.alpha
        self.beta = args.beta
        self.gamma = args.gamma
        self.sigma1 = args.sigma1
        self.sigma2 = args.sigma2
        self.drop = args.drop
        self.dropout = args.dropout
        self.calg = 'g3'
        if args.dataset == 'pubmed':
            self.calg = 'g4'
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self._cached_edge_index = None
        self._cached_adj_t = None
        self.lin1 = Linear(in_channels, args.hidden, bias=False, weight_initializer='glorot')
        self.lin2 = Linear(args.hidden, out_channels, bias=False, weight_initializer='glorot')
        self.relu = ReLU()
        self.reg_params = list(self.lin1.parameters())
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):

        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None


    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        if self.normalize:
            edgei = edge_index
            edgew = edge_weight
            cache = self._cached_edge_index
            if cache is None:
                edge_index, edge_weight = gcn_norm(  # yapf: disable
                    edgei, edgew, x.size(self.node_dim), False,
                    self.add_self_loops, dtype=x.dtype)
                edge_index2, edge_weight2 = gcn_norm(  # yapf: disable
                    edgei, edgew, x.size(self.node_dim), False,
                    False, dtype=x.dtype)

                if self.cached:
                    self._cached_edge_index = (edge_index, edge_weight)
            else:
                edge_index, edge_weight = cache[0], cache[1]
            ew = edge_weight.view(-1, 1)
            ew2 = edge_weight2.view(-1, 1)



        # preprocess
        if self.drop == 'True':
            x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.lin2(x)

        h = x
        for k in range(self.k):

            if self.calg == 'g3' or self.calg == 'cal_gradient_2':  # TODO
                g = cal_g_gradient3(edge_index2, x, edge_weight=ew2, sigma1=self.sigma1, sigma2=self.sigma2)
            elif self.calg == 'g1':
                g = cal_g_gradient1(edge_index2, x, edge_weight=ew2, sigma1=self.sigma1, sigma2=self.sigma2)
            elif self.calg == 'g2':
                g = cal_g_gradient2(edge_index2, x, edge_weight=ew2, sigma1=self.sigma1, sigma2=self.sigma2)
            elif self.calg == 'g4':
                g = cal_g_gradient4(edge_index2, x, edge_weight=ew2, sigma1=self.sigma1, sigma2=self.sigma2)
            elif self.calg == 'g5':
                g = cal_g_gradient5(edge_index2, x, edge_weight=ew2, sigma1=self.sigma1, sigma2=self.sigma2)
            elif self.calg == 'ggat':
                g = cal_g_gradient_gat(edge_index2, x, self.gat1, edge_weight=ew2, sigma1=self.sigma1, sigma2=self.sigma2)

            adj = torch.sparse_coo_tensor(edge_index, edge_weight, [x.size(0), x.size(0)])
            Ax = torch.spmm(adj, x)
            Gx = torch.spmm(adj, g)
            x = self.alpha * h + (1 - self.alpha - self.beta) * x  \
                + self.beta * Ax \
                + self.beta * self.gamma * Gx

        out = F.log_softmax(x, dim=-1)

        return out


    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        # return edge_weight
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(K={self.k}, alpha={self.alpha})'

