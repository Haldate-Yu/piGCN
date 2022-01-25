import os
import random
import math
import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from torch_sparse import SparseTensor


def label_propagation(data, k=3, alpha=0.2):
    adj = to_dense_adj(data.edge_index)
    # train_idx = data.train_mask.nonzero(as_tuple=True)[0]
    
    y0 = torch.zeros(data.y.shape[0], data.y.max().item() + 1)
    y = F.one_hot(data.y, data.y.max().item() + 1).type(torch.FloatTensor)
    y0[data.train_mask] = y[data.train_mask]
    y = y0

    for _ in range(k):
        y = torch.matmul(adj, y)
        # y[data.train_mask] = y0[data.train_mask]
        y = (1 - alpha) * y + alpha * y0
        y.clamp_(0., 1.)

    return y


def adj_pinv(dataname, data, topk_nodes=100):
    os.makedirs("./pinv-dataset", exist_ok=True)
    path = './pinv-dataset/{}.txt'.format(dataname)
    nnode = data.x.size(0)

    if os.path.exists(path):
        pinv = np.loadtxt(path, delimiter=',')
    else:
        # adj = to_dense_adj(data.edge_index).numpy()
        nnode = data.x.size(0)
        adj = SparseTensor(row=data.edge_index[0], col=data.edge_index[1], sparse_sizes=(nnode, nnode)).to_dense().numpy()
        # print(adj)
        degree = np.diag(adj.sum(axis=0))
        laplacian = degree - adj
        pinv = np.linalg.pinv(laplacian)

        np.savetxt(path, pinv, fmt='%f', delimiter=',')
    
    # alpha * adj + (1 - alpha) * pinv ?
    pinv[pinv <= 0] = 0
    
    ## method 1
    # make it symmetric
    L_i = np.apply_along_axis(topk, 1, pinv, topk=topk_nodes)
    L_i = np.triu(L_i)
    L_i += L_i.T - np.diag(L_i.diagonal())
    
    ## method 2
    # take the whole topk
    alpha = 0.1 # remain percentage
    
    # topk = int(pinv[np.nonzero(pinv)].size * alpha)
    # topk_num = np.partition(pinv[np.nonzero(pinv)], -topk)[-topk]

    # pinv[pinv < topk_num] = 0
    # L_i = pinv

    # assert L_i.T == L_i
    L_i = np.squeeze(np.squeeze(L_i))

    sparse_pinv = torch.from_numpy(L_i).to_sparse()
    edge_index, edge_attr = sparse_pinv.indices(), sparse_pinv.values().to(torch.float32)

    return edge_index, edge_attr


def topk(array, topk=80):
    index = array.argsort()[-topk:][::-1]
    a = set(index)
    b = set(list(range(array.shape[0])))
    array[list(b.difference(a))] = 0

    return array

def set_seeds(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
