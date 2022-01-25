import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import random_split
import copy

import torch_geometric.transforms as T
from torch_geometric.datasets import PPI, Planetoid, Coauthor, Amazon, Reddit
from torch_geometric.utils import remove_isolated_nodes
from utils import *


def load_citation(dataname, path, opt='none'):
    if opt == 'gcn':
        dataset = Planetoid(path, dataname, transform=T.NormalizeFeatures())
    else:
        dataset = Planetoid(path, dataname)
    
    data = dataset[0]
   
    if dataname == 'citeseer':
        data.edge_index, data.edge_attr, node_mask = remove_isolated_nodes(data.edge_index, data.edge_attr)
        data.x = data.x[node_mask]
        data.y = data.y[node_mask]
        data.train_mask = data.train_mask[node_mask]
        data.val_mask = data.val_mask[node_mask]
        data.test_mask = data.test_mask[node_mask]
        
    soft_y = label_propagation(data)[0]
    # print(soft_y)
    if opt == 'add-label':
        # num_node_features = dataset.num_node_features + dataset.num_classes
        # dataset.num_node_features = num_node_features

        temp_y = torch.zeros(data.num_nodes, dataset.num_classes)
        temp_y[data.train_mask] = soft_y[data.train_mask]
        data.x = torch.cat((data.x, temp_y), dim=1)
    elif opt == 'label-only':
        # dataset.num_node_features = dataset.num_classes
        temp_y = torch.zeros(data.num_nodes, dataset.num_classes)
        # temp_y[data.train_mask | data.val_mask] = soft_y[data.train_mask | data.val_mask]
        # data.x = temp_y
        data.x = soft_y

    return dataset, data

