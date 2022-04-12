import numpy as np
import scipy.sparse as sp
import torch
import sys, os, re
import pickle as pkl
import networkx as nx
from normalization import fetch_normalization, row_normalize
from time import perf_counter
from sklearn.model_selection import ShuffleSplit
import torch.nn.functional as F
# import matplotlib.pyplot as plt


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def preprocess_citation(adj, features, args, normalization="FirstOrderGCN", ectd_data=''):
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj, ectd_data, args)
    features = row_normalize(features)
    return adj, features


def preprocess_test(adj, normalization="FirstOrderGCN"):
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    return adj


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_citation(dataset_str="cora", args=None, normalization="AugNormAdj", cuda=True, ectd_data=''):
    """
    Load Citation Networks Datasets.
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str.lower(), names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    adj, features = preprocess_citation(adj, features, args, normalization, ectd_data)

    # porting to pytorch
    features = torch.FloatTensor(np.array(features.todense())).float()
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    # print("adj: {}\nfeatures: {}\nlabels: {}\nidx_train: {}\n".format(adj, features, labels, idx_train))
    return adj, features, labels, idx_train, idx_val, idx_test


def full_load_citation(dataset_str="cora", args=None, normalization="AugNormAdj", cuda=True, ectd_data=''):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    adj, features = preprocess_citation(adj, features, args, normalization, ectd_data)

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    if cuda:
        if cuda:
            features = features.cuda()
            adj = adj.cuda()
            labels = labels.cuda()
            train_mask = train_mask.cuda()
            val_mask = val_mask.cuda()
            test_mask = test_mask.cuda()

    return adj, features, labels, train_mask, val_mask, test_mask


def load_webkb(dataset_name, splits_file_path=None, train_percentage=None, val_percentage=None,
               normalization="AugNormAdj", cuda=True, ectd_data='', args=None):

    graph_adjacency_list_file_path = os.path.join('new_data', dataset_name, 'out1_graph_edges.txt')
    graph_node_features_and_labels_file_path = os.path.join('new_data', dataset_name,
                                                            f'out1_node_feature_label.txt')

    G = nx.DiGraph()
    graph_node_features_dict = {}
    graph_labels_dict = {}

    if dataset_name == 'film':
        with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
            graph_node_features_and_labels_file.readline()
            for line in graph_node_features_and_labels_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 3)
                assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                feature_blank = np.zeros(932, dtype=np.uint8)
                feature_blank[np.array(line[1].split(','), dtype=np.uint16)] = 1
                graph_node_features_dict[int(line[0])] = feature_blank
                graph_labels_dict[int(line[0])] = int(line[2])
    else:
        with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
            graph_node_features_and_labels_file.readline()
            for line in graph_node_features_and_labels_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 3)
                assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                graph_node_features_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.uint8)
                graph_labels_dict[int(line[0])] = int(line[2])

    with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
        graph_adjacency_list_file.readline()
        for line in graph_adjacency_list_file:
            line = line.rstrip().split('\t')
            assert (len(line) == 2)
            if int(line[0]) not in G:
                G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                           label=graph_labels_dict[int(line[0])])
            if int(line[1]) not in G:
                G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                           label=graph_labels_dict[int(line[1])])
            G.add_edge(int(line[0]), int(line[1]))

    adj = nx.adjacency_matrix(G, sorted(G.nodes()))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = np.array(
        [features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
    labels = np.array(
        [label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])

    adj, features = preprocess_citation(adj, features, args, normalization, ectd_data)

    if splits_file_path:
        with np.load(splits_file_path) as splits_file:
            train_mask = splits_file['train_mask']
            val_mask = splits_file['val_mask']
            test_mask = splits_file['test_mask']
    else:
        train_and_val_index, test_index = next(
            ShuffleSplit(n_splits=1, train_size=train_percentage + val_percentage).split(
                np.empty_like(labels), labels))
        train_index, val_index = next(ShuffleSplit(n_splits=1, train_size=train_percentage).split(
            np.empty_like(labels[train_and_val_index]), labels[train_and_val_index]))
        train_index = train_and_val_index[train_index]
        val_index = train_and_val_index[val_index]

        train_mask = np.zeros_like(labels)
        train_mask[train_index] = 1
        val_mask = np.zeros_like(labels)
        val_mask[val_index] = 1
        test_mask = np.zeros_like(labels)
        test_mask[test_index] = 1

    features = torch.FloatTensor(features).float()

    labels = torch.LongTensor(labels)
    labels = F.one_hot(labels)
    labels = torch.max(labels, dim=1)[1]
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()

    train_mask = torch.BoolTensor(train_mask)
    val_mask = torch.BoolTensor(val_mask)
    test_mask = torch.BoolTensor(test_mask)

    train_idx = torch.LongTensor(np.squeeze(np.where(train_mask)))
    val_idx = torch.LongTensor(np.squeeze(np.where(val_mask)))
    test_idx = torch.LongTensor(np.squeeze(np.where(test_mask)))

    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        train_idx = train_idx.cuda()
        val_idx = val_idx.cuda()
        test_idx = test_idx.cuda()

    # print("adj: {}\nfeatures: {}\nlabels: {}\nidx_train: {}\n".format(adj, features, labels, train_idx))
    return adj, features, labels, train_idx, val_idx, test_idx


def load_test(dataset_str="triangle", normalization="AugNormAdj", cuda=True, pinv=False):
    """
    Load Test Networks Datasets.
    Triangle / Rectangle
    """
    graph = nx.Graph()

    if dataset_str == 'triangle':
        graph.add_nodes_from([1, 2, 3])
        graph.add_edges_from([
            (1, 2), (1, 3),
            (2, 1), (2, 3),
            (3, 1), (3, 2)
        ])

    elif dataset_str == 'rectangle':
        graph.add_nodes_from([1, 2, 3, 4])
        graph.add_edges_from([
            (1, 2), (1, 3),
            (2, 1), (2, 4),
            (3, 1), (3, 4),
            (4, 2), (4, 3)
        ])
    graph = nx.erdos_renyi_graph(n=6, p=0.3, seed=42)
    adj = nx.adjacency_matrix(graph)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # print("init adj: {}".format(adj))
    adj = preprocess_test(adj, normalization)

    # porting to pytorch
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()

    if cuda:
        adj = adj.cuda()

    return adj


def sgc_precompute(features, adj, degree):
    print("pre-compute adj:\n {}".format(adj))
    t = perf_counter()
    # ssgc like - init residual
    alpha = 0.5

    ori = alpha * features
    emb = alpha * features
    former = alpha * features
    # standard sgc
    for i in range(degree):
        # former layer residual
        # emb = alpha * features
        features = torch.spmm(adj, features)
        # ssgc w/o degree
        emb = emb + (1. - alpha) * features
        # ssgc
        # emb = emb + (1 - alpha) * features / degree
        # former = features
        # emb = emb + features
        # emb = (1. - alpha) * features
        # emb = features
        # features = emb + torch.spmm(adj, features)
    # emb += ori
    precompute_time = perf_counter() - t

    # Normalization? - max/arctan/f2
    # max = torch.max(emb, 1)[0]
    # emb /= max.repeat(emb.shape[1], 1).t()
    # emb = (emb - torch.min(emb)) / (torch.max(emb) - torch.min(emb))
    # emb = torch.atan(emb)
    # emb = F.normalize(emb, p=2, dim=1)

    return emb, precompute_time


def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda: torch.cuda.manual_seed(seed)


def loadRedditFromNPZ(dataset_dir):
    adj = sp.load_npz(dataset_dir + "reddit_adj.npz")
    data = np.load(dataset_dir + "reddit.npz")

    return adj, data['feats'], data['y_train'], data['y_val'], data['y_test'], data['train_index'], data['val_index'], \
           data['test_index']


def load_reddit_data(data_path="data/", normalization="AugNormAdj", cuda=True):
    adj, features, y_train, y_val, y_test, train_index, val_index, test_index = loadRedditFromNPZ("data/")
    labels = np.zeros(adj.shape[0])
    labels[train_index] = y_train
    labels[val_index] = y_val
    labels[test_index] = y_test
    adj = adj + adj.T

    train_adj = adj[train_index, :][:, train_index]
    features = torch.FloatTensor(np.array(features))
    features = (features - features.mean(dim=0)) / features.std(dim=0)
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    train_adj = adj_normalizer(train_adj)
    train_adj = sparse_mx_to_torch_sparse_tensor(train_adj).float()
    labels = torch.LongTensor(labels)
    if cuda:
        adj = adj.cuda()
        train_adj = train_adj.cuda()
        features = features.cuda()
        labels = labels.cuda()
    return adj, train_adj, features, labels, train_index, val_index, test_index
