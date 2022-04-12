import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import load_citation, load_webkb, sgc_precompute, set_seed
from models import get_model
from metrics import accuracy
import pickle as pkl
from args import get_citation_args
from time import perf_counter

# Arguments
args = get_citation_args()

if args.tuned:
    if args.model == "SGC":
        with open("{}-tuning/{}.txt".format(args.model, args.dataset), 'rb') as f:
            args.weight_decay = pkl.load(f)['weight_decay']
            print("using tuned weight decay: {}".format(args.weight_decay))
    else:
        raise NotImplemented

# setting random seeds
set_seed(args.seed, args.cuda)

if args.dataset in {'cora', 'citeseer', 'pubmed'}:
    data_path = './pinv-dataset/ectd_' + args.dataset + '_'
    if args.using_vg: data_path += 'vg' + str(args.vg) + '_'
    data_path += args.method + '_' + args.adj + '.txt'
    print(data_path)

    adj, features, labels, idx_train, idx_val, idx_test = load_citation(args.dataset, args, args.normalization, args.cuda,
                                                                    data_path)
elif args.dataset in {'cornell', 'film', 'texas', 'wisconsin'}:
    # test
    data_path = './pinv-dataset/ectd_' + args.dataset + '_'
    if args.using_vg: data_path += 'vg' + str(args.vg) + '_'
    data_path += args.method + '_' + args.adj + '.txt'

    splits = './splits/' + args.dataset + '_split_0.6_0.2_' + str(args.split) + '.npz'

    adj, features, labels, idx_train, idx_val, idx_test = load_webkb(args.dataset, splits,
                                                                     None, None, args.normalization, args.cuda, data_path,
                                                                     args)


model = get_model(args.model, features.size(1), labels.max().item() + 1, args.hidden, args.dropout, args.cuda)

if args.model == "SGC" or args.model == "MSGC":
    features, precompute_time = sgc_precompute(features, adj, args.degree)
    print("{:.4f}s".format(precompute_time))


def train_regression(model,
                     train_features, train_labels,
                     val_features, val_labels,
                     epochs=args.epochs, weight_decay=args.weight_decay,
                     lr=args.lr, dropout=args.dropout):
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    t = perf_counter()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(train_features)
        loss_train = F.cross_entropy(output, train_labels)
        loss_train.backward()
        optimizer.step()
    train_time = perf_counter() - t

    with torch.no_grad():
        model.eval()
        output = model(val_features)
        acc_val = accuracy(output, val_labels)

    return model, acc_val, train_time


def test_regression(model, test_features, test_labels):
    model.eval()
    return accuracy(model(test_features), test_labels)


if args.model == "SGC" or args.model == "IDSGC" or args.model == "MSGC":
    model, acc_val, train_time = train_regression(model, features[idx_train], labels[idx_train], features[idx_val],
                                                  labels[idx_val],
                                                  args.epochs, args.weight_decay, args.lr, args.dropout)
    acc_test = test_regression(model, features[idx_test], labels[idx_test])

print("Validation Accuracy: {:.4f} Test Accuracy: {:.4f}".format(acc_val, acc_test))
print("Pre-compute time: {:.4f}s, train time: {:.4f}s, total: {:.4f}s".format(precompute_time, train_time,
                                                                              precompute_time + train_time))
