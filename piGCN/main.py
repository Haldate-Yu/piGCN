import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from torch_sparse import SparseTensor
from torch_geometric.utils import to_dense_adj, dense_to_sparse

from utils import *
from models import *
from dataset import *


def train(model, data):
    model.train()

    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    _, pred = torch.max(out[data.train_mask], dim=1)
    correct = (pred == data.y[data.train_mask]).sum().item()
    acc = correct / data.train_mask.sum().item()

    return loss.item(), acc


def test(model, data):
    model.eval()

    out = model(data)
    loss = criterion(out[data.test_mask], data.y[data.test_mask])

    _, pred = torch.max(out[data.test_mask], dim=1)
    correct = (pred == data.y[data.test_mask]).sum().item()
    acc = correct / data.test_mask.sum().item()

    val_loss = criterion(out[data.val_mask], data.y[data.val_mask])

    _, pred = torch.max(out[data.val_mask], dim=1)
    correct = (pred == data.y[data.val_mask]).sum().item()
    val_acc = correct / data.val_mask.sum().item()

    return val_loss.item(), val_acc, loss.item(), acc


if __name__ == '__main__':
    test_accs = []

    for run in range(20):
        set_seeds(run)
        print('')
        print(f'Run {run: 02d}')
        print('')

        # 加载数据集
        dataset, data = load_citation(dataname='cora', path='./data', opt='none')

        # 实例化模型
        # opt = none 
        model = GCN(dataset=dataset, hidden=16, num_layers=2)
        # opt = add-label
        # model = GCNwithLabel(dataset=dataset, hidden=16, num_layers=2)
        # opt = label-only
        # model = GCNLabelOnly(dataset=dataset, hidden=16, num_layers=2)
        # model = SGC(dataset=dataset, hidden=16, num_layers=2, k=2)

        # 转换为cpu或cuda格式
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # device = torch.device('cpu')
        # print(device)
        model.to(device)
        data = data.to(device)

        # 定义损失函数和优化器
        criterion = nn.NLLLoss().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        best_val_acc = final_test_acc = 0
        for epoch in range(100):
            loss, acc = train(model, data)
            if epoch % 20 == 0:
                print('Epoch {:03d} train_loss: {:.4f} train_acc: {:.4f}'.format(epoch, loss, acc))

            if epoch > 50 and epoch % 10 == 0:
                val_loss, val_acc, test_loss, test_acc = test(model, data)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    final_test_acc = test_acc

                print("test_loss: {:.4f} test_acc: {:.4f}".format(test_loss, test_acc))
        print('----------')
        print(f'final test acc this run: {final_test_acc:.4f}')
        print('----------')
        test_accs.append(final_test_acc)

test_acc = torch.tensor(test_accs)
print(f'Final Test: {test_acc.mean():.4f} +- {test_acc.std():.4f}')
