import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GATConv, SAGEConv, SGConv


class GCN(nn.Module):
    def __init__(self, dataset, hidden=16, num_layers=2):
        super(GCN, self).__init__()

        self.num_layers = num_layers
        self.convs = nn.ModuleList()

        self.convs.append(GCNConv(dataset.num_node_features, hidden))
        for i in range(self.num_layers - 2):
            self.convs.append(GCNConv(hidden, hidden))

        self.convs.append(GCNConv(hidden, dataset.num_classes))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)

        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)


class GCNwithLabel(nn.Module):
    def __init__(self, dataset, hidden=16, num_layers=2):
        super(GCNwithLabel, self).__init__()

        self.num_layers = num_layers
        self.convs = nn.ModuleList()

        self.convs.append(GCNConv(dataset.num_node_features+dataset.num_classes, hidden))
        for i in range(self.num_layers - 2):
            self.convs.append(GCNConv(hidden, hidden))

        self.convs.append(GCNConv(hidden, dataset.num_classes))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)

        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)


class GCNLabelOnly(nn.Module):
    def __init__(self, dataset, hidden=16, num_layers=2):
        super(GCNLabelOnly, self).__init__()

        self.num_layers = num_layers
        self.convs = nn.ModuleList()

        self.convs.append(GCNConv(dataset.num_classes, hidden))
        for i in range(self.num_layers - 2):
            self.convs.append(GCNConv(hidden, hidden))

        self.convs.append(GCNConv(hidden, dataset.num_classes))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)

        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)


class SGC(nn.Module):
    def __init__(self, dataset, hidden=16, num_layers=2, k=1):
        super(SGC, self).__init__()

        self.hidden = hidden
        self.num_layers = num_layers
        self.k = k

        self.convs = nn.ModuleList()
        self.convs.append(SGConv(dataset.num_features, dataset.num_classes, K=self.k, cached=True))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.convs[0](x, edge_index, edge_attr)
        # x = F.relu(x)
        # x = F.dropout(x, p=0.5, training=self.training)

        return F.log_softmax(x, dim=1)

class piSGC(nn.Module):
    def __init__(self, dataset, hidden=16, num_layers=2, k=1):
        super(piSGC, self).__init__()

        self.hidden = hidden
        self.num_layers = num_layers
        self.k = k

        self.convs = nn.ModuleList()
        self.convs.append(SGConv(dataset.num_features, dataset.num_classes, K=self.k, cached=True, add_self_loops=False))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.convs[0](x, edge_index, edge_attr)
        # x = F.relu(x)
        # x = F.dropout(x, p=0.5, training=self.training)

        return F.log_softmax(x, dim=1)


