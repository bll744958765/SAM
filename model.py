


import torch.nn.parallel
import torch.backends.cudnn as cudnn
import warnings

warnings.filterwarnings("ignore")
import torch as torch
import torch.nn.parallel
import warnings
import torch.distributions.normal as normal_dist

warnings.filterwarnings("ignore")
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, knn_graph
import random
import os
import time
import numpy as np
from scipy.stats import wasserstein_distance
from math import radians, cos, sin, asin, sqrt
import math


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


start = time.perf_counter()
time.sleep(2)


def knn_to_adj(knn, n):
    adj_matrix = torch.zeros(n, n, dtype=float)  # lil_matrix((n, n), dtype=float)
    for i in range(len(knn[0])):
        tow = knn[0][i]
        fro = knn[1][i]
        adj_matrix[tow, fro] = 1  # should be bidectional?
    return adj_matrix.T


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r * 1000


def newDistance(a, b, nd_dist="euclidean"):
    # Distance options are ["great_circle" (2D only), "euclidean", "wasserstein" (for higher-dimensional coordinate embeddings)]

    if a.shape[0] == 2:
        x1, y1 = a[0], a[1]
        x2, y2 = b[0], b[1]
        if nd_dist == "euclidean":
            d = math.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))
        else:  # nd_dist="great_circle"
            d = haversine(x1, y1, x2, y2)
    if a.shape[0] == 3:
        x1, y1, z1 = a[0], a[1], a[2]
        x2, y2, z2 = b[0], b[1], b[2]
        d = math.sqrt(math.pow(x2 - x1, 2) +
                      math.pow(y2 - y1, 2) +
                      math.pow(z2 - z1, 2) * 1.0)
    if a.shape[0] > 3:
        if nd_dist == "wasserstein":
            d = wasserstein_distance(a.reshape(-1).detach(), b.reshape(-1).detach())
            # d = sgw_cpu(a.reshape(1,-1).detach(),b.reshape(1,-1).detach())
        else:
            d = torch.pow(a.reshape(1, 1, -1) - b.reshape(1, 1, -1), 2).sum(2)
    return d


# Helper function for edge weights
def makeEdgeWeight(x, edge_index):
    to = edge_index[0]
    fro = edge_index[1]
    edge_weight = []
    for i in range(len(to)):
        edge_weight.append(newDistance(x[to[i]], x[fro[i]]))  # probably want to do inverse distance eventually
    max_val = max(edge_weight)
    rng = max_val - min(edge_weight)
    edge_weight = [(max_val - elem) / rng for elem in edge_weight]
    return torch.Tensor(edge_weight)


class PEGCN(nn.Module):
    """
        GCN with positional encoder and auxiliary tasks
    """

    def __init__(self, num_features_c, num_features_x, k, emb_dim, res=True):
        super(PEGCN, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.emb_dim = emb_dim
        self.k = k
        self.res = res
        self.softplus = nn.Softplus()

        self.fc_coords = nn.Linear(num_features_c, self.emb_dim)
        self.fc_attri = nn.Linear(num_features_x, self.emb_dim)

        self.gcn1 = GCNConv(num_features_c, self.emb_dim)
        self.gcn2 = GCNConv(self.emb_dim, self.emb_dim)
        self.gcn3 = GCNConv(self.emb_dim, self.emb_dim)

        self.fc1 = nn.Linear(num_features_x, self.emb_dim)
        self.fc2 = nn.Linear(self.emb_dim, self.emb_dim)
        self.fc3 = nn.Linear(self.emb_dim, self.emb_dim)

        self.decoder = Decoder(self.emb_dim * 2, 2)

    def forward(self, c, x):
        x = x.float()
        c = c.float()

        edge_index = knn_graph(c, k=self.k).to(self.device)

        row, col = edge_index
        distances = torch.norm(c[row] - c[col], p=2, dim=1)
        edge_weight = torch.exp(-distances / 2)  # 使用距离的指数函数
        edge_weight = torch.where(edge_weight > 10, torch.tensor(0).to(self.device), edge_weight)

        coords0 = self.fc_coords(c)
        attri0 = self.fc_attri(x)

        gcn_output1 = F.relu(F.dropout(self.gcn1(c, edge_index, edge_weight), training=self.training))
        mlp_output1 = F.relu(F.dropout(self.fc1(x), training=self.training))

        if self.res:
            gcn_output1 = torch.add(coords0, gcn_output1)
            mlp_output1 = torch.add(attri0, mlp_output1)

        gcn_output2 = F.relu(F.dropout(self.gcn2(gcn_output1, edge_index, edge_weight), training=self.training))
        mlp_output2 = F.relu(F.dropout(self.fc2(mlp_output1), training=self.training))

        if self.res:
            gcn_output2 = torch.add(gcn_output1, gcn_output2)
            mlp_output2 = torch.add(mlp_output1, mlp_output2)




        gcn_output3 = F.relu(self.gcn3(gcn_output2, edge_index, edge_weight))
        mlp_output3 = F.relu(self.fc3(mlp_output2))
        if self.res:
            gcn_output3 = torch.add(gcn_output2, gcn_output3)
            mlp_output3 = torch.add(mlp_output2, mlp_output3)


        combined = torch.cat([gcn_output3, mlp_output3], dim=1)
        output = self.decoder(combined)
        mean = output[:, 0]
        std = output[:, 1]
        sigma = 0.2 + 0.8 * self.softplus(std)

        return mean.reshape(-1, 1), sigma.reshape(-1, 1), gcn_output3, mlp_output3


class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Decoder, self).__init__()
        # self.fc = nn.Linear(input_dim, output_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),

            nn.ReLU(),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, x):
        return self.mlp(x)


def log_likelihood_loss(y_true, y_pred, variance):
    y_true = y_true.cpu()
    y_pred = y_pred.cpu()
    variance = variance.cpu()

    normal = normal_dist.Normal(loc=y_pred, scale=torch.tensor(variance))
    log_likelihood = normal.log_prob(y_true)

    loss = -torch.nanmean(log_likelihood)
