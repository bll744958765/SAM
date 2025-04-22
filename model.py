

import torch.nn.parallel
import warnings

warnings.filterwarnings("ignore")
import torch as torch
import torch.nn.parallel
import warnings


warnings.filterwarnings("ignore")
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, knn_graph
import random
import os
import time
import numpy as np



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

class SAM(nn.Module):
    """
        GCN with positional encoder and auxiliary tasks
    """

    def __init__(self, num_features_c, num_features_x, k, emb_dim, res=True):
        super(SAM, self).__init__()
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
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, x):
        return self.mlp(x)

