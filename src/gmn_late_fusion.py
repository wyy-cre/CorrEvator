import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn.conv import GATConv, GCNConv
from torch_geometric.nn.pool import global_mean_pool
from torch_geometric.utils import to_dense_batch


NUM_GMNLAYER = 5  # 图匹配层数
NODE_DIM = 32  # 节点维度


class GraphMatchingNet(nn.Module):
    def __init__(self, input_dim=300, node_dim=NODE_DIM):
        super().__init__()
        # Encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, node_dim)
        )
        # 图匹配层
        self.gmn_layers = nn.ModuleList([
            GMNLayerWithNorm(node_dim) for _ in range(NUM_GMNLAYER)
        ])
        # Aggregator
        self.aggregator = BinaryMLPAggregator(node_dim)
        # self.aggregator = CosineWithTempAggregator(node_dim)
        self.gat = GATConv(node_dim, node_dim // 4, heads=4, concat=True, dropout=0.3)
        self.dropout = nn.Dropout(0.3)
        self.norm1 = nn.LayerNorm(node_dim)

    def forward(self, data):
        data.x = self.node_encoder(data.x)
        for _ in range(NUM_GMNLAYER):
            x, edge_index, edge_attr = data.x, data.edge_index.to(torch.int64), data.edge_attr
            h1 = self.gat(x, edge_index, edge_attr)
            data.x = self.norm1(x + self.dropout(h1))
        for layer in self.gmn_layers:
            data = layer(data)
        similarity, logits = self.aggregator(data)
        return similarity, logits


class GMNLayerWithNorm(nn.Module):
    def __init__(self, node_dim=32, heads=4, dropout=0.3):
        super().__init__()

        self.norm1 = nn.LayerNorm(node_dim)
        self.dropout = nn.Dropout(dropout)
        self.cross_attn = nn.MultiheadAttention(node_dim, heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(node_dim)

    def forward(self, data: Data):
        x, edge_index, edge_attr = data.x, data.edge_index.to(torch.int64), data.edge_attr
        mask0 = data.node_graph_id == 0
        mask1 = data.node_graph_id == 1
        h0, pad0 = to_dense_batch(x[mask0], data.batch[mask0])
        h1_, pad1 = to_dense_batch(x[mask1], data.batch[mask1])
        seq0, seq1 = h0.transpose(0, 1), h1_.transpose(0, 1)
        key0, key1 = ~pad0, ~pad1
        o0, _ = self.cross_attn(seq0, seq1, seq1, key_padding_mask=key1)
        o1, _ = self.cross_attn(seq1, seq0, seq0, key_padding_mask=key0)
        out0 = o0.transpose(0, 1)[pad0]
        out1 = o1.transpose(0, 1)[pad1]

        h2 = x.clone()
        h2[mask0] = out0
        h2[mask1] = out1
        data.x = self.norm2(x + self.dropout(h2))
        return data


class BinaryMLPAggregator(nn.Module):
    def __init__(self, node_dim=32, hidden_dim=32):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(4 * node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, data):
        x0 = global_mean_pool(data.x[data.node_graph_id == 0], data.batch[data.node_graph_id == 0])
        x1 = global_mean_pool(data.x[data.node_graph_id == 1], data.batch[data.node_graph_id == 1])
        features = torch.cat([x0, x1, torch.abs(x0 - x1), x0 * x1], dim=1)
        logits = self.classifier(features)
        # probs = torch.softmax(logits, dim=1)
        similarity = F.cosine_similarity(x0, x1, dim=1)
        similarity = nn.Sigmoid()(similarity)
        return similarity, logits
