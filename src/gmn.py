import torch
import torch_scatter
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

    def forward(self, data):
        data.x = self.node_encoder(data.x)
        for layer in self.gmn_layers:
            data = layer(data)
        similarity, logits = self.aggregator(data)
        return similarity, logits


# class GMNLayer(nn.Module):
#     def __init__(self, node_dim=32):
#         super().__init__()
#         # 聚合
#         # self.GcnConv1 = GCNConv(node_dim, node_dim, improved=True, add_self_loops=True)
#         self.GatConv1 = GATConv(node_dim, int(node_dim/4), heads=4, concat=True, add_self_loops=True)
#         # 交叉注意力
#         self.cross_attn_0to1 = nn.MultiheadAttention(node_dim, 4)
#         self.cross_attn_1to0 = nn.MultiheadAttention(node_dim, 4)
#         self.gate = nn.Sequential(
#             nn.Linear(node_dim * 2, node_dim),
#             nn.Sigmoid()
#         )
#
#     def forward(self, data):
#         # message_inner = self.GcnConv1(data.x, data.edge_index.to(torch.int64), data.edge_attr)
#         message_inner = self.GatConv1(data.x, data.edge_index.to(torch.int64), data.edge_attr)
#
#         # 交叉注意力
#         message_outer = []
#         for graph in data.to_data_list():
#             # 分开子图
#             x0 = graph.x[graph.node_graph_id == 0]
#             x1 = graph.x[graph.node_graph_id == 1]
#             # 转换为注意力层需要的格式(seq_len, batch, embed_dim)
#             x0 = x0.unsqueeze(1)
#             x1 = x1.unsqueeze(1)
#             # 计算注意力
#             x0_updated, _ = self.cross_attn_0to1(x0, x1, x1)
#             x0_updated = x0_updated.squeeze(1)
#             x1_updated, _ = self.cross_attn_1to0(x1, x0, x0)
#             x1_updated = x1_updated.squeeze(1)
#             message_outer.append(x0_updated)
#             message_outer.append(x1_updated)
#         message_outer = torch.cat(message_outer, dim=0).to(data.x.device)
#
#         gate = self.gate(torch.cat([message_inner, message_outer], dim=1))
#         data.x = gate * message_inner + (1 - gate) * message_outer
#         return data


class GMNLayerWithNorm(nn.Module):
    def __init__(self, node_dim=32, heads=4, dropout=0.3):
        super().__init__()
        self.gcn = GCNConv(node_dim, node_dim, improved=True, add_self_loops=True)
        self.gat = GATConv(node_dim, node_dim // heads, heads=heads, concat=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(node_dim)
        self.dropout = nn.Dropout(dropout)
        self.cross_attn = nn.MultiheadAttention(node_dim, heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(node_dim)

    def forward(self, data: Data):
        x, edge_index, edge_attr = data.x, data.edge_index.to(torch.int64), data.edge_attr
        h1 = self.gat(x, edge_index, edge_attr)
        # h1 = self.gcn(x, edge_index, edge_attr)
        h1 = self.norm1(x + self.dropout(h1))

        mask0 = data.node_graph_id == 0
        mask1 = data.node_graph_id == 1
        h0, pad0 = to_dense_batch(h1[mask0], data.batch[mask0])
        h1_, pad1 = to_dense_batch(h1[mask1], data.batch[mask1])
        seq0, seq1 = h0.transpose(0, 1), h1_.transpose(0, 1)
        key0, key1 = ~pad0, ~pad1
        o0, _ = self.cross_attn(seq0, seq1, seq1, key_padding_mask=key1)
        o1, _ = self.cross_attn(seq1, seq0, seq0, key_padding_mask=key0)
        out0 = o0.transpose(0, 1)[pad0]
        out1 = o1.transpose(0, 1)[pad1]

        h2 = x.clone()
        h2[mask0] = out0
        h2[mask1] = out1
        data.x = self.norm2(h1 + self.dropout(h2))
        return data


# class Aggregator(nn.Module):
#     def __init__(self, input_dim=32, graph_dim=128):
#         super().__init__()
#         # self.mlp = nn.Linear(input_dim, graph_dim)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, data):
#         # data.x = self.mlp(data.x)
#         x0 = data.x[data.node_graph_id == 0]
#         x1 = data.x[data.node_graph_id == 1]
#         x0 = global_mean_pool(x0, data.batch[data.node_graph_id == 0])
#         x1 = global_mean_pool(x1, data.batch[data.node_graph_id == 1])
#         # 计算相似度
#         similarity = F.cosine_similarity(x0, x1, dim=1)
#         logits = torch.stack([-similarity, similarity], dim=1)
#         # probs = torch.softmax(logits, dim=1)
#         similarity = self.sigmoid(similarity)
#         return similarity, logits


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


# class CosineWithTempAggregator(nn.Module):
#     def __init__(self, node_dim=32):
#         super().__init__()
#         self.tau = nn.Parameter(torch.tensor(1.0))
#
#     def forward(self, data):
#         x0 = global_mean_pool(data.x[data.node_graph_id == 0], data.batch[data.node_graph_id == 0])
#         x1 = global_mean_pool(data.x[data.node_graph_id == 1], data.batch[data.node_graph_id == 1])
#         similarity = F.cosine_similarity(x0, x1, dim=1) / self.tau
#         logits = torch.stack([-similarity, similarity], dim=1)
#         # probs = torch.softmax(logits, dim=1)
#         similarity = nn.Sigmoid()(similarity)
#         return similarity, logits
