"""Ablation-study helpers: GNN-only, LLM-only, GCN, GAT, attention-fusion, and RF baselines."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, GATConv

from src.models.graph_encoder import GraphEncoder


# ──────────────────────────────────────────────
# Single-modality baselines
# ──────────────────────────────────────────────

class GNNOnlyClassifier(nn.Module):
    """GraphSAGE → linear head (no source code)."""
    def __init__(self, in_dim=8, hidden_dim=128, out_dim=64, num_classes=2, dropout=0.5):
        super().__init__()
        self.gnn = GraphEncoder(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, dropout=dropout)
        self.classifier = nn.Linear(out_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph_data, batch):
        embs = self.dropout(self.gnn(graph_data.x, graph_data.edge_index)[batch["node_idx"]])
        return self.classifier(embs)


class GCNOnlyClassifier(nn.Module):
    """Two-layer GCN → linear head (no source code)."""
    def __init__(self, in_dim=8, hidden_dim=128, out_dim=64, num_classes=2, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.classifier = nn.Linear(out_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph_data, batch):
        h = F.relu(self.conv1(graph_data.x, graph_data.edge_index))
        h = self.dropout(h)
        h = self.conv2(h, graph_data.edge_index)
        embs = self.dropout(h[batch["node_idx"]])
        return self.classifier(embs)


class GATOnlyClassifier(nn.Module):
    """Two-layer GAT (4 heads) → linear head (no source code)."""
    def __init__(self, in_dim=8, hidden_dim=128, out_dim=64, num_classes=2, dropout=0.5, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim // heads, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim, out_dim, heads=1, concat=False, dropout=dropout)
        self.classifier = nn.Linear(out_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph_data, batch):
        h = F.elu(self.conv1(graph_data.x, graph_data.edge_index))
        h = self.dropout(h)
        h = self.conv2(h, graph_data.edge_index)
        embs = self.dropout(h[batch["node_idx"]])
        return self.classifier(embs)


class LLMOnlyClassifier(nn.Module):
    """CodeBERT pooler → linear head (no graph)."""
    def __init__(self, codebert_model, hidden_dim=768, num_classes=2, dropout=0.5):
        super().__init__()
        self.lang_model = codebert_model
        for p in self.lang_model.parameters():
            p.requires_grad = False
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def train(self, mode=True):
        super().train(mode)
        self.lang_model.eval()
        return self

    def forward(self, graph_data, batch):
        with torch.no_grad():
            out = self.lang_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        return self.classifier(self.dropout(out.pooler_output))


# ──────────────────────────────────────────────
# Attention-based fusion (replaces concatenation)
# ──────────────────────────────────────────────

class AttentionFusionClassifier(nn.Module):
    """GraphSAGE + frozen CodeBERT fused via cross-attention instead of concatenation."""
    def __init__(self, graph_feat_dim, codebert_model, gnn_hidden_dim=128,
                 gnn_out_dim=64, fusion_dim=128, num_classes=2, dropout=0.5):
        super().__init__()
        from src.models.graph_encoder import GraphEncoder
        self.gnn = GraphEncoder(in_dim=graph_feat_dim, hidden_dim=gnn_hidden_dim,
                                out_dim=gnn_out_dim, dropout=dropout)
        self.lang_model = codebert_model
        for p in self.lang_model.parameters():
            p.requires_grad = False

        lang_dim = self.lang_model.config.hidden_size  # 768
        # Project both modalities to same dim for attention
        self.q_proj = nn.Linear(gnn_out_dim, fusion_dim)
        self.k_proj = nn.Linear(lang_dim, fusion_dim)
        self.v_proj = nn.Linear(lang_dim, fusion_dim)
        self.out_proj = nn.Linear(fusion_dim + gnn_out_dim, fusion_dim)
        self.classifier = nn.Linear(fusion_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def train(self, mode=True):
        super().train(mode)
        self.lang_model.eval()
        return self

    def forward(self, graph_data, batch):
        all_embs = self.gnn(graph_data.x, graph_data.edge_index)
        g = all_embs[batch["node_idx"]]            # (B, 64)

        with torch.no_grad():
            lang_out = self.lang_model(input_ids=batch["input_ids"],
                                       attention_mask=batch["attention_mask"])
        c = lang_out.pooler_output                 # (B, 768)

        # Scaled dot-product attention: graph queries, code keys/values
        q = self.q_proj(g)                         # (B, D)
        k = self.k_proj(c)
        v = self.v_proj(c)
        scale = q.size(-1) ** 0.5
        attn = torch.softmax((q * k) / scale, dim=-1)
        attended = attn * v                        # (B, D)

        fused = torch.cat([g, attended], dim=-1)   # (B, 64+D)
        fused = F.relu(self.out_proj(self.dropout(fused)))
        return self.classifier(self.dropout(fused))


# ──────────────────────────────────────────────
# Factory functions
# ──────────────────────────────────────────────

def build_gnn_only_model(in_dim, **kw):   return GNNOnlyClassifier(in_dim=in_dim, **kw)
def build_gcn_only_model(in_dim, **kw):   return GCNOnlyClassifier(in_dim=in_dim, **kw)
def build_gat_only_model(in_dim, **kw):   return GATOnlyClassifier(in_dim=in_dim, **kw)
def build_llm_only_model(codebert_model, **kw): return LLMOnlyClassifier(codebert_model, **kw)
def build_attention_fusion_model(graph_feat_dim, codebert_model, **kw):
    return AttentionFusionClassifier(graph_feat_dim, codebert_model, **kw)
