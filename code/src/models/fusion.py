"""Hybrid GNN-LLM fusion classifier."""
from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from .graph_encoder import GraphEncoder


class FusionClassifier(nn.Module):
    """Fuses GraphSAGE node embeddings with CodeBERT pooled embeddings.

    Forward computes graph embeddings over the entire transaction graph once
    per batch, gathers the per-sample slice, concatenates with the CodeBERT
    pooler output, then projects through a fusion MLP into class logits.
    """

    def __init__(
        self,
        graph_feat_dim: int,
        codebert_model: nn.Module,
        gnn_hidden_dim: int = 128,
        gnn_out_dim: int = 64,
        fusion_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.gnn = GraphEncoder(
            in_dim=graph_feat_dim,
            hidden_dim=gnn_hidden_dim,
            out_dim=gnn_out_dim,
            dropout=dropout,
        )
        self.lang_model = codebert_model
        for param in self.lang_model.parameters():
            param.requires_grad = False
        lang_hidden = self.lang_model.config.hidden_size
        self.fusion_layer = nn.Linear(gnn_out_dim + lang_hidden, fusion_dim)
        self.classifier = nn.Linear(fusion_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def train(self, mode: bool = True):
        super().train(mode)
        self.lang_model.eval()
        return self

    def forward(
        self,
        graph_data: Data,
        batch: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        all_graph_embs = self.gnn(graph_data.x, graph_data.edge_index)
        node_idx = batch["node_idx"]
        graph_embs = all_graph_embs[node_idx]

        with torch.no_grad():
            lang_outputs = self.lang_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
        lang_embs = lang_outputs.pooler_output

        fused = torch.cat([graph_embs, lang_embs], dim=-1)
        fused = self.dropout(fused)
        hidden = F.relu(self.fusion_layer(fused))
        hidden = self.dropout(hidden)
        return self.classifier(hidden)
