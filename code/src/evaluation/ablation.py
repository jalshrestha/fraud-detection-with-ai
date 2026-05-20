"""Ablation-study helpers: GNN-only and LLM-only baseline models."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.graph_encoder import GraphEncoder


class GNNOnlyClassifier(nn.Module):
    """GraphSAGE features fed directly to a linear head; no source-code input."""

    def __init__(
        self,
        in_dim: int = 3,
        hidden_dim: int = 128,
        out_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.gnn = GraphEncoder(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, dropout=dropout)
        self.classifier = nn.Linear(out_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph_data, batch):
        all_embs = self.gnn(graph_data.x, graph_data.edge_index)
        embs = self.dropout(all_embs[batch["node_idx"]])
        return self.classifier(embs)


class LLMOnlyClassifier(nn.Module):
    """CodeBERT pooler output fed directly to a linear head; no graph input."""

    def __init__(
        self,
        codebert_model: nn.Module,
        hidden_dim: int = 768,
        num_classes: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.lang_model = codebert_model
        for param in self.lang_model.parameters():
            param.requires_grad = False
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def train(self, mode: bool = True):
        super().train(mode)
        self.lang_model.eval()
        return self

    def forward(self, graph_data, batch):
        with torch.no_grad():
            outputs = self.lang_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
        embs = self.dropout(outputs.pooler_output)
        return self.classifier(embs)


def build_gnn_only_model(in_dim: int, **kwargs) -> GNNOnlyClassifier:
    return GNNOnlyClassifier(in_dim=in_dim, **kwargs)


def build_llm_only_model(codebert_model: nn.Module, **kwargs) -> LLMOnlyClassifier:
    return LLMOnlyClassifier(codebert_model=codebert_model, **kwargs)
