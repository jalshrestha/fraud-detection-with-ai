"""Two-layer GraphSAGE encoder used by the fusion model."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphEncoder(nn.Module):
    """GraphSAGE encoder.

    Architecture follows the paper exactly:
        SAGEConv(in_dim,  hidden_dim) -> ReLU -> Dropout(p)
        SAGEConv(hidden_dim, out_dim)
    """

    def __init__(
        self,
        in_dim: int = 3,
        hidden_dim: int = 128,
        out_dim: int = 64,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x, edge_index))
        h = self.dropout(h)
        h = self.conv2(h, edge_index)
        return h
