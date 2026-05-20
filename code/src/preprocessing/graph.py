"""PyTorch Geometric graph construction with degree-based node features."""
from __future__ import annotations

import logging
from typing import Tuple

import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree

log = logging.getLogger(__name__)


def build_pyg_graph(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    split: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> Tuple[Data, dict[str, int]]:
    """Construct a PyG ``Data`` object and an address-to-index mapping.

    Node features are ``[in_degree, out_degree, in_degree + out_degree]`` per
    the paper. Train/val/test masks are computed over the subset of nodes
    that have both verified source code and an explicit fraud/benign label.
    """
    if not abs(sum(split) - 1.0) < 1e-6:
        raise ValueError(f"split must sum to 1.0, got {split}")

    addr2idx = {addr: i for i, addr in enumerate(nodes_df["address"].tolist())}

    edges_df = edges_df[
        edges_df["src"].isin(addr2idx) & edges_df["dst"].isin(addr2idx)
    ].copy()
    src_idx = edges_df["src"].map(addr2idx).to_numpy()
    dst_idx = edges_df["dst"].map(addr2idx).to_numpy()
    edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)

    num_nodes = len(addr2idx)
    in_deg = degree(edge_index[1], num_nodes=num_nodes)
    out_deg = degree(edge_index[0], num_nodes=num_nodes)
    x = torch.stack([in_deg, out_deg, in_deg + out_deg], dim=1).float()

    y = torch.tensor(nodes_df["y"].to_numpy(), dtype=torch.long)
    has_contract = torch.tensor(
        nodes_df["contract_text"].astype(str).str.len() > 10,
        dtype=torch.bool,
    )
    is_labeled = torch.tensor(
        nodes_df["is_labeled"].astype(bool).to_numpy(),
        dtype=torch.bool,
    )

    eligible_mask = has_contract & is_labeled
    eligible_indices = torch.nonzero(eligible_mask, as_tuple=False).flatten()

    generator = torch.Generator().manual_seed(seed)
    permuted = eligible_indices[torch.randperm(len(eligible_indices), generator=generator)]
    n = len(permuted)
    n_train = int(split[0] * n)
    n_val = int(split[1] * n)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[permuted[:n_train]] = True
    val_mask[permuted[n_train : n_train + n_val]] = True
    test_mask[permuted[n_train + n_val :]] = True

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        num_nodes=num_nodes,
    )
    data.has_contract = has_contract
    data.is_labeled = is_labeled
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    log.info(
        "Built graph: %d nodes, %d edges, %d eligible (with code and label), "
        "split %d/%d/%d",
        num_nodes,
        edge_index.size(1),
        n,
        train_mask.sum().item(),
        val_mask.sum().item(),
        test_mask.sum().item(),
    )
    return data, addr2idx
