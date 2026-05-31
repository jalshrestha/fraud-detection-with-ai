"""PyTorch Geometric graph construction with enriched node features."""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree

log = logging.getLogger(__name__)

# Number of node features produced by build_pyg_graph — update config to match.
NODE_FEAT_DIM = 8


def _value_features(
    edges_df: pd.DataFrame,
    addr2idx: dict[str, int],
    num_nodes: int,
) -> torch.Tensor:
    """Return per-node value statistics: mean_in_value, mean_out_value,
    log1p_total_value, unique_in_neighbours, unique_out_neighbours."""
    # safe numeric conversion of value column
    vals = pd.to_numeric(edges_df["value"], errors="coerce").fillna(0.0)
    edges_df = edges_df.copy()
    edges_df["_val"] = vals

    # --- mean in-value per destination ---
    mean_in = np.zeros(num_nodes, dtype=np.float32)
    grp_in = edges_df.groupby("dst")["_val"].mean()
    for addr, v in grp_in.items():
        if addr in addr2idx:
            mean_in[addr2idx[addr]] = float(v)

    # --- mean out-value per source ---
    mean_out = np.zeros(num_nodes, dtype=np.float32)
    grp_out = edges_df.groupby("src")["_val"].mean()
    for addr, v in grp_out.items():
        if addr in addr2idx:
            mean_out[addr2idx[addr]] = float(v)

    # --- log1p total value (in + out) ---
    total = np.zeros(num_nodes, dtype=np.float32)
    for addr, v in edges_df.groupby("dst")["_val"].sum().items():
        if addr in addr2idx:
            total[addr2idx[addr]] += float(v)
    for addr, v in edges_df.groupby("src")["_val"].sum().items():
        if addr in addr2idx:
            total[addr2idx[addr]] += float(v)
    log_total = np.log1p(total)

    # --- unique in-neighbours ---
    uniq_in = np.zeros(num_nodes, dtype=np.float32)
    for addr, cnt in edges_df.groupby("dst")["src"].nunique().items():
        if addr in addr2idx:
            uniq_in[addr2idx[addr]] = float(cnt)

    # --- unique out-neighbours ---
    uniq_out = np.zeros(num_nodes, dtype=np.float32)
    for addr, cnt in edges_df.groupby("src")["dst"].nunique().items():
        if addr in addr2idx:
            uniq_out[addr2idx[addr]] = float(cnt)

    return torch.stack([
        torch.from_numpy(mean_in),
        torch.from_numpy(mean_out),
        torch.from_numpy(log_total),
        torch.from_numpy(uniq_in),
        torch.from_numpy(uniq_out),
    ], dim=1)


def build_pyg_graph(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    split: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> Tuple[Data, dict[str, int]]:
    """Construct a PyG ``Data`` object and an address-to-index mapping.

    Node features (8-dim):
      [in_degree, out_degree, total_degree,
       mean_in_value, mean_out_value, log1p_total_value,
       unique_in_neighbours, unique_out_neighbours]
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
    in_deg  = degree(edge_index[1], num_nodes=num_nodes)
    out_deg = degree(edge_index[0], num_nodes=num_nodes)

    # 3 structural + 5 value/neighbour features = 8 total
    struct_feats = torch.stack([in_deg, out_deg, in_deg + out_deg], dim=1).float()
    value_feats  = _value_features(edges_df, addr2idx, num_nodes)
    x = torch.cat([struct_feats, value_feats], dim=1)   # (N, 8)

    # log-normalise value columns to bring them to similar scale as degree
    for col in [3, 4, 5, 6, 7]:
        x[:, col] = torch.log1p(x[:, col])

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
    n_val   = int(split[1] * n)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask   = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask  = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[permuted[:n_train]] = True
    val_mask[permuted[n_train : n_train + n_val]] = True
    test_mask[permuted[n_train + n_val :]] = True

    data = Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)
    data.has_contract = has_contract
    data.is_labeled   = is_labeled
    data.train_mask   = train_mask
    data.val_mask     = val_mask
    data.test_mask    = test_mask

    log.info(
        "Built graph: %d nodes, %d edges, %d features, %d eligible, split %d/%d/%d",
        num_nodes, edge_index.size(1), x.size(1), n,
        train_mask.sum().item(), val_mask.sum().item(), test_mask.sum().item(),
    )
    return data, addr2idx
