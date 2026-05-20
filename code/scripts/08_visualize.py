"""Step 8: render dataset and graph figures into data/processed/figures/."""
from __future__ import annotations

import argparse
import logging

import _bootstrap  # noqa: F401
import pandas as pd

from config import PATHS
from src.visualization.dataset_plots import (
    plot_account_types,
    plot_label_balance,
    plot_transaction_values,
)
from src.visualization.graph_plots import plot_subgraph

log = logging.getLogger("08_visualize")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-fraud", type=int, default=20)
    p.add_argument("--max-benign", type=int, default=100)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    nodes_df = pd.read_parquet(PATHS["processed"] / "nodes.parquet")
    edges_df = pd.read_parquet(PATHS["processed"] / "edges.parquet")

    plot_transaction_values(edges_df, PATHS["figures"] / "transaction_values.png")
    plot_account_types(nodes_df, PATHS["figures"] / "account_types.png")
    plot_label_balance(nodes_df, PATHS["figures"] / "label_balance.png")
    plot_subgraph(
        nodes_df,
        edges_df,
        PATHS["figures"] / "subgraph.png",
        max_fraud=args.max_fraud,
        max_benign=args.max_benign,
    )
    log.info("Figures written to %s", PATHS["figures"])


if __name__ == "__main__":
    main()
