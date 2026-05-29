"""Step 4: join everything via DuckDB and serialize the PyG graph."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd
import torch

from config import PATHS
from data.addresses import BENIGN_ADDRESSES
from src.collection.forta_labels import load_forta_labels
from src.collection.transactions import load_transactions_csv
from src.preprocessing.database import build_node_edge_tables, export_tables
from src.preprocessing.graph import build_pyg_graph

log = logging.getLogger("04_build_dataset")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--transactions-csv",
        default=str(PATHS["interim"] / "transactions.csv"),
    )
    p.add_argument(
        "--benign-csv",
        default=str(PATHS["interim"] / "benign_contracts.csv"),
    )
    p.add_argument(
        "--fraud-csv",
        default=str(PATHS["interim"] / "fraud_contracts.csv"),
    )
    return p.parse_args()


def _read_or_empty(path: Path) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path)
        df["address"] = df["address"].astype(str).str.lower()
        return df[["address", "source_code"]]
    log.warning("Missing source CSV: %s (continuing with empty table)", path)
    return pd.DataFrame(columns=["address", "source_code"])


def main() -> None:
    args = parse_args()
    transactions = load_transactions_csv(Path(args.transactions_csv))

    benign_df = _read_or_empty(Path(args.benign_csv))
    fraud_df = _read_or_empty(Path(args.fraud_csv))
    source_code_df = pd.concat([benign_df, fraud_df], ignore_index=True).drop_duplicates(
        subset="address", keep="last"
    )

    # Use our curated fraud_contracts.csv addresses as ground-truth fraud labels.
    # Forta labels are merged in as additional signal where available.
    fraud_contracts_df = pd.read_csv(Path(args.fraud_csv))
    fraud_contracts_df["address"] = fraud_contracts_df["address"].astype(str).str.lower()
    forta_labels = load_forta_labels()
    combined_fraud_labels = pd.concat(
        [fraud_contracts_df[["address"]], forta_labels[["address"]]],
        ignore_index=True,
    ).drop_duplicates(subset="address")

    con = build_node_edge_tables(
        transactions_df=transactions,
        source_code_df=source_code_df,
        fraud_labels_df=combined_fraud_labels,
        benign_seed_addresses=BENIGN_ADDRESSES,
    )
    paths = export_tables(con, PATHS["processed"])

    nodes_df = con.execute("SELECT * FROM nodes").df()
    edges_df = con.execute("SELECT * FROM edges").df()

    graph_data, addr2idx = build_pyg_graph(nodes_df, edges_df)
    graph_path = PATHS["processed"] / "graph.pt"
    addr2idx_path = PATHS["processed"] / "addr2idx.json"
    torch.save(graph_data, graph_path)
    addr2idx_path.write_text(json.dumps(addr2idx))

    log.info("Wrote nodes -> %s, edges -> %s, graph -> %s", paths["nodes"], paths["edges"], graph_path)


if __name__ == "__main__":
    main()
