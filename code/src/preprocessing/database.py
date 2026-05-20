"""DuckDB join engine for assembling the nodes and edges tables."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd

log = logging.getLogger(__name__)


def build_node_edge_tables(
    transactions_df: pd.DataFrame,
    source_code_df: pd.DataFrame,
    fraud_labels_df: pd.DataFrame,
    benign_seed_addresses: Iterable[str],
) -> duckdb.DuckDBPyConnection:
    """Build the in-memory ``nodes`` and ``edges`` tables.

    ``nodes`` columns: ``address``, ``contract_text`` (Base64 source or empty),
    ``y`` (1 fraud, 0 benign), ``is_labeled`` (True if Forta-fraud or
    explicit benign seed).
    ``edges`` columns: ``src``, ``dst``, ``value``, ``tx_hash``.
    """
    con = duckdb.connect(database=":memory:")

    benign_seed_df = pd.DataFrame(
        sorted({a.lower() for a in benign_seed_addresses}),
        columns=["address"],
    )
    benign_seed_df["y_benign"] = 0
    fraud_df = fraud_labels_df[["address"]].copy()
    fraud_df["y_fraud"] = 1

    con.register("transactions_df", transactions_df)
    con.register("source_code_df", source_code_df)
    con.register("benign_seed_df", benign_seed_df)
    con.register("fraud_df", fraud_df)

    all_addresses = pd.concat(
        [transactions_df["from_address"], transactions_df["to_address"]]
    ).dropna().str.lower().unique()
    con.register(
        "all_addresses_df",
        pd.DataFrame(all_addresses, columns=["address"]),
    )

    edges_select = "from_address AS src, to_address AS dst"
    if "value" in transactions_df.columns:
        edges_select += ", CAST(value AS DOUBLE) AS value"
    else:
        edges_select += ", CAST(0.0 AS DOUBLE) AS value"
    if "tx_hash" in transactions_df.columns:
        edges_select += ", tx_hash"
    else:
        edges_select += ", CAST(NULL AS VARCHAR) AS tx_hash"

    con.execute(f"CREATE TABLE edges AS SELECT {edges_select} FROM transactions_df")
    edge_count = con.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    log.info("Created edges table with %d rows", edge_count)

    con.execute(
        """
        CREATE TABLE nodes AS
        SELECT
            a.address,
            COALESCE(sc.source_code, '') AS contract_text,
            CASE
                WHEN f.y_fraud = 1 THEN 1
                ELSE 0
            END AS y,
            CASE
                WHEN f.y_fraud = 1 THEN TRUE
                WHEN b.y_benign = 0 THEN TRUE
                ELSE FALSE
            END AS is_labeled
        FROM all_addresses_df a
        LEFT JOIN source_code_df sc ON a.address = sc.address
        LEFT JOIN fraud_df f ON a.address = f.address
        LEFT JOIN benign_seed_df b ON a.address = b.address
        """
    )
    total, fraud, labelled = con.execute(
        "SELECT COUNT(*), SUM(CASE WHEN y=1 THEN 1 ELSE 0 END),"
        " SUM(CASE WHEN is_labeled THEN 1 ELSE 0 END) FROM nodes"
    ).fetchone()
    log.info(
        "Created nodes table: %d total, %d fraud, %d labelled",
        total,
        fraud or 0,
        labelled or 0,
    )
    return con


def export_tables(con: duckdb.DuckDBPyConnection, out_dir: Path) -> dict[str, Path]:
    """Export ``nodes`` and ``edges`` tables to Parquet files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes_path = out_dir / "nodes.parquet"
    edges_path = out_dir / "edges.parquet"
    con.execute(f"COPY nodes TO '{nodes_path}' (FORMAT PARQUET)")
    con.execute(f"COPY edges TO '{edges_path}' (FORMAT PARQUET)")
    log.info("Exported nodes to %s and edges to %s", nodes_path, edges_path)
    return {"nodes": nodes_path, "edges": edges_path}
