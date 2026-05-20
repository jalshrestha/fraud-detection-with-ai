"""Step 1: load transactions and restrict them to the seed neighbourhood.

The pipeline assumes ``data/raw/transactions.csv`` is a BigQuery export with
columns ``from_address``, ``to_address``, ``value`` (and optionally ``hash``,
``block_timestamp``, etc.). The script lower-cases addresses, filters to
the union of benign + fraud seed lists, and saves both the filtered
transactions and the discovered common-address pool.
"""
from __future__ import annotations

import argparse
import json
import logging

import _bootstrap  # noqa: F401  (sys.path side-effect)
import pandas as pd

from config import PATHS
from data.addresses import BENIGN_ADDRESSES, FRAUD_ADDRESSES
from src.collection.transactions import (
    filter_transactions_to_seeds,
    find_common_contracts,
    load_transactions_csv,
)

log = logging.getLogger("01_fetch_transactions")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--transactions-csv",
        default=str(PATHS["raw"] / "transactions.csv"),
        help="Path to BigQuery export CSV (default: data/raw/transactions.csv).",
    )
    p.add_argument(
        "--restrict-to-seeds",
        action="store_true",
        help="If set, drop transactions that do not touch a benign or fraud seed address.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    transactions = load_transactions_csv(args.transactions_csv)
    if args.restrict_to_seeds:
        transactions = filter_transactions_to_seeds(
            transactions,
            BENIGN_ADDRESSES,
            FRAUD_ADDRESSES,
        )
    common = find_common_contracts(transactions)

    out_tx = PATHS["interim"] / "transactions.csv"
    out_addresses = PATHS["interim"] / "common_addresses.json"
    transactions.to_csv(out_tx, index=False)
    out_addresses.write_text(json.dumps(common, indent=2))

    log.info(
        "Wrote %d transactions to %s and %d common addresses to %s",
        len(transactions),
        out_tx,
        len(common),
        out_addresses,
    )


if __name__ == "__main__":
    main()
