"""Transaction collection and common-contract discovery.

The paper's needle-first strategy starts from the two seed lists
(benign + fraud contracts) and pulls every transaction that has either
address as ``from_address`` or ``to_address``. The union of unique
addresses that appears in those transactions is the "common contract"
candidate pool for which we later attempt to fetch verified source code.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

log = logging.getLogger(__name__)


TRANSACTION_REQUIRED_COLS = ("from_address", "to_address", "value")


def load_transactions_csv(path: Path) -> pd.DataFrame:
    """Load a pre-saved BigQuery export and standardize address casing."""
    df = pd.read_csv(path)
    missing = [c for c in TRANSACTION_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Transactions file {path} is missing required columns: {missing}"
        )
    df["from_address"] = df["from_address"].astype(str).str.lower()
    df["to_address"] = df["to_address"].astype(str).str.lower()
    if "hash" in df.columns:
        df = df.rename(columns={"hash": "tx_hash"})
    log.info("Loaded %d transactions from %s", len(df), path)
    return df


def filter_transactions_to_seeds(
    transactions: pd.DataFrame,
    benign_addresses: Iterable[str],
    fraud_addresses: Iterable[str],
) -> pd.DataFrame:
    """Restrict transactions to those that touch a seed address on either side."""
    seeds = {a.lower() for a in benign_addresses} | {a.lower() for a in fraud_addresses}
    mask = transactions["from_address"].isin(seeds) | transactions["to_address"].isin(seeds)
    filtered = transactions.loc[mask].copy()
    log.info(
        "Kept %d / %d transactions involving a seed address (%d seeds total)",
        len(filtered),
        len(transactions),
        len(seeds),
    )
    return filtered


def fetch_transactions(
    csv_path: Path,
    benign_addresses: Optional[Iterable[str]] = None,
    fraud_addresses: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Load transactions and (optionally) restrict them to the seed neighbourhood."""
    transactions = load_transactions_csv(csv_path)
    if benign_addresses is None and fraud_addresses is None:
        return transactions
    return filter_transactions_to_seeds(
        transactions,
        benign_addresses or [],
        fraud_addresses or [],
    )


def find_common_contracts(transactions: pd.DataFrame) -> list[str]:
    """Return the deduplicated union of addresses that appear in any transaction."""
    unique = pd.concat(
        [transactions["from_address"], transactions["to_address"]]
    ).dropna().str.lower().unique()
    log.info("Discovered %d unique addresses in the transaction neighbourhood", len(unique))
    return sorted(set(unique))
