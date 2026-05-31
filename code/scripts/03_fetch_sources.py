"""Step 3: fetch verified Solidity source code from Etherscan for both seed lists.

Outputs two CSVs in ``data/interim/``:
  - ``benign_contracts.csv``: Base64-encoded source for the benign seed list.
  - ``fraud_contracts.csv``: Base64-encoded source for the fraud seed list
    plus any Forta-flagged address that also appears in the transaction
    neighbourhood (``data/interim/common_addresses.json`` if present).

The script reads ``ETHERSCAN_API_KEY`` from the environment via ``config``.
"""
from __future__ import annotations

import argparse
import json
import logging

import _bootstrap  # noqa: F401

from config import ETHERSCAN_API_KEY, PATHS
from data.addresses import BENIGN_ADDRESSES, FRAUD_ADDRESSES
from src.collection.etherscan import fetch_many_source_codes
from src.collection.forta_labels import load_forta_labels

log = logging.getLogger("03_fetch_sources")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--include-common",
        action="store_true",
        help=(
            "Also try to fetch source for every address in "
            "data/interim/common_addresses.json that is Forta-flagged."
        ),
    )
    p.add_argument(
        "--max-fraud",
        type=int,
        default=None,
        help="Cap the number of fraud addresses to fetch (useful while iterating).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not ETHERSCAN_API_KEY:
        raise SystemExit(
            "ETHERSCAN_API_KEY is not set. Export it in your shell before running this script."
        )

    benign_df = fetch_many_source_codes(
        BENIGN_ADDRESSES,
        api_key=ETHERSCAN_API_KEY,
        progress_desc="Fetching benign contracts",
    )
    benign_out = PATHS["interim"] / "benign_contracts.csv"
    benign_df.to_csv(benign_out, index=False)
    log.info("Saved %d benign contracts to %s", len(benign_df), benign_out)

    fraud_seed = set(FRAUD_ADDRESSES)
    forta_df = load_forta_labels()
    fraud_pool = sorted(fraud_seed.union(forta_df["address"].tolist()))

    common_file = PATHS["interim"] / "common_addresses.json"
    if args.include_common and common_file.exists():
        common = set(json.loads(common_file.read_text()))
        fraud_pool = sorted(set(fraud_pool) & common)
        log.info(
            "Restricted fraud-source fetch to %d addresses present in the common-address pool",
            len(fraud_pool),
        )
    if args.max_fraud is not None:
        fraud_pool = fraud_pool[: args.max_fraud]

    fraud_df = fetch_many_source_codes(
        fraud_pool,
        api_key=ETHERSCAN_API_KEY,
        progress_desc="Fetching fraud contracts",
    )
    fraud_out = PATHS["interim"] / "fraud_contracts.csv"
    fraud_df.to_csv(fraud_out, index=False)
    log.info("Saved %d fraud contracts to %s", len(fraud_df), fraud_out)


if __name__ == "__main__":
    main()
