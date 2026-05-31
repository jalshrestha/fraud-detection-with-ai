"""Step 2b: expand the benign address seed list from the Uniswap token list.

Fetches https://tokens.uniswap.org (the official Uniswap Labs token list),
filters for Ethereum mainnet (chainId 1), deduplicates against the existing
BENIGN_ADDRESSES seed, and rewrites data/addresses/benign_addresses.py with
the merged result.

Run this script whenever you want to refresh or grow the benign seed list.
No API key required — the Uniswap token list is publicly accessible.
"""
from __future__ import annotations

import argparse
import logging
import sys
import textwrap
from pathlib import Path

import requests

import _bootstrap  # noqa: F401

from data.addresses import BENIGN_ADDRESSES

UNISWAP_TOKEN_LIST_URL = "https://tokens.uniswap.org"
ETHEREUM_CHAIN_ID = 1

log = logging.getLogger("02b_fetch_benign_addresses")


def fetch_uniswap_addresses() -> list[str]:
    """Return lowercased Ethereum mainnet addresses from the Uniswap token list."""
    log.info("Fetching Uniswap token list from %s", UNISWAP_TOKEN_LIST_URL)
    resp = requests.get(UNISWAP_TOKEN_LIST_URL, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    tokens = data.get("tokens", [])
    addresses = [
        t["address"].lower()
        for t in tokens
        if t.get("chainId") == ETHEREUM_CHAIN_ID and t.get("address")
    ]
    log.info("Found %d Ethereum tokens in Uniswap list", len(addresses))
    return addresses


def write_benign_addresses(addresses: list[str], out_path: Path) -> None:
    """Rewrite benign_addresses.py with the merged address list."""
    lines = ',\n    '.join(f"'{a}'" for a in addresses)
    content = textwrap.dedent(f'''\
        """Curated list of verified benign Ethereum contract addresses.

        This is the first of the two seed arrays that drive the data-collection
        pipeline. Each address is a publicly verified ERC-20 token, exchange,
        or blue-chip protocol contract on Ethereum mainnet. Addresses are stored
        lowercased to match Forta and Etherscan conventions.

        Sources:
          - CoinGecko top-N tokens by market cap, cross-checked against
            Etherscan\'s verified-contracts registry.
          - Uniswap Labs official token list (https://tokens.uniswap.org),
            filtered for Ethereum mainnet (chainId=1).
        """
        from __future__ import annotations

        BENIGN_ADDRESSES: list[str] = [
            {lines},
        ]
    ''')
    out_path.write_text(content)
    log.info("Wrote %d addresses to %s", len(addresses), out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stats without writing the file.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    uniswap_addrs = fetch_uniswap_addresses()

    existing = set(a.lower() for a in BENIGN_ADDRESSES)
    new_addrs = [a for a in uniswap_addrs if a not in existing]
    log.info(
        "New addresses not already in seed list: %d (existing: %d)",
        len(new_addrs),
        len(existing),
    )

    merged = list(existing) + new_addrs
    merged_deduped = sorted(set(merged))

    log.info("Merged benign address count: %d", len(merged_deduped))

    if args.dry_run:
        log.info("Dry run — skipping file write.")
        print(f"Would write {len(merged_deduped)} addresses ({len(new_addrs)} new).")
        return

    out_path = Path(__file__).resolve().parent.parent / "data" / "addresses" / "benign_addresses.py"
    write_benign_addresses(merged_deduped, out_path)
    print(f"Done. {len(merged_deduped)} total benign addresses ({len(new_addrs)} added from Uniswap).")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    main()
