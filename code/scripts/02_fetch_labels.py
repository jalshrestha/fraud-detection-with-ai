"""Step 2: build the labelled fraud/benign contract dataset.

Strategy (updated after empirical testing):
  - Forta addresses yield 0% verified Solidity source on Etherscan because
    they are attacker wallets and unverified exploit contracts.
  - The correct fraud sources are DeFi exploit *victim* contracts extracted
    from DeFiHackLabs PoC files: these are deployed protocols that have
    verified source on Etherscan AND have on-chain transaction history.
  - Running this script produces data/interim/fraud_contracts.csv and
    data/interim/benign_contracts.csv, which feed into step 04.

Requires: ETHERSCAN_API_KEY in environment (uses Etherscan API v2).
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import re
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import requests

from config import (
    ETHERSCAN_API_KEY,
    ETHERSCAN_API_URL,
    ETHERSCAN_CHAIN_ID,
    ETHERSCAN_RATE_LIMIT_SLEEP_S,
    PATHS,
)
from data.addresses import BENIGN_ADDRESSES

log = logging.getLogger("02_fetch_labels")

DEFIHACKLABS_DIR = Path(__file__).resolve().parents[1] / "DeFiHackLabs" / "src" / "test"


def extract_eth_mainnet_addresses(sol_dir: Path) -> list[dict]:
    """Parse DeFiHackLabs .sol files for Ethereum mainnet victim contract addresses."""
    results: list[dict] = []
    for sol_file in sorted(sol_dir.rglob("*.sol")):
        content = sol_file.read_text(errors="ignore")
        year_match = re.search(r"(20[12][0-9])", str(sol_file))
        year = int(year_match.group(1)) if year_match else 0
        for line in content.split("\n"):
            if "Vulnerable Contract" in line and "etherscan.io" in line:
                addr = re.search(r"0x[a-fA-F0-9]{40}", line)
                if addr:
                    results.append({"address": addr.group(0).lower(), "year": year, "file": sol_file.name})
    seen: set[str] = set()
    unique = []
    for r in results:
        if r["address"] not in seen:
            seen.add(r["address"])
            unique.append(r)
    return unique


def fetch_source(address: str, api_key: str) -> tuple[str, str]:
    """Return (source_code_b64, contract_name) or ('', '') if not verified."""
    params = {
        "chainid": ETHERSCAN_CHAIN_ID,
        "module": "contract",
        "action": "getsourcecode",
        "address": address,
        "apikey": api_key,
    }
    try:
        resp = requests.get(ETHERSCAN_API_URL, params=params, timeout=12).json()
        result = resp.get("result") or []
        if isinstance(result, list) and result:
            src = result[0].get("SourceCode", "") or ""
            cname = result[0].get("ContractName", "")
            if src and len(src) > 50:
                return base64.b64encode(src.encode()).decode(), cname
    except Exception as exc:
        log.debug("Etherscan error for %s: %s", address, exc)
    return "", ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", action="store_true", help="Re-fetch even if output files exist.")
    p.add_argument("--max-fraud", type=int, default=None, help="Cap fraud contracts fetched.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not ETHERSCAN_API_KEY:
        raise SystemExit("ETHERSCAN_API_KEY is not set.")

    fraud_out = PATHS["interim"] / "fraud_contracts.csv"
    benign_out = PATHS["interim"] / "benign_contracts.csv"

    # --- Fraud contracts from DeFiHackLabs ---
    if not fraud_out.exists() or args.force:
        if not DEFIHACKLABS_DIR.exists():
            raise SystemExit(
                f"DeFiHackLabs not found at {DEFIHACKLABS_DIR}. "
                "Run: git clone --depth=1 https://github.com/SunWeb3Sec/DeFiHackLabs.git"
            )
        candidates = extract_eth_mainnet_addresses(DEFIHACKLABS_DIR)
        if args.max_fraud:
            candidates = candidates[: args.max_fraud]
        log.info("Found %d Ethereum mainnet victim addresses in DeFiHackLabs", len(candidates))

        import pandas as pd

        rows = []
        for i, r in enumerate(candidates):
            src_b64, cname = fetch_source(r["address"], ETHERSCAN_API_KEY)
            if src_b64:
                rows.append(
                    {
                        "address": r["address"],
                        "contract_name": cname,
                        "year": r["year"],
                        "label": 1,
                        "source_code": src_b64,
                        "source": "defihacklabs",
                    }
                )
                log.info("[%d/%d] ✓ %s  %s  %d", i + 1, len(candidates), cname, r["address"], r["year"])
            time.sleep(ETHERSCAN_RATE_LIMIT_SLEEP_S)

        pd.DataFrame(rows).to_csv(fraud_out, index=False)
        log.info("Saved %d fraud contracts → %s", len(rows), fraud_out)
    else:
        log.info("Fraud contracts already exist at %s (use --force to re-fetch)", fraud_out)

    # --- Benign contracts from seed list ---
    if not benign_out.exists() or args.force:
        import pandas as pd

        rows = []
        for i, addr in enumerate(BENIGN_ADDRESSES):
            src_b64, cname = fetch_source(addr, ETHERSCAN_API_KEY)
            if src_b64:
                rows.append(
                    {
                        "address": addr.lower(),
                        "contract_name": cname,
                        "label": 0,
                        "source_code": src_b64,
                        "source": "benign_seeds",
                    }
                )
                log.info("[%d/%d] ✓ %s  %s", i + 1, len(BENIGN_ADDRESSES), cname, addr)
            time.sleep(ETHERSCAN_RATE_LIMIT_SLEEP_S)

        pd.DataFrame(rows).to_csv(benign_out, index=False)
        log.info("Saved %d benign contracts → %s", len(rows), benign_out)
    else:
        log.info("Benign contracts already exist at %s (use --force to re-fetch)", benign_out)


if __name__ == "__main__":
    main()
