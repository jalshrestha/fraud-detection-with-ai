"""Step 2e: mine exploit contract addresses from DeFiHackLabs PoC files.

DeFiHackLabs (github.com/SunWeb3Sec/DeFiHackLabs) has 729 Solidity PoC files
documenting real DeFi hacks. Each file contains the addresses of the actual
exploit/attacker contracts. This script:

  1. Fetches the full repo tree to get all 729 PoC .sol file paths.
  2. Downloads each file and extracts every 0x Ethereum address via regex.
  3. Filters out known-benign addresses (our benign seed list).
  4. Runs Etherscan getsourcecode on each unique address.
  5. Appends any with verified Solidity to fraud_contracts.csv.
"""
from __future__ import annotations

import base64
import logging
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

import _bootstrap  # noqa: F401

from config import (
    ETHERSCAN_API_URL,
    ETHERSCAN_RATE_LIMIT_SLEEP_S,
    ETHERSCAN_REQUEST_TIMEOUT_S,
    PATHS,
)

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

log = logging.getLogger("02e_fetch_defihacklabs")

GITHUB_TREE_URL = (
    "https://api.github.com/repos/SunWeb3Sec/DeFiHackLabs/git/trees/main?recursive=1"
)
GITHUB_RAW = "https://raw.githubusercontent.com/SunWeb3Sec/DeFiHackLabs/main/{path}"
ETH_ADDR_RE = re.compile(r'\b0x[a-fA-F0-9]{40}\b')


def get_poc_paths(session: requests.Session) -> list[str]:
    r = session.get(GITHUB_TREE_URL, timeout=30)
    r.raise_for_status()
    tree = r.json().get("tree", [])
    return [
        item["path"] for item in tree
        if item["path"].startswith("src/test/") and item["path"].endswith(".sol")
    ]


def extract_addresses_from_poc(path: str, session: requests.Session) -> set[str]:
    url = GITHUB_RAW.format(path=path)
    try:
        r = session.get(url, timeout=15)
        if r.status_code == 200:
            return {a.lower() for a in ETH_ADDR_RE.findall(r.text)}
    except requests.RequestException:
        pass
    return set()


def fetch_source_etherscan(address: str, api_key: str, session: requests.Session) -> str | None:
    params = {
        "module": "contract",
        "action": "getsourcecode",
        "address": address,
        "apikey": api_key,
    }
    try:
        r = session.get(ETHERSCAN_API_URL, params=params, timeout=ETHERSCAN_REQUEST_TIMEOUT_S)
        r.raise_for_status()
        payload = r.json()
        if payload.get("status") != "1":
            return None
        result = payload.get("result") or []
        source = (result[0].get("SourceCode") or "") if result else ""
        return source if source.strip() else None
    except requests.RequestException:
        return None


def load_existing(path: Path) -> set[str]:
    if path.exists():
        return set(pd.read_csv(path)["address"].str.lower())
    return set()


def append_rows(path: Path, rows: list[dict]) -> None:
    df = pd.DataFrame(rows, columns=["address", "source_code"])
    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)


def main() -> None:
    api_key = os.environ.get("ETHERSCAN_API_KEY", "")
    if not api_key:
        raise SystemExit("ETHERSCAN_API_KEY not set.")

    fraud_out = PATHS["interim"] / "fraud_contracts.csv"
    PATHS["interim"].mkdir(parents=True, exist_ok=True)

    done = load_existing(fraud_out)

    session = requests.Session()
    session.headers["User-Agent"] = "fraud-detection-research/1.0"

    # --- Step 1: get all PoC paths ---
    log.info("Fetching DeFiHackLabs repo tree ...")
    poc_paths = get_poc_paths(session)
    log.info("Found %d PoC .sol files", len(poc_paths))

    # --- Step 2: extract all addresses from PoC files ---
    log.info("Extracting addresses from PoC files ...")
    all_addresses: set[str] = set()
    for path in tqdm(poc_paths, desc="Parsing PoC files", unit="file"):
        all_addresses |= extract_addresses_from_poc(path, session)

    # filter out already-fetched
    candidates = sorted(all_addresses - done)
    log.info(
        "Unique addresses extracted: %d  |  after filtering benign+done: %d",
        len(all_addresses), len(candidates),
    )

    # --- Step 3: Etherscan source check ---
    log.info("Checking %d addresses on Etherscan ...", len(candidates))
    rows: list[dict] = []
    found = 0
    for addr in tqdm(candidates, desc="Etherscan", unit="addr"):
        src = fetch_source_etherscan(addr, api_key, session)
        if src:
            encoded = base64.b64encode(src.encode("utf-8")).decode("utf-8")
            rows.append({"address": addr, "source_code": encoded})
            found += 1
        time.sleep(ETHERSCAN_RATE_LIMIT_SLEEP_S)

    if rows:
        append_rows(fraud_out, rows)

    total = len(pd.read_csv(fraud_out)) if fraud_out.exists() else 0
    print(f"\nDone.")
    print(f"  DeFiHackLabs addresses checked : {len(candidates)}")
    print(f"  New fraud contracts found       : {found}")
    print(f"  fraud_contracts.csv total       : {total}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, stream=sys.stderr,
        format="%(asctime)s %(levelname)-8s %(name)s :: %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
