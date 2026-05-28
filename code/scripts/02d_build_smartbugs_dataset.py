"""Step 2d: build fraud_contracts.csv and benign_contracts.csv from SmartBugs Wild.

Strategy (no wasted API calls):
  1. Load SmartBugs Wild addresses (data/raw/smartbugs_wild_addresses.txt).
  2. Load Forta fraud labels (already downloaded by 02_fetch_labels.py).
  3. Classify every SmartBugs address:
       - SmartBugs ∩ Forta  → FRAUD   (source pulled from GitHub raw)
       - SmartBugs - Forta  → BENIGN  (source pulled from GitHub raw, capped at --max-benign)
  4. Forta fraud addresses NOT in SmartBugs → FRAUD (source fetched via Etherscan API).
  5. Append to / create data/interim/fraud_contracts.csv and benign_contracts.csv.

Run after 02_fetch_labels.py.  Requires ETHERSCAN_API_KEY in the environment.
"""
from __future__ import annotations

import argparse
import base64
import logging
import os
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
from src.collection.forta_labels import load_forta_labels

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

log = logging.getLogger("02d_build_smartbugs_dataset")

SMARTBUGS_RAW = (
    "https://raw.githubusercontent.com/smartbugs/smartbugs-wild/master/contracts/{address}.sol"
)
ADDRESSES_FILE = PATHS["raw"] / "smartbugs_wild_addresses.txt"


# ---------------------------------------------------------------------------
# Source fetchers
# ---------------------------------------------------------------------------

def _encode(source: str) -> str:
    return base64.b64encode(source.encode("utf-8")).decode("utf-8")


def fetch_source_github(address: str, session: requests.Session) -> str | None:
    """Pull .sol source directly from SmartBugs Wild GitHub raw."""
    url = SMARTBUGS_RAW.format(address=address.lower())
    try:
        r = session.get(url, timeout=15)
        if r.status_code == 200 and r.text.strip():
            return r.text
    except requests.RequestException:
        pass
    return None


def fetch_source_etherscan(address: str, api_key: str, session: requests.Session) -> str | None:
    """Pull verified Solidity source from Etherscan."""
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-benign", type=int, default=2000,
                   help="Max benign contracts to include (default 2000).")
    p.add_argument("--max-etherscan-fraud", type=int, default=500,
                   help="Max Forta-only fraud addresses to query via Etherscan (default 500).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print stats only, do not fetch or write.")
    return p.parse_args()


def load_existing(path: Path) -> set[str]:
    if path.exists():
        df = pd.read_csv(path)
        return set(df["address"].str.lower())
    return set()


def append_rows(path: Path, rows: list[dict]) -> None:
    df_new = pd.DataFrame(rows, columns=["address", "source_code"])
    if path.exists():
        df_new.to_csv(path, mode="a", header=False, index=False)
    else:
        df_new.to_csv(path, index=False)


def main() -> None:
    args = parse_args()

    api_key = os.environ.get("ETHERSCAN_API_KEY", "")
    if not api_key and not args.dry_run:
        raise SystemExit("ETHERSCAN_API_KEY not set.")

    # --- load addresses ---
    if not ADDRESSES_FILE.exists():
        raise SystemExit(f"SmartBugs address file not found: {ADDRESSES_FILE}\n"
                         "Run the GitHub tree fetch step first.")
    smartbugs = set(ADDRESSES_FILE.read_text().splitlines())
    log.info("SmartBugs Wild addresses: %d", len(smartbugs))

    forta_df = load_forta_labels()
    forta_fraud = set(forta_df["address"].str.lower())
    log.info("Forta fraud addresses: %d", len(forta_fraud))

    # --- classify ---
    sb_fraud   = smartbugs & forta_fraud          # have source AND fraud label
    sb_benign  = smartbugs - forta_fraud           # have source, not fraud
    forta_only = forta_fraud - smartbugs           # fraud label, no SmartBugs source

    log.info("SmartBugs fraud (in both):      %d", len(sb_fraud))
    log.info("SmartBugs benign (not in Forta): %d  → capping at %d", len(sb_benign), args.max_benign)
    log.info("Forta-only fraud (Etherscan):   %d  → capping at %d", len(forta_only), args.max_etherscan_fraud)

    if args.dry_run:
        print("Dry run complete — nothing fetched.")
        return

    # --- skip already-fetched addresses ---
    fraud_out  = PATHS["interim"] / "fraud_contracts.csv"
    benign_out = PATHS["interim"] / "benign_contracts.csv"
    PATHS["interim"].mkdir(parents=True, exist_ok=True)

    done_fraud  = load_existing(fraud_out)
    done_benign = load_existing(benign_out)
    log.info("Already fetched — fraud: %d  benign: %d", len(done_fraud), len(done_benign))

    session = requests.Session()
    session.headers["User-Agent"] = "fraud-detection-research/1.0"

    # ── 1. SmartBugs fraud → GitHub raw ──────────────────────────────────
    to_fetch_fraud = [a for a in sorted(sb_fraud) if a not in done_fraud]
    log.info("Fetching %d SmartBugs fraud contracts from GitHub raw ...", len(to_fetch_fraud))
    rows: list[dict] = []
    for addr in tqdm(to_fetch_fraud, desc="SmartBugs fraud", unit="contract"):
        src = fetch_source_github(addr, session)
        if src:
            rows.append({"address": addr, "source_code": _encode(src)})
    if rows:
        append_rows(fraud_out, rows)
        log.info("Saved %d SmartBugs fraud contracts", len(rows))

    # ── 2. SmartBugs benign → GitHub raw (capped) ────────────────────────
    benign_candidates = sorted(sb_benign - done_benign)
    benign_needed = max(0, args.max_benign - len(done_benign))
    benign_candidates = benign_candidates[:benign_needed]
    log.info("Fetching %d SmartBugs benign contracts from GitHub raw ...", len(benign_candidates))
    rows = []
    for addr in tqdm(benign_candidates, desc="SmartBugs benign", unit="contract"):
        src = fetch_source_github(addr, session)
        if src:
            rows.append({"address": addr, "source_code": _encode(src)})
    if rows:
        append_rows(benign_out, rows)
        log.info("Saved %d SmartBugs benign contracts", len(rows))

    # ── 3. Forta-only fraud → Etherscan (capped) ─────────────────────────
    etherscan_candidates = [a for a in sorted(forta_only) if a not in done_fraud]
    etherscan_candidates = etherscan_candidates[:args.max_etherscan_fraud]
    log.info("Querying Etherscan for %d Forta-only fraud addresses ...", len(etherscan_candidates))
    rows = []
    for addr in tqdm(etherscan_candidates, desc="Etherscan fraud", unit="addr"):
        src = fetch_source_etherscan(addr, api_key, session)
        if src:
            rows.append({"address": addr, "source_code": _encode(src)})
        time.sleep(ETHERSCAN_RATE_LIMIT_SLEEP_S)
    if rows:
        append_rows(fraud_out, rows)
        log.info("Saved %d Etherscan fraud contracts", len(rows))

    # ── Summary ───────────────────────────────────────────────────────────
    fraud_total  = len(pd.read_csv(fraud_out))  if fraud_out.exists()  else 0
    benign_total = len(pd.read_csv(benign_out)) if benign_out.exists() else 0
    print(f"\nDone.")
    print(f"  fraud_contracts.csv  : {fraud_total} contracts")
    print(f"  benign_contracts.csv : {benign_total} contracts")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)-8s %(name)s :: %(message)s",
                        datefmt="%H:%M:%S")
    main()
