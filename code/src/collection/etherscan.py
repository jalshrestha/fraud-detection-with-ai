"""Etherscan source-code fetcher with rate limiting and Base64 encoding."""
from __future__ import annotations

import base64
import logging
import time
from typing import Iterable, Optional

import pandas as pd
import requests
from tqdm import tqdm

from config import (
    ETHERSCAN_API_URL,
    ETHERSCAN_RATE_LIMIT_SLEEP_S,
    ETHERSCAN_REQUEST_TIMEOUT_S,
)

log = logging.getLogger(__name__)


def fetch_source_code(address: str, api_key: str) -> Optional[str]:
    """Return raw Solidity source for an address, or None if not verified.

    Returns None on any HTTP error, rate-limit response, or empty payload.
    The caller is expected to handle the None case rather than retry here,
    because the Etherscan API distinguishes "not verified" from "rate-limited"
    in a stable way that does not benefit from automatic retries.
    """
    params = {
        "module": "contract",
        "action": "getsourcecode",
        "address": address,
        "apikey": api_key,
    }
    try:
        response = requests.get(
            ETHERSCAN_API_URL,
            params=params,
            timeout=ETHERSCAN_REQUEST_TIMEOUT_S,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        log.debug("Etherscan request failed for %s: %s", address, exc)
        return None

    payload = response.json()
    if payload.get("status") != "1":
        return None
    result = payload.get("result") or []
    if not result:
        return None
    source_code = result[0].get("SourceCode", "") or ""
    return source_code if source_code else None


def fetch_many_source_codes(
    addresses: Iterable[str],
    api_key: str,
    sleep_s: float = ETHERSCAN_RATE_LIMIT_SLEEP_S,
    progress_desc: str = "Fetching contracts",
) -> pd.DataFrame:
    """Fetch source code for many addresses, Base64-encode it, and return a DataFrame.

    Columns of the returned DataFrame: ``address`` (lowercased) and
    ``source_code`` (Base64-encoded UTF-8 string of the raw Solidity source).
    Addresses without verified source are dropped from the output.
    """
    rows: list[dict[str, str]] = []
    addr_list = list(addresses)
    for address in tqdm(addr_list, desc=progress_desc, unit="addr"):
        raw_source = fetch_source_code(address, api_key)
        if raw_source:
            encoded = base64.b64encode(raw_source.encode("utf-8")).decode("utf-8")
            rows.append({"address": address.lower(), "source_code": encoded})
        time.sleep(sleep_s)
    df = pd.DataFrame(rows, columns=["address", "source_code"])
    log.info("Fetched verified source for %d / %d addresses", len(df), len(addr_list))
    return df


def decode_source_code(encoded: str) -> str:
    """Decode a Base64-encoded source-code string back to raw Solidity."""
    if not encoded:
        return ""
    return base64.b64decode(encoded.encode("utf-8")).decode("utf-8", errors="replace")
