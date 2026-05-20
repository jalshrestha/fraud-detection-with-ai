"""Seed list of fraudulent Ethereum contract addresses.

This is the second of the two seed arrays that drive the data-collection
pipeline. Seeds are used only to bootstrap transaction graph discovery
before the full Forta label set is available; they are NOT the training
labels.

At import time this module tries to load a sample from the Forta
labelled-datasets CSV that ``scripts/02_fetch_labels.py`` downloads.
If that file is not yet present on disk, it falls back to a minimal
hardcoded list of two well-documented, publicly post-mortemed exploit
contracts (The DAO hack and the Ronin Bridge exploit).

The canonical ground-truth for fraud labels is always the full Forta
dataset loaded by ``src.collection.forta_labels.load_forta_labels``.
"""
from __future__ import annotations

import csv
import logging
import random
from pathlib import Path

log = logging.getLogger(__name__)

_FALLBACK: list[str] = [
    # The DAO (2016 reentrancy exploit — the most cited case in the literature)
    "0xbb9bc244d798123fde783fcc1c72d3bb8c189413",
    # Ronin Bridge / Axie Infinity (2022 validator-key compromise)
    "0x098b716b8aaf21512996dc57eb0615e2383e2f96",
]

_SEED_SAMPLE_SIZE = 210
_FORTA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "forta"


def _load_from_forta(n: int = _SEED_SAMPLE_SIZE, seed: int = 42) -> list[str] | None:
    """Return up to *n* addresses sampled from the downloaded Forta CSVs.

    Returns None if no Forta CSV files are present yet.
    """
    csvs = list(_FORTA_DIR.glob("*.csv"))
    if not csvs:
        return None

    addresses: list[str] = []
    for path in csvs:
        try:
            with open(path, newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                addr_col = next(
                    (c for c in (reader.fieldnames or []) if "address" in c.lower()),
                    None,
                )
                if addr_col is None:
                    continue
                for row in reader:
                    addr = row[addr_col].strip().lower()
                    if addr.startswith("0x") and len(addr) == 42:
                        addresses.append(addr)
        except Exception as exc:
            log.warning("Could not read Forta file %s: %s", path, exc)

    if not addresses:
        return None

    rng = random.Random(seed)
    sample = rng.sample(addresses, min(n, len(addresses)))
    log.debug("Loaded %d fraud seed addresses from Forta CSVs", len(sample))
    return sample


def _resolve() -> list[str]:
    forta_seeds = _load_from_forta()
    if forta_seeds:
        return forta_seeds
    log.info(
        "Forta CSV not found — using %d hardcoded fallback seeds. "
        "Run scripts/02_fetch_labels.py to download the full Forta dataset.",
        len(_FALLBACK),
    )
    return list(_FALLBACK)


FRAUD_ADDRESSES: list[str] = _resolve()
