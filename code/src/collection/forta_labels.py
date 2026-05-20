"""Forta Network labelled-datasets downloader and loader."""
from __future__ import annotations

import glob
import logging
from pathlib import Path

import pandas as pd
import requests

from config import FORTA_URLS, PATHS

log = logging.getLogger(__name__)


def download_forta_labels(out_dir: Path | None = None, force: bool = False) -> list[Path]:
    """Download Forta phishing and malicious-contract CSVs to ``out_dir``.

    Returns the list of local file paths. If ``force`` is False, existing
    files are not re-downloaded.
    """
    out_dir = out_dir or PATHS["raw_forta"]
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for filename, url in FORTA_URLS.items():
        target = out_dir / filename
        if target.exists() and not force:
            log.info("Forta label file already present: %s", target)
            paths.append(target)
            continue
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        target.write_bytes(response.content)
        log.info("Downloaded Forta label file: %s (%d bytes)", target, len(response.content))
        paths.append(target)
    return paths


def load_forta_labels(forta_dir: Path | None = None) -> pd.DataFrame:
    """Load all Forta CSVs from ``forta_dir`` and return a deduped fraud table.

    Output columns: ``address`` (lowercased) and ``y`` (always 1).
    """
    forta_dir = forta_dir or PATHS["raw_forta"]
    frames: list[pd.DataFrame] = []
    for path in glob.glob(str(forta_dir / "*.csv")):
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            log.warning("Skipping unreadable Forta file %s: %s", path, exc)
            continue
        addr_col = next(
            (c for c in df.columns if "address" in c.lower()),
            df.columns[0],
        )
        df = df.rename(columns={addr_col: "address"})
        df["address"] = df["address"].astype(str).str.lower()
        frames.append(df[["address"]].copy())
    if not frames:
        return pd.DataFrame(columns=["address", "y"])
    merged = pd.concat(frames, ignore_index=True).dropna().drop_duplicates()
    merged["y"] = 1
    log.info("Loaded %d unique Forta-labelled fraud addresses", len(merged))
    return merged
