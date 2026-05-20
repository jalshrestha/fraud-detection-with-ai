"""Step 2: download Forta labelled-datasets release into data/raw/forta/."""
from __future__ import annotations

import argparse
import logging

import _bootstrap  # noqa: F401

from src.collection.forta_labels import download_forta_labels, load_forta_labels

log = logging.getLogger("02_fetch_labels")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-download Forta CSVs even if they already exist on disk.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    paths = download_forta_labels(force=args.force)
    df = load_forta_labels()
    log.info("Forta files: %s", [str(p) for p in paths])
    log.info("Total deduplicated fraud labels: %d", len(df))


if __name__ == "__main__":
    main()
