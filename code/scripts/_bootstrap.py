"""Shared bootstrap for all top-level scripts.

Each script imports this first to ensure the repository root is on sys.path
so that ``src.*`` and ``config`` imports resolve consistently regardless of
the current working directory.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
    datefmt="%H:%M:%S",
)
