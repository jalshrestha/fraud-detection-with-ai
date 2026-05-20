"""Address-normalization helpers used everywhere we touch raw data."""
from __future__ import annotations

from typing import Iterable

import pandas as pd


def standardize_address_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Lowercase and strip the given address columns in place and return the frame."""
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
    return df


def union_addresses(*lists: Iterable[str]) -> list[str]:
    """Return a sorted, deduplicated union of the given iterables, lowercased."""
    seen: set[str] = set()
    for items in lists:
        for value in items:
            if value:
                seen.add(value.lower())
    return sorted(seen)
