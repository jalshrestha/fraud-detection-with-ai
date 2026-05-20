"""Torch ``Dataset`` for tokenized Solidity source paired with graph node indices."""
from __future__ import annotations

import logging
from typing import Mapping

import pandas as pd
import torch
from torch.utils.data import Dataset, Subset

from src.collection.etherscan import decode_source_code

log = logging.getLogger(__name__)


class ContractDataset(Dataset):
    """Yields tokenized CodeBERT input plus the matching node index in the graph."""

    def __init__(
        self,
        nodes_df: pd.DataFrame,
        addr2idx: Mapping[str, int],
        tokenizer,
        max_len: int = 512,
        decode_base64: bool = True,
    ) -> None:
        if "address" not in nodes_df.columns:
            raise ValueError("nodes_df must contain an 'address' column")
        if "contract_text" not in nodes_df.columns:
            raise ValueError("nodes_df must contain a 'contract_text' column")
        if "is_labeled" not in nodes_df.columns:
            raise ValueError("nodes_df must contain an 'is_labeled' column")

        eligible = nodes_df[
            (nodes_df["contract_text"].astype(str).str.len() > 10)
            & nodes_df["is_labeled"].astype(bool)
            & nodes_df["address"].isin(addr2idx)
        ].reset_index(drop=True)

        self._contracts = eligible
        self._addr2idx = addr2idx
        self._tokenizer = tokenizer
        self._max_len = max_len
        self._decode_base64 = decode_base64

        log.info(
            "ContractDataset prepared with %d labelled contracts (eligible for training)",
            len(eligible),
        )

    def __len__(self) -> int:
        return len(self._contracts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self._contracts.iloc[idx]
        node_idx = self._addr2idx[row["address"]]
        raw_text = row["contract_text"]
        if self._decode_base64:
            raw_text = decode_source_code(raw_text)
        tokens = self._tokenizer(
            raw_text,
            truncation=True,
            max_length=self._max_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "node_idx": torch.tensor(node_idx, dtype=torch.long),
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "label": torch.tensor(int(row["y"]), dtype=torch.long),
        }

    @property
    def labels(self) -> list[int]:
        return self._contracts["y"].astype(int).tolist()


def split_dataset(
    dataset: ContractDataset,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    addr2idx: Mapping[str, int],
) -> tuple[Subset, Subset, Subset]:
    """Split ``ContractDataset`` rows by the graph-level train/val/test masks."""
    rows = dataset._contracts
    node_indices = rows["address"].map(addr2idx).to_numpy()
    train_idx, val_idx, test_idx = [], [], []
    for i, nidx in enumerate(node_indices):
        if train_mask[nidx]:
            train_idx.append(i)
        elif val_mask[nidx]:
            val_idx.append(i)
        elif test_mask[nidx]:
            test_idx.append(i)
    log.info(
        "Dataset split: train=%d, val=%d, test=%d",
        len(train_idx),
        len(val_idx),
        len(test_idx),
    )
    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)
