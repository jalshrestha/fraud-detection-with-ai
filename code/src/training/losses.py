"""Class-weighted cross-entropy loss construction."""
from __future__ import annotations

import logging
from typing import Sequence

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


def compute_class_weights(
    labels: Sequence[int],
    device: torch.device | str,
    max_weight: float = 50.0,
) -> torch.Tensor:
    """Return a 2-element tensor ``[1.0, neg/pos]`` clamped at ``max_weight``."""
    neg = sum(1 for lab in labels if lab == 0)
    pos = sum(1 for lab in labels if lab == 1)
    if pos == 0 or neg == 0:
        log.warning(
            "Only one class present in training labels (neg=%d, pos=%d); using equal weights",
            neg,
            pos,
        )
        return torch.tensor([1.0, 1.0], device=device)
    weight_pos = min(neg / pos, max_weight)
    log.info(
        "Computed class weights: benign=1.0, fraud=%.2f (neg=%d, pos=%d)",
        weight_pos,
        neg,
        pos,
    )
    return torch.tensor([1.0, weight_pos], device=device)


def build_criterion(class_weights: torch.Tensor) -> nn.CrossEntropyLoss:
    return nn.CrossEntropyLoss(weight=class_weights)
