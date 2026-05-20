"""Frozen CodeBERT loader used as the LLM component of the fusion model."""
from __future__ import annotations

import logging
from typing import Tuple

import torch
from transformers import AutoModel, AutoTokenizer

log = logging.getLogger(__name__)


def load_codebert(
    name: str = "microsoft/codebert-base",
    device: torch.device | str | None = None,
) -> Tuple[AutoTokenizer, AutoModel]:
    """Load CodeBERT and its tokenizer with all parameters frozen.

    The model is moved to ``device`` if provided and switched to eval mode.
    The caller is responsible for keeping it in eval mode during training
    (the fusion classifier does this internally).
    """
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    if device is not None:
        model = model.to(device)
    log.info("Loaded frozen CodeBERT model: %s", name)
    return tokenizer, model
