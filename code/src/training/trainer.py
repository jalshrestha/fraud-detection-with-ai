"""Training loop and checkpoint helpers for the fusion classifier."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from tqdm import tqdm

from .losses import build_criterion, compute_class_weights

log = logging.getLogger(__name__)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    graph_data: Data,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device | str,
    grad_clip_norm: float = 1.0,
) -> float:
    """Run a single training epoch and return the average loss."""
    model.train()
    running_loss = 0.0
    n_batches = 0
    nan_skipped = 0
    for batch in tqdm(loader, desc="train", unit="batch"):
        optimizer.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(graph_data, batch)
        loss = criterion(logits, batch["label"])
        if torch.isnan(loss):
            nan_skipped += 1
            continue
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        running_loss += loss.item()
        n_batches += 1
    if nan_skipped:
        log.warning("Skipped %d batches due to NaN loss", nan_skipped)
    return running_loss / max(n_batches, 1)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    graph_data: Data,
    train_labels: Iterable[int],
    device: torch.device | str,
    epochs: int = 5,
    lr: float = 5e-5,
    grad_clip_norm: float = 1.0,
) -> list[float]:
    """Train the fusion model and return the per-epoch average losses."""
    class_weights = compute_class_weights(list(train_labels), device=device)
    criterion = build_criterion(class_weights)
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
    )
    losses: list[float] = []
    for epoch in range(1, epochs + 1):
        avg_loss = train_one_epoch(
            model,
            train_loader,
            graph_data,
            optimizer,
            criterion,
            device=device,
            grad_clip_norm=grad_clip_norm,
        )
        losses.append(avg_loss)
        log.info("Epoch %d/%d completed - avg loss: %.4f", epoch, epochs, avg_loss)
    return losses


def save_checkpoint(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    log.info("Saved checkpoint to %s", path)


def load_checkpoint(model: nn.Module, path: Path, device: torch.device | str) -> nn.Module:
    state = torch.load(path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device)
    log.info("Loaded checkpoint from %s", path)
    return model
