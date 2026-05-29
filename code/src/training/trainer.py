"""Training loop and checkpoint helpers for the fusion classifier."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
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
    model.train()
    running_loss = 0.0
    n_batches = 0
    nan_skipped = 0
    for batch in loader:
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


@torch.no_grad()
def _val_f1_fraud(
    model: nn.Module,
    loader: DataLoader,
    graph_data: Data,
    device: torch.device | str,
) -> float:
    """Return F1 score for the fraud class on the validation split."""
    model.eval()
    y_true, y_pred = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(graph_data, batch)
        preds = torch.argmax(logits, dim=1)
        y_true.extend(batch["label"].cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return 2 * precision * recall / max(precision + recall, 1e-8)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    graph_data: Data,
    train_labels: Iterable[int],
    device: torch.device | str,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    grad_clip_norm: float = 1.0,
    early_stopping_patience: int = 20,
    best_ckpt_path: Path | None = None,
) -> list[float]:
    class_weights = compute_class_weights(list(train_labels), device=device)
    criterion = build_criterion(class_weights)
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    best_val_f1 = -1.0
    patience_counter = 0
    losses: list[float] = []

    for epoch in range(1, epochs + 1):
        avg_loss = train_one_epoch(
            model, train_loader, graph_data, optimizer, criterion,
            device=device, grad_clip_norm=grad_clip_norm,
        )
        scheduler.step()
        val_f1 = _val_f1_fraud(model, val_loader, graph_data, device)
        losses.append(avg_loss)

        log.info(
            "Epoch %3d/%d | loss=%.4f | val_F1_fraud=%.4f | lr=%.2e",
            epoch, epochs, avg_loss, val_f1,
            scheduler.get_last_lr()[0],
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            if best_ckpt_path is not None:
                save_checkpoint(model, best_ckpt_path)
                log.info("  ↑ New best val F1=%.4f — checkpoint saved", best_val_f1)
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                log.info(
                    "Early stopping at epoch %d (patience=%d, best val F1=%.4f)",
                    epoch, early_stopping_patience, best_val_f1,
                )
                break

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
