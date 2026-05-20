"""Evaluation utilities returning standardised metric dictionaries."""
from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from tqdm import tqdm

log = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    graph_data: Data,
    device: torch.device | str,
) -> dict[str, Any]:
    """Run the trained model over ``loader`` and return prediction tensors plus metrics."""
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    for batch in tqdm(loader, desc="eval", unit="batch"):
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(graph_data, batch)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(batch["label"].cpu().tolist())
    return summarize_metrics(all_labels, all_preds)


def summarize_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    if not y_true:
        return {"y_true": [], "y_pred": [], "report": "", "confusion_matrix": [[0, 0], [0, 0]]}
    labels = [0, 1]
    target_names = ["Benign", "Fraud"]
    report_text = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "report": report_text,
        "confusion_matrix": cm,
        "precision_benign": float(precision[0]),
        "recall_benign": float(recall[0]),
        "f1_benign": float(f1[0]),
        "precision_fraud": float(precision[1]),
        "recall_fraud": float(recall[1]),
        "f1_fraud": float(f1[1]),
        "accuracy": float(sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)),
    }
