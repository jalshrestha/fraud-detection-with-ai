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
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
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
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    all_probs: list[float] = []
    for batch in tqdm(loader, desc="eval", unit="batch"):
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(graph_data, batch)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)
        all_probs.extend(probs.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(batch["label"].cpu().tolist())
    return summarize_metrics(all_labels, all_preds, all_probs)


def summarize_metrics(
    y_true: list[int],
    y_pred: list[int],
    y_prob: list[float] | None = None,
) -> dict[str, Any]:
    if not y_true:
        return {"y_true": [], "y_pred": [], "report": "", "confusion_matrix": [[0, 0], [0, 0]]}

    labels = [0, 1]
    target_names = ["Benign", "Fraud"]
    report_text = classification_report(
        y_true, y_pred, labels=labels, target_names=target_names, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0,
    )
    accuracy = sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)
    mcc = float(matthews_corrcoef(y_true, y_pred))

    result = {
        "y_true": y_true,
        "y_pred": y_pred,
        "report": report_text,
        "confusion_matrix": cm,
        "accuracy": float(accuracy),
        "mcc": mcc,
        # Fraud class (positive class) — primary metrics for the paper
        "precision_fraud": float(precision[1]),
        "recall_fraud": float(recall[1]),
        "f1_fraud": float(f1[1]),
        "support_fraud": int(support[1]),
        # Benign class
        "precision_benign": float(precision[0]),
        "recall_benign": float(recall[0]),
        "f1_benign": float(f1[0]),
        "support_benign": int(support[0]),
    }

    # AUC-ROC and AUC-PR require probability scores
    if y_prob is not None and len(set(y_true)) > 1:
        result["auc_roc"] = float(roc_auc_score(y_true, y_prob))
        result["auc_pr"] = float(average_precision_score(y_true, y_prob))
    else:
        result["auc_roc"] = None
        result["auc_pr"] = None

    return result
