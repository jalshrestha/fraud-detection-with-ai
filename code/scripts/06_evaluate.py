"""Step 6: evaluate the trained fusion classifier on the held-out test split."""
from __future__ import annotations

import argparse
import json
import logging

import _bootstrap  # noqa: F401
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import MODEL_CFG, PATHS, TRAIN_CFG
from src.datasets.contract_dataset import ContractDataset, split_dataset
from src.evaluation.metrics import evaluate_model
from src.models.fusion import FusionClassifier
from src.models.language_encoder import load_codebert
from src.training.trainer import load_checkpoint
from src.visualization.graph_plots import plot_confusion_matrix

log = logging.getLogger("06_evaluate")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-size", type=int, default=TRAIN_CFG["batch_size"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nodes_df = pd.read_parquet(PATHS["processed"] / "nodes.parquet")
    graph_data = torch.load(PATHS["processed"] / "graph.pt", weights_only=False)
    addr2idx = json.loads((PATHS["processed"] / "addr2idx.json").read_text())

    tokenizer, codebert = load_codebert(MODEL_CFG["codebert_name"], device=device)
    dataset = ContractDataset(
        nodes_df=nodes_df,
        addr2idx=addr2idx,
        tokenizer=tokenizer,
        max_len=MODEL_CFG["max_token_len"],
    )
    _, _, test_ds = split_dataset(
        dataset,
        graph_data.train_mask,
        graph_data.val_mask,
        graph_data.test_mask,
        addr2idx=addr2idx,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=TRAIN_CFG["num_workers"],
    )

    model = FusionClassifier(
        graph_feat_dim=graph_data.x.size(1),
        codebert_model=codebert,
        gnn_hidden_dim=MODEL_CFG["gnn_hidden_dim"],
        gnn_out_dim=MODEL_CFG["gnn_out_dim"],
        fusion_dim=MODEL_CFG["fusion_dim"],
        num_classes=MODEL_CFG["num_classes"],
        dropout=MODEL_CFG["dropout"],
    ).to(device)
    load_checkpoint(model, PATHS["checkpoints"] / "fusion.pt", device=device)
    graph_data = graph_data.to(device)

    metrics = evaluate_model(model, test_loader, graph_data, device=device)
    print(metrics["report"])

    metrics_out = PATHS["processed"] / "metrics.json"
    metrics_payload = {k: v for k, v in metrics.items() if k not in {"y_true", "y_pred"}}
    metrics_out.write_text(json.dumps(metrics_payload, indent=2))
    log.info("Metrics written to %s", metrics_out)

    cm_path = PATHS["figures"] / "confusion_matrix.png"
    plot_confusion_matrix(metrics["confusion_matrix"], savepath=cm_path)
    log.info("Confusion matrix saved to %s", cm_path)


if __name__ == "__main__":
    main()
