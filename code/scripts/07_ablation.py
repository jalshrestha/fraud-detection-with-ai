"""Step 7: run the GNN-only and LLM-only ablation models."""
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
from src.evaluation.ablation import build_gnn_only_model, build_llm_only_model
from src.evaluation.metrics import evaluate_model
from src.models.language_encoder import load_codebert
from src.training.trainer import train_model

log = logging.getLogger("07_ablation")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=TRAIN_CFG["epochs"])
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
    train_ds, _val_ds, test_ds = split_dataset(
        dataset,
        graph_data.train_mask,
        graph_data.val_mask,
        graph_data.test_mask,
        addr2idx=addr2idx,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    train_labels = [dataset.labels[i] for i in train_ds.indices]
    graph_data = graph_data.to(device)

    log.info("Training GNN-only ablation model")
    gnn_model = build_gnn_only_model(
        in_dim=graph_data.x.size(1),
        hidden_dim=MODEL_CFG["gnn_hidden_dim"],
        out_dim=MODEL_CFG["gnn_out_dim"],
        num_classes=MODEL_CFG["num_classes"],
        dropout=MODEL_CFG["dropout"],
    ).to(device)
    train_model(
        gnn_model,
        train_loader,
        graph_data,
        train_labels=train_labels,
        device=device,
        epochs=args.epochs,
        lr=TRAIN_CFG["lr"],
        grad_clip_norm=TRAIN_CFG["grad_clip_norm"],
    )
    gnn_metrics = evaluate_model(gnn_model, test_loader, graph_data, device=device)

    log.info("Training LLM-only ablation model")
    llm_model = build_llm_only_model(
        codebert_model=codebert,
        hidden_dim=MODEL_CFG["codebert_dim"],
        num_classes=MODEL_CFG["num_classes"],
        dropout=MODEL_CFG["dropout"],
    ).to(device)
    train_model(
        llm_model,
        train_loader,
        graph_data,
        train_labels=train_labels,
        device=device,
        epochs=args.epochs,
        lr=TRAIN_CFG["lr"],
        grad_clip_norm=TRAIN_CFG["grad_clip_norm"],
    )
    llm_metrics = evaluate_model(llm_model, test_loader, graph_data, device=device)

    ablation = {
        "gnn_only": {k: v for k, v in gnn_metrics.items() if k not in {"y_true", "y_pred"}},
        "llm_only": {k: v for k, v in llm_metrics.items() if k not in {"y_true", "y_pred"}},
    }
    out_path = PATHS["processed"] / "ablation.json"
    out_path.write_text(json.dumps(ablation, indent=2))
    log.info("Ablation results written to %s", out_path)
    print(gnn_metrics["report"])
    print(llm_metrics["report"])


if __name__ == "__main__":
    main()
