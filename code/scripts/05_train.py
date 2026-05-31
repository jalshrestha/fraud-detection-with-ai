"""Step 5: train the GNN-LLM fusion classifier on the processed dataset."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import MODEL_CFG, PATHS, TRAIN_CFG
from src.datasets.contract_dataset import ContractDataset, split_dataset
from src.models.fusion import FusionClassifier
from src.models.language_encoder import load_codebert
from src.training.trainer import load_checkpoint, save_checkpoint, train_model

log = logging.getLogger("05_train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=TRAIN_CFG["epochs"])
    p.add_argument("--batch-size", type=int, default=TRAIN_CFG["batch_size"])
    p.add_argument("--lr", type=float, default=TRAIN_CFG["lr"])
    p.add_argument("--weight-decay", type=float, default=TRAIN_CFG["weight_decay"])
    p.add_argument("--patience", type=int, default=TRAIN_CFG["early_stopping_patience"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Training device: %s", device)

    torch.manual_seed(TRAIN_CFG["seed"])

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
    train_ds, val_ds, _test_ds = split_dataset(
        dataset,
        graph_data.train_mask,
        graph_data.val_mask,
        graph_data.test_mask,
        addr2idx=addr2idx,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=TRAIN_CFG["num_workers"],
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
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
    graph_data = graph_data.to(device)

    train_labels = [dataset.labels[i] for i in train_ds.indices]
    log.info(
        "Training: %d samples (%d fraud, %d benign) | Val: %d samples",
        len(train_labels),
        sum(l == 1 for l in train_labels),
        sum(l == 0 for l in train_labels),
        len(val_ds),
    )

    best_ckpt = PATHS["checkpoints"] / "fusion_best.pt"
    losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        graph_data=graph_data,
        train_labels=train_labels,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip_norm=TRAIN_CFG["grad_clip_norm"],
        early_stopping_patience=args.patience,
        best_ckpt_path=best_ckpt,
    )

    # Also save final epoch checkpoint
    save_checkpoint(model, PATHS["checkpoints"] / "fusion_last.pt")

    # Load best checkpoint for evaluation
    load_checkpoint(model, best_ckpt, device=device)
    save_checkpoint(model, PATHS["checkpoints"] / "fusion.pt")

    losses_path = PATHS["processed"] / "loss_history.json"
    losses_path.write_text(json.dumps(losses, indent=2))
    log.info("Training complete. Best checkpoint: %s", best_ckpt)
    log.info("Loss history written to %s", losses_path)


if __name__ == "__main__":
    main()
