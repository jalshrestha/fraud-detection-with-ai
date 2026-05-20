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
from src.training.trainer import save_checkpoint, train_model

log = logging.getLogger("05_train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=TRAIN_CFG["epochs"])
    p.add_argument("--batch-size", type=int, default=TRAIN_CFG["batch_size"])
    p.add_argument("--lr", type=float, default=TRAIN_CFG["lr"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Training device: %s", device)

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
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
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
    losses = train_model(
        model=model,
        train_loader=train_loader,
        graph_data=graph_data,
        train_labels=train_labels,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        grad_clip_norm=TRAIN_CFG["grad_clip_norm"],
    )

    ckpt_path = PATHS["checkpoints"] / "fusion.pt"
    save_checkpoint(model, ckpt_path)

    losses_path = PATHS["processed"] / "loss_history.json"
    losses_path.write_text(json.dumps(losses, indent=2))
    log.info("Loss history written to %s", losses_path)


if __name__ == "__main__":
    main()
