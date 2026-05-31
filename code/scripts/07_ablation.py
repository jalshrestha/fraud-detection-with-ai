"""Step 7: comprehensive ablation — GNN variants, LLM-only, attention fusion, RF baseline."""
from __future__ import annotations

import json
import logging

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score,
    f1_score, matthews_corrcoef, roc_auc_score,
)
from torch.utils.data import DataLoader

from config import MODEL_CFG, PATHS, TRAIN_CFG
from src.datasets.contract_dataset import ContractDataset, split_dataset
from src.evaluation.ablation import (
    build_attention_fusion_model,
    build_gat_only_model,
    build_gcn_only_model,
    build_gnn_only_model,
    build_llm_only_model,
)
from src.evaluation.metrics import evaluate_model
from src.models.language_encoder import load_codebert
from src.training.trainer import train_model

log = logging.getLogger("07_ablation")


def _quick_metrics(y_true, y_pred, y_prob=None):
    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_fraud": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
    if y_prob is not None and len(set(y_true)) > 1:
        m["auc_roc"] = float(roc_auc_score(y_true, y_prob))
        m["auc_pr"]  = float(average_precision_score(y_true, y_prob))
    return m


def _train_and_eval(model, name, train_loader, val_loader, test_loader,
                    graph_data, train_labels, device):
    log.info("Training %s ...", name)
    train_model(
        model=model, train_loader=train_loader, val_loader=val_loader,
        graph_data=graph_data, train_labels=train_labels, device=device,
        epochs=TRAIN_CFG["epochs"], lr=TRAIN_CFG["lr"],
        weight_decay=TRAIN_CFG["weight_decay"],
        grad_clip_norm=TRAIN_CFG["grad_clip_norm"],
        early_stopping_patience=TRAIN_CFG["early_stopping_patience"],
        best_ckpt_path=PATHS["checkpoints"] / f"{name}_best.pt",
    )
    return evaluate_model(model, test_loader, graph_data, device=device)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(TRAIN_CFG["seed"])

    nodes_df   = pd.read_parquet(PATHS["processed"] / "nodes.parquet")
    graph_data = torch.load(PATHS["processed"] / "graph.pt", weights_only=False)
    addr2idx   = json.loads((PATHS["processed"] / "addr2idx.json").read_text())

    tokenizer, codebert = load_codebert(MODEL_CFG["codebert_name"], device=device)
    dataset = ContractDataset(nodes_df=nodes_df, addr2idx=addr2idx,
                              tokenizer=tokenizer, max_len=MODEL_CFG["max_token_len"])
    train_ds, val_ds, test_ds = split_dataset(
        dataset, graph_data.train_mask, graph_data.val_mask,
        graph_data.test_mask, addr2idx=addr2idx)

    BS = TRAIN_CFG["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BS, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BS, shuffle=False)
    train_labels = [dataset.labels[i] for i in train_ds.indices]
    graph_data   = graph_data.to(device)
    in_dim = graph_data.x.size(1)

    kw = dict(hidden_dim=MODEL_CFG["gnn_hidden_dim"], out_dim=MODEL_CFG["gnn_out_dim"],
               num_classes=MODEL_CFG["num_classes"], dropout=MODEL_CFG["dropout"])

    results = {}

    # ── 1. GraphSAGE-only ──────────────────────────────────────────────────
    results["GraphSAGE-only"] = _train_and_eval(
        build_gnn_only_model(in_dim, **kw), "sage_only",
        train_loader, val_loader, test_loader, graph_data, train_labels, device)

    # ── 2. GCN-only ────────────────────────────────────────────────────────
    results["GCN-only"] = _train_and_eval(
        build_gcn_only_model(in_dim, **kw), "gcn_only",
        train_loader, val_loader, test_loader, graph_data, train_labels, device)

    # ── 3. GAT-only ────────────────────────────────────────────────────────
    results["GAT-only"] = _train_and_eval(
        build_gat_only_model(in_dim, **kw), "gat_only",
        train_loader, val_loader, test_loader, graph_data, train_labels, device)

    # ── 4. CodeBERT-only ───────────────────────────────────────────────────
    results["CodeBERT-only"] = _train_and_eval(
        build_llm_only_model(codebert, hidden_dim=MODEL_CFG["codebert_dim"],
                             num_classes=MODEL_CFG["num_classes"],
                             dropout=MODEL_CFG["dropout"]),
        "llm_only", train_loader, val_loader, test_loader,
        graph_data, train_labels, device)

    # ── 5. Attention-based fusion ──────────────────────────────────────────
    results["Attention-Fusion"] = _train_and_eval(
        build_attention_fusion_model(
            in_dim, codebert,
            gnn_hidden_dim=MODEL_CFG["gnn_hidden_dim"],
            gnn_out_dim=MODEL_CFG["gnn_out_dim"],
            fusion_dim=MODEL_CFG["fusion_dim"],
            num_classes=MODEL_CFG["num_classes"],
            dropout=MODEL_CFG["dropout"]),
        "attn_fusion", train_loader, val_loader, test_loader,
        graph_data, train_labels, device)

    # ── 6. Random Forest on raw node features ─────────────────────────────
    log.info("Training Random Forest baseline ...")
    X_all = graph_data.x.cpu().numpy()
    y_all = graph_data.y.cpu().numpy()

    train_idx = graph_data.train_mask.nonzero(as_tuple=False).flatten().numpy()
    test_idx  = graph_data.test_mask.nonzero(as_tuple=False).flatten().numpy()
    X_tr, y_tr = X_all[train_idx], y_all[train_idx]
    X_te, y_te = X_all[test_idx],  y_all[test_idx]

    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    rf_pred = rf.predict(X_te)
    rf_prob = rf.predict_proba(X_te)[:, 1]
    results["Random-Forest"] = _quick_metrics(y_te.tolist(), rf_pred.tolist(), rf_prob.tolist())

    # ── 7. Load saved fusion (concat) result ──────────────────────────────
    fusion_m = json.loads((PATHS["processed"] / "metrics.json").read_text())
    results["Concat-Fusion (ours)"] = {
        k: v for k, v in fusion_m.items() if k not in {"y_true", "y_pred", "report"}
    }

    # Save
    out = {k: {kk: vv for kk, vv in v.items() if kk not in {"y_true", "y_pred", "report"}}
           for k, v in results.items()}
    (PATHS["processed"] / "ablation.json").write_text(json.dumps(out, indent=2))

    # Print comparison table
    cols = ["accuracy", "f1_fraud", "auc_roc", "auc_pr", "mcc"]
    print("\n" + "=" * 78)
    print(f"{'COMPREHENSIVE ABLATION STUDY — TEST SET':^78}")
    print("=" * 78)
    print(f"{'Model':<26} {'Acc':>6} {'F1-Fraud':>9} {'AUC-ROC':>8} {'AUC-PR':>8} {'MCC':>7}")
    print("-" * 78)
    for name, m in results.items():
        acc = m.get("accuracy", 0)
        f1  = m.get("f1_fraud", 0)
        auc = m.get("auc_roc") or 0
        apr = m.get("auc_pr")  or 0
        mcc = m.get("mcc", 0)
        marker = " ◀" if "Fusion" in name and "Concat" in name else ""
        print(f"{name:<26} {acc:>6.3f} {f1:>9.3f} {auc:>8.3f} {apr:>8.3f} {mcc:>7.3f}{marker}")
    print("=" * 78)
    log.info("Ablation results written to %s", PATHS["processed"] / "ablation.json")


if __name__ == "__main__":
    main()
