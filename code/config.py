"""Single source of truth for paths, hyperparameters, and external endpoints."""
from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

PATHS = {
    "raw": DATA / "raw",
    "raw_forta": DATA / "raw" / "forta",
    "interim": DATA / "interim",
    "processed": DATA / "processed",
    "figures": DATA / "processed" / "figures",
    "checkpoints": DATA / "processed" / "checkpoints",
}

for p in PATHS.values():
    p.mkdir(parents=True, exist_ok=True)


ETHERSCAN_API_KEY = os.environ.get("ETHERSCAN_API_KEY", "")
ETHERSCAN_API_URL = "https://api.etherscan.io/v2/api"
ETHERSCAN_CHAIN_ID = 1  # Ethereum mainnet

FORTA_URLS = {
    "phishing_scams.csv": (
        "https://raw.githubusercontent.com/forta-network/labelled-datasets/"
        "main/labels/1/phishing_scams.csv"
    ),
    "malicious_smart_contracts.csv": (
        "https://raw.githubusercontent.com/forta-network/labelled-datasets/"
        "main/labels/1/malicious_smart_contracts.csv"
    ),
}

MODEL_CFG = {
    "gnn_in_dim": 3,
    "gnn_hidden_dim": 128,
    "gnn_out_dim": 64,
    "codebert_name": "microsoft/codebert-base",
    "codebert_dim": 768,
    "fusion_dim": 128,
    "num_classes": 2,
    "dropout": 0.5,
    "max_token_len": 512,
}

TRAIN_CFG = {
    "lr": 5e-5,
    "batch_size": 8,
    "epochs": 5,
    "grad_clip_norm": 1.0,
    "split": (0.70, 0.15, 0.15),
    "seed": 42,
    "num_workers": 0,
}

ETHERSCAN_RATE_LIMIT_SLEEP_S = 0.25
ETHERSCAN_REQUEST_TIMEOUT_S = 15
