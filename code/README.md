# Hybrid GNN-LLM Fraud Detection: Code

Reference implementation for the paper *A Hybrid GNN-LLM Fusion
Architecture for Multi-Modal Fraud Detection in Ethereum Smart Contracts*.

The pipeline starts from two seed arrays of Ethereum contract addresses
(benign and fraudulent), expands them into a transaction neighbourhood
pulled from the BigQuery public Ethereum dataset, fetches verified
Solidity source code from Etherscan, builds a heterogeneous transaction
graph in DuckDB, and trains a GraphSAGE + CodeBERT fusion classifier.

## Repository layout

```
code/
  config.py                 Single source of truth for paths, hyperparameters, and endpoints.
  requirements.txt          Pinned dependencies.
  data/
    addresses/              Two seed arrays: benign_addresses.py, fraud_addresses.py.
    raw/                    Output: raw transaction CSV and Forta label CSVs.
    interim/                Output: filtered transactions, fetched source code.
    processed/              Output: nodes.parquet, edges.parquet, graph.pt, metrics.
  src/
    collection/             External-data fetchers (Etherscan, Forta, transactions).
    preprocessing/          Address normalisation, DuckDB joins, PyG graph builder.
    datasets/               ContractDataset: tokenised source paired with node indices.
    models/                 GraphEncoder (GraphSAGE), language encoder, FusionClassifier.
    training/               Class-weighted loss and training loop.
    evaluation/             Metrics and ablation models.
    visualization/          Dataset summary plots, subgraph and confusion-matrix plots.
  scripts/
    01_fetch_transactions.py
    02_fetch_labels.py
    03_fetch_sources.py
    04_build_dataset.py
    05_train.py
    06_evaluate.py
    07_ablation.py
    08_visualize.py
  notebooks/
    pipeline.ipynb          End-to-end notebook that calls the scripts in order.
```

## Setup

```
pip install -r requirements.txt
export ETHERSCAN_API_KEY=YOUR_KEY_HERE
```

Place a BigQuery export of Ethereum transactions at
`data/raw/transactions.csv`. The CSV must contain at least the columns
`from_address`, `to_address`, and `value`. The query used in the paper
is included in `data/raw/QUERY.sql`.

## Running the pipeline

```
python scripts/01_fetch_transactions.py --restrict-to-seeds
python scripts/02_fetch_labels.py
python scripts/03_fetch_sources.py
python scripts/04_build_dataset.py
python scripts/05_train.py
python scripts/06_evaluate.py
python scripts/07_ablation.py
python scripts/08_visualize.py
```

Each step writes its outputs under `data/` so that downstream steps can
be re-run independently after their upstream artefacts exist.

## Model summary

* GraphEncoder: two-layer GraphSAGE, 3 -> 128 -> 64 with ReLU and
  Dropout(0.5) between layers.
* LanguageEncoder: frozen `microsoft/codebert-base`, 768-dim pooler
  output.
* FusionClassifier: concatenate the 64-dim GNN embedding with the
  768-dim CodeBERT embedding, apply a 128-dim linear fusion layer
  with ReLU and Dropout(0.5), then a 2-class linear head.
* Loss: class-weighted cross-entropy with weight ratio `neg / pos`.
* Optimizer: AdamW, lr 5e-5, batch size 8, 5 epochs.
* Gradient clipping at norm 1.0.
