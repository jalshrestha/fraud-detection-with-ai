# Fraud Detection with AI: Hybrid GNN-LLM for Ethereum Smart Contracts

This repository contains the reference implementation and the paper for
*A Hybrid GNN-LLM Fusion Architecture for Multi-Modal Fraud Detection in
Ethereum Smart Contracts*.

The system pairs a two-layer GraphSAGE encoder over the Ethereum
transaction graph with a frozen CodeBERT encoder over verified Solidity
source code, fusing the two representations into a single fraud
classifier. The data pipeline starts from two seed arrays of contract
addresses, one benign and one fraudulent, and expands outward through the
transaction neighbourhood.

## Repository structure

```
.
├── code/      Reference implementation (data pipeline, models, training, evaluation).
└── paper/     LaTeX source for the paper (IEEEtran), figures, and bibliography.
```

See `code/README.md` for the pipeline and how to run it, and
`paper/README.md` for how to build the paper.

## Quick start

```
cd code
pip install -r requirements.txt
export ETHERSCAN_API_KEY=YOUR_KEY_HERE
# place a BigQuery transaction export at data/raw/transactions.csv
python scripts/01_fetch_transactions.py --restrict-to-seeds
python scripts/02_fetch_labels.py
python scripts/03_fetch_sources.py
python scripts/04_build_dataset.py
python scripts/05_train.py
python scripts/06_evaluate.py
python scripts/07_ablation.py
python scripts/08_visualize.py
```

## Pipeline at a glance

1. Two seed arrays (`code/data/addresses/`) anchor the collection: 210
   verified benign contracts and a curated fraud seed list.
2. The transaction neighbourhood of the seeds is pulled from the BigQuery
   public Ethereum dataset.
3. Forta labelled-datasets supply the canonical fraud ground truth.
4. Etherscan supplies verified Solidity source for contract addresses.
5. DuckDB joins everything into node and edge tables; PyTorch Geometric
   builds the graph.
6. The fusion classifier (GraphSAGE + frozen CodeBERT) is trained with
   class-weighted cross-entropy and evaluated against GNN-only and
   LLM-only ablations.
