"""Dataset-summary plots: transaction values, account types, label balance."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .style import apply_style


def plot_transaction_values(edges_df: pd.DataFrame, savepath: Path) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    eth = edges_df["value"].astype(float) / 1e18
    eth = eth[eth > 0]
    sns.histplot(eth, bins=50, log_scale=True, color="navy", ax=ax)
    ax.set_title("Distribution of Transaction Values")
    ax.set_xlabel("Transaction Value in ETH (log scale)")
    ax.set_ylabel("Number of Transactions")
    fig.tight_layout()
    fig.savefig(savepath)
    plt.close(fig)


def plot_account_types(nodes_df: pd.DataFrame, savepath: Path) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    has_contract = nodes_df["contract_text"].astype(str).str.len() > 10
    counts = has_contract.value_counts().reindex([False, True], fill_value=0)
    sns.barplot(
        x=["Wallet (EOA)", "Contract"],
        y=counts.values,
        palette="viridis",
        ax=ax,
    )
    ax.set_title("Account Types: Contracts vs Wallets")
    ax.set_ylabel("Number of Unique Addresses")
    fig.tight_layout()
    fig.savefig(savepath)
    plt.close(fig)


def plot_label_balance(nodes_df: pd.DataFrame, savepath: Path) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = nodes_df["y"].value_counts().reindex([0, 1], fill_value=0)
    sns.barplot(x=["Benign", "Fraud"], y=counts.values, palette="magma", ax=ax)
    ax.set_title("Label Balance: Benign vs Fraud Addresses")
    ax.set_ylabel("Number of Unique Addresses")
    fig.tight_layout()
    fig.savefig(savepath)
    plt.close(fig)
