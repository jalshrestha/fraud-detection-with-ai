"""Graph-structure visualisations and confusion-matrix display."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from .style import apply_style


def plot_subgraph(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    savepath: Path,
    max_fraud: int = 20,
    max_benign: int = 100,
    random_state: int = 42,
) -> None:
    """Render a small subgraph that includes some fraud nodes and a benign sample."""
    apply_style()
    fraud_nodes = nodes_df[nodes_df["y"] == 1]["address"].tolist()[:max_fraud]
    benign_pool = nodes_df[nodes_df["y"] == 0]["address"]
    n_benign = min(max_benign, len(benign_pool))
    benign_nodes = benign_pool.sample(n=n_benign, random_state=random_state).tolist()
    sample = set(fraud_nodes) | set(benign_nodes)
    sub_edges = edges_df[edges_df["src"].isin(sample) & edges_df["dst"].isin(sample)]
    if sub_edges.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No subgraph edges available", ha="center", va="center")
        ax.axis("off")
        fig.savefig(savepath)
        plt.close(fig)
        return

    graph = nx.from_pandas_edgelist(
        sub_edges,
        "src",
        "dst",
        create_using=nx.DiGraph(),
    )
    fraud_set = set(fraud_nodes)
    colors = ["#d6336c" if n in fraud_set else "#1c7ed6" for n in graph.nodes()]
    degrees = dict(graph.degree())
    sizes = [degrees.get(n, 1) * 40 + 40 for n in graph.nodes()]

    fig, ax = plt.subplots(figsize=(8, 8))
    pos = nx.spring_layout(graph, k=0.9, iterations=50, seed=random_state)
    nx.draw(
        graph,
        pos,
        with_labels=False,
        node_color=colors,
        node_size=sizes,
        width=0.6,
        alpha=0.8,
        arrowsize=10,
        edge_color="#495057",
        ax=ax,
    )
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label="Benign", markerfacecolor="#1c7ed6", markersize=9),
        plt.Line2D([0], [0], marker="o", color="w", label="Fraud", markerfacecolor="#d6336c", markersize=9),
    ]
    ax.legend(handles=handles, title="Node Type", loc="upper right")
    ax.set_title("Subgraph of the Transaction Network")
    fig.tight_layout()
    fig.savefig(savepath)
    plt.close(fig)


def plot_confusion_matrix(
    cm: Sequence[Sequence[int]],
    savepath: Path,
    labels: Sequence[str] = ("Benign", "Fraud"),
    title: str = "Confusion Matrix",
) -> None:
    apply_style()
    cm_arr = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm_arr,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=list(labels),
        yticklabels=list(labels),
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(savepath)
    plt.close(fig)
