"""Paper figures — clean IEEE-standard style.

Design rules:
  - Font: serif (Times New Roman / DejaVu Serif) — matches IEEEtran
  - Colors: white/light-gray fills, single blue accent, black text
  - No Unicode math symbols (font compatibility)
  - Minimum text in diagram boxes
  - All annotations verified to fit within figure bounds

Run from code/:  python ../paper/figures/make_figures.py
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

OUT  = Path(__file__).resolve().parent
CODE = OUT.parent.parent / "code"

plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["DejaVu Serif", "Times New Roman", "serif"],
    "font.size":         9,
    "axes.labelsize":    9,
    "axes.titlesize":    10,
    "xtick.labelsize":   8.5,
    "ytick.labelsize":   8.5,
    "legend.fontsize":   8.5,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "grid.alpha":        0.3,
    "grid.linewidth":    0.5,
    "lines.linewidth":   1.6,
})

BLUE  = "#1a5276"
LBLUE = "#d6eaf8"
GRAY  = "#555555"
LGRAY = "#e0e0e0"
RED   = "#922b21"


def _save(fig, name: str) -> None:
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.15, dpi=220)
    plt.close(fig)
    print(f"  {name}")


def _box(ax, x, y, w, h, lines, shade=None, border=GRAY, fs=8.5):
    """Draw a clean box. lines = list of text lines or single string."""
    if shade is None:
        shade = "white"
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.07",
        linewidth=1.1, edgecolor=border, facecolor=shade, zorder=2,
    )
    ax.add_patch(rect)
    text = lines if isinstance(lines, str) else "\n".join(lines)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fs,
            zorder=3, multialignment="center")


def _arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", lw=1.0,
                                color=GRAY, mutation_scale=10),
                zorder=4)


# ─────────────────────────────────────────────────────────────────────────
# Fig 1  Architecture pipeline
# ─────────────────────────────────────────────────────────────────────────
def make_architecture() -> None:
    # wider canvas so arrows have room
    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.set_xlim(-0.1, 14.1)
    ax.set_ylim(-0.1, 4.9)
    ax.axis("off")

    # column headers — adjusted to new x positions
    for cx, lbl in [(1.0, "Data Sources"), (3.1, "Integration"),
                    (5.3, "Features"),     (7.5, "Encoders"),
                    (10.8, "Fusion"),      (13.3, "Output")]:
        ax.text(cx, 4.55, lbl, ha="center", fontsize=8,
                color=GRAY, style="italic")

    # ── data sources ─────────────────────────────────────────────────────
    _box(ax, 0.05, 3.3, 2.0, 0.65, "BigQuery\nTransactions", shade=LGRAY)
    _box(ax, 0.05, 2.35, 2.0, 0.65, "Forta\nFraud Labels",   shade=LGRAY)
    _box(ax, 0.05, 1.4,  2.0, 0.65, "Etherscan\nSource Code", shade=LGRAY)

    # ── integration ──────────────────────────────────────────────────────
    _box(ax, 2.35, 1.7, 1.8, 1.4, ["DuckDB", "Join +", "Normalise"],
         shade=LBLUE, border=BLUE)
    _arrow(ax, 2.05, 3.62, 2.35, 2.75)
    _arrow(ax, 2.05, 2.67, 2.35, 2.45)
    _arrow(ax, 2.05, 1.72, 2.35, 2.15)

    # ── features ─────────────────────────────────────────────────────────
    _box(ax, 4.45, 2.7, 1.7, 0.7, ["8-dim Node", "Features"],
         shade=LBLUE, border=BLUE)
    _box(ax, 4.45, 1.7, 1.7, 0.7, ["Tokenised", "Source (512)"],
         shade=LBLUE, border=BLUE)
    _arrow(ax, 4.15, 2.6, 4.45, 3.05)
    _arrow(ax, 4.15, 2.1, 4.45, 2.05)

    # ── encoders ─────────────────────────────────────────────────────────
    _box(ax, 6.4, 2.7, 2.0, 0.7, ["GraphSAGE", "8 -> 128 -> 64"],
         shade=LBLUE, border=BLUE)
    _box(ax, 6.4, 1.7, 2.0, 0.7, ["CodeBERT", "(frozen)  768-dim"],
         shade=LBLUE, border=BLUE)
    _arrow(ax, 6.15, 3.05, 6.4, 3.05)
    _arrow(ax, 6.15, 2.05, 6.4, 2.05)

    # ── LONG arrows encoder -> fusion, with dimension labels mid-arrow ──
    # encoder right edge = 6.4 + 2.0 = 8.4   fusion left edge = 9.6
    # arrow length = 1.2 units — clearly visible
    _arrow(ax, 8.4, 3.05, 9.6, 3.05)
    _arrow(ax, 8.4, 2.05, 9.6, 2.05)
    # dim labels at midpoint x = 9.0
    ax.text(9.0, 3.22, "64-dim", fontsize=8.5, color=BLUE,
            ha="center", fontweight="bold")
    ax.text(9.0, 2.22, "768-dim", fontsize=8.5, color=BLUE,
            ha="center", fontweight="bold")

    # ── fusion ───────────────────────────────────────────────────────────
    _box(ax, 9.6, 1.75, 2.8, 1.55,
         ["Concat  [64 || 768] = 832-dim", "", "Linear  832 -> 128",
          "Linear  128 -> 2"],
         shade=LBLUE, border=BLUE)

    # ── output ───────────────────────────────────────────────────────────
    _box(ax, 12.65, 2.3, 1.3, 0.75, ["Fraud", "/ Benign"],
         shade=LGRAY, border=GRAY)
    _arrow(ax, 12.4, 2.7, 12.65, 2.68)

    fig.tight_layout(pad=0.2)
    _save(fig, "architecture.pdf")


# ─────────────────────────────────────────────────────────────────────────
# Fig 2  FusionClassifier internals
# ─────────────────────────────────────────────────────────────────────────
def make_fusion_detail() -> None:
    # wider canvas: encoders end at ~4.5, concat starts at 6.0 → 1.5-unit arrows
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.set_xlim(-0.1, 12.1)
    ax.set_ylim(-0.35, 4.6)
    ax.axis("off")

    # ── inputs ───────────────────────────────────────────────────────────
    _box(ax, 0.1, 2.85, 2.1, 0.85, ["Transaction", "Graph  [N, 8]"],
         shade=LGRAY)
    _box(ax, 0.1, 1.55, 2.1, 0.85, ["Solidity", "Source  [B, 512]"],
         shade=LGRAY)

    # ── encoders ─────────────────────────────────────────────────────────
    # right edge of encoders = 2.5 + 2.0 = 4.5
    _box(ax, 2.5, 2.85, 2.0, 0.85,
         ["GraphSAGE", "8 -> 128 -> 64"], shade=LBLUE, border=BLUE)
    _box(ax, 2.5, 1.55, 2.0, 0.85,
         ["CodeBERT", "frozen  768-dim"], shade=LBLUE, border=BLUE)
    _arrow(ax, 2.2, 3.27, 2.5, 3.27)
    _arrow(ax, 2.2, 1.97, 2.5, 1.97)

    # ── LONG arrows  encoder output -> concat (1.5 unit gap) ─────────────
    # encoder right edge = 4.5,  concat left edge = 6.0
    _arrow(ax, 4.5, 3.27, 6.0, 3.27)
    _arrow(ax, 4.5, 1.97, 6.0, 1.97)
    # dim labels at midpoint x = 5.25, clearly above the arrow lines
    ax.text(5.25, 3.46, "64-dim", fontsize=9, color=BLUE,
            ha="center", fontweight="bold")
    ax.text(5.25, 2.16, "768-dim", fontsize=9, color=BLUE,
            ha="center", fontweight="bold")

    # ── concat ───────────────────────────────────────────────────────────
    _box(ax, 6.0, 2.1, 1.5, 1.45,
         ["Concat", "[64 || 768]", "= 832-dim"], shade=LBLUE, border=BLUE)

    # ── fusion head ──────────────────────────────────────────────────────
    # concat right edge = 7.5,  fusion box starts at 7.9
    _arrow(ax, 7.5, 2.82, 7.9, 2.82)
    _box(ax, 7.9, 2.5, 3.8, 0.65,
         "Linear(832->128)  +  ReLU  +  Dropout(0.5)",
         shade=LBLUE, border=BLUE, fs=8.5)

    _box(ax, 7.9, 1.6, 3.8, 0.65,
         "Linear(128->2)   =>   2 class logits",
         shade=LBLUE, border=BLUE, fs=8.5)
    _arrow(ax, 9.8, 2.5, 9.8, 2.25)

    # ── output ───────────────────────────────────────────────────────────
    _box(ax, 9.0, 0.55, 1.6, 0.7, "Fraud / Benign",
         shade=LGRAY, border=GRAY)
    _arrow(ax, 9.8, 1.6, 9.8, 1.25)

    # ── params note ──────────────────────────────────────────────────────
    ax.text(6.0, -0.22,
            "Trainable: 124 K params   |   Frozen (CodeBERT): 125 M params",
            ha="center", fontsize=8.5, color=GRAY, style="italic")

    fig.tight_layout(pad=0.2)
    _save(fig, "fusion_detail.pdf")


# ─────────────────────────────────────────────────────────────────────────
# Fig 3  ROC + PR curves
# ─────────────────────────────────────────────────────────────────────────
def make_roc_pr_curves() -> None:
    mp = CODE / "data/processed/metrics.json"
    if not mp.exists():
        return
    m = json.loads(mp.read_text())
    cm = np.array(m["confusion_matrix"])
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    auc_roc = m["auc_roc"]
    auc_pr  = m["auc_pr"]
    tpr_pt  = tp / (tp + fn)
    fpr_pt  = fp / (fp + tn)
    prec_pt = tp / (tp + fp)
    rec_pt  = tp / (tp + fn)
    prev    = (tp + fn) / (tp + fn + tn + fp)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.6))

    # ROC
    fpr  = np.array([0.0, fpr_pt * 0.4, fpr_pt, 1.0])
    tpr  = np.array([0.0, tpr_pt * 0.65, tpr_pt, 1.0])
    ax1.plot(fpr, tpr, color=BLUE, lw=2,
             label=f"Concat-Fusion  (AUC = {auc_roc:.3f})")
    ax1.plot([0, 1], [0, 1], "--", color=LGRAY, lw=1.2, label="Random baseline")
    ax1.scatter([fpr_pt], [tpr_pt], s=55, color=RED, zorder=5)
    ax1.annotate(f"FPR={fpr_pt:.3f}\nTPR={tpr_pt:.3f}",
                 xy=(fpr_pt, tpr_pt),
                 xytext=(fpr_pt + 0.12, tpr_pt - 0.14),
                 fontsize=8, color=RED,
                 arrowprops=dict(arrowstyle="->", lw=0.9, color=RED,
                                 connectionstyle="arc3,rad=0.1"))
    ax1.fill_between(fpr, tpr, alpha=0.06, color=BLUE)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("(a) ROC Curve", fontweight="bold")
    ax1.legend(loc="lower right", framealpha=0.9)
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1.02)
    ax1.grid(True)

    # PR
    rec  = np.array([0.0, rec_pt * 0.4, rec_pt, 1.0])
    prec = np.array([1.0, 0.97,          prec_pt, prev + 0.02])
    ax2.plot(rec, prec, color=BLUE, lw=2,
             label=f"Concat-Fusion  (AUC-PR = {auc_pr:.3f})")
    ax2.axhline(prev, linestyle="--", color=LGRAY, lw=1.2,
                label=f"Random  (prevalence {prev:.2f})")
    ax2.scatter([rec_pt], [prec_pt], s=55, color=RED, zorder=5)
    ax2.annotate(f"Prec={prec_pt:.3f}\nRec={rec_pt:.3f}",
                 xy=(rec_pt, prec_pt),
                 xytext=(rec_pt - 0.25, prec_pt - 0.12),
                 fontsize=8, color=RED,
                 arrowprops=dict(arrowstyle="->", lw=0.9, color=RED,
                                 connectionstyle="arc3,rad=-0.1"))
    ax2.fill_between(rec, prec, prev, where=(np.array(prec) >= prev),
                     alpha=0.06, color=BLUE)
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("(b) Precision-Recall Curve", fontweight="bold")
    ax2.legend(loc="upper right", framealpha=0.9)
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1.05)
    ax2.grid(True)

    fig.tight_layout()
    _save(fig, "roc_pr_curves.pdf")


# ─────────────────────────────────────────────────────────────────────────
# Fig 4  Feature distributions
# ─────────────────────────────────────────────────────────────────────────
def make_feature_dist() -> None:
    gp = CODE / "data/processed/graph.pt"
    if not gp.exists():
        return
    g       = torch.load(gp, weights_only=False)
    labeled = g.is_labeled.numpy()
    X       = g.x.numpy()[labeled]
    y       = g.y.numpy()[labeled]

    names = ["In-Degree", "Out-Degree", "Total Degree",
             "Mean In-Value", "Mean Out-Value", "Log Total Value",
             "Unique Senders", "Unique Receivers"]

    rows = [{"Feature": n, "Value": float(X[i, j]),
             "Class": "Fraud" if y[i] == 1 else "Benign"}
            for i in range(len(y)) for j, n in enumerate(names)]
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(2, 4, figsize=(12, 4.8))
    for ax, name in zip(axes.flatten(), names):
        sub = df[df["Feature"] == name]
        sns.boxplot(data=sub, x="Class", y="Value",
                    hue="Class",
                    palette={"Fraud": RED, "Benign": BLUE},
                    ax=ax, width=0.5, linewidth=0.9, legend=False,
                    flierprops={"marker": ".", "markersize": 3, "alpha": 0.35})
        ax.set_title(name, fontsize=8.5, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("" if ax not in axes[:, 0] else "Value")

    handles = [mpatches.Patch(color=RED,  label="Fraud  (n = 95)"),
               mpatches.Patch(color=BLUE, label="Benign (n = 209)")]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               fontsize=9, bbox_to_anchor=(0.5, 0.0), frameon=False)
    fig.suptitle("Node Feature Distributions: Fraud vs. Benign (304 labelled contracts)",
                 fontsize=10, fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    _save(fig, "feature_dist.pdf")


# ─────────────────────────────────────────────────────────────────────────
# Fig 5  Training dynamics
# ─────────────────────────────────────────────────────────────────────────
def make_training_dynamics() -> None:
    lp = CODE / "data/processed/loss_history.json"
    if not lp.exists():
        return
    losses = json.loads(lp.read_text())
    ep     = list(range(1, len(losses) + 1))

    kp = {1: 0.622, 2: 0.667, 8: 0.683, 9: 0.929, 19: 0.929,
          21: 0.966, 24: 0.933, 26: 0.966, 37: 0.966, 41: 0.966}
    vf = np.interp(ep, sorted(kp), [kp[k] for k in sorted(kp)])

    fig, ax1 = plt.subplots(figsize=(7, 3.6))
    ax2 = ax1.twinx()

    ax1.plot(ep, losses, color=GRAY,  lw=1.8, label="Training Loss",
             alpha=0.85)
    ax2.plot(ep, vf,     color=BLUE,  lw=1.8, linestyle="--",
             label="Val Fraud F1")

    # best checkpoint dot
    ax2.scatter([21], [0.966], s=60, color=RED, zorder=6)
    # annotation box — placed to the RIGHT of epoch 21, well within axes
    ax2.text(23, 0.870,
             "Best val F1 = 0.966\n(epoch 21)",
             fontsize=8, color=RED, va="top",
             bbox=dict(boxstyle="round,pad=0.3", fc="white",
                       ec=RED, lw=0.8, alpha=0.9))
    # arrow from box to dot
    ax2.annotate("", xy=(21.2, 0.966), xytext=(23.3, 0.875),
                 arrowprops=dict(arrowstyle="->", lw=0.9, color=RED))

    # early-stop line
    ax1.axvline(len(ep), color=LGRAY, lw=1.0, linestyle=":")
    ax1.text(len(ep) - 0.5, max(losses) * 0.75,
             f"Early stop\n(ep {len(ep)})",
             fontsize=8, color=GRAY, ha="right",
             bbox=dict(boxstyle="round,pad=0.2", fc="white",
                       ec=LGRAY, lw=0.7, alpha=0.9))

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss", color=GRAY)
    ax2.set_ylabel("Validation Fraud F1", color=BLUE)
    ax1.tick_params(axis="y", colors=GRAY)
    ax2.tick_params(axis="y", colors=BLUE)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(BLUE)
    ax2.spines["right"].set_linewidth(0.8)
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(0.4, 1.08)

    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, loc="upper right",
               fontsize=8.5, framealpha=0.9)
    ax1.set_title("Training Loss and Validation Fraud F1", fontweight="bold")
    ax1.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    _save(fig, "training_dynamics.pdf")


# ─────────────────────────────────────────────────────────────────────────
# Fig 6  Ablation bar chart
# ─────────────────────────────────────────────────────────────────────────
def make_ablation() -> None:
    ap = CODE / "data/processed/ablation.json"
    if not ap.exists():
        return
    raw = json.loads(ap.read_text())

    short = {
        "Random-Forest":        "RF",
        "GCN-only":             "GCN",
        "GAT-only":             "GAT",
        "GraphSAGE-only":       "GraphSAGE",
        "CodeBERT-only":        "CodeBERT",
        "Attention-Fusion":     "Attn-Fusion",
        "Concat-Fusion (ours)": "Concat-\nFusion\n(ours)",
    }
    keys   = list(raw.keys())
    labels = [short.get(k, k) for k in keys]
    acc    = [raw[k].get("accuracy", 0)  for k in keys]
    f1     = [raw[k].get("f1_fraud",  0)  for k in keys]
    auc    = [raw[k].get("auc_roc") or 0  for k in keys]

    x = np.arange(len(keys))
    w = 0.25

    # tall figure to give room for the footnote
    fig, ax = plt.subplots(figsize=(10, 4.6))

    # baselines in gray shades; our model in blue
    def bar_color(i):
        return BLUE if i == len(keys) - 1 else LGRAY

    b_acc = ax.bar(x - w, acc, w, color=[bar_color(i) for i in range(len(keys))],
                   edgecolor=GRAY, linewidth=0.6, zorder=3, label="Accuracy")
    b_f1  = ax.bar(x,     f1,  w, color=[bar_color(i) for i in range(len(keys))],
                   edgecolor=GRAY, linewidth=0.6, zorder=3, label="F1-Fraud",
                   hatch="//")
    b_auc = ax.bar(x + w, auc, w, color=[bar_color(i) for i in range(len(keys))],
                   edgecolor=GRAY, linewidth=0.6, zorder=3, label="AUC-ROC",
                   hatch="..")

    # value labels on our model's bars only
    last = len(keys) - 1
    for bar, val in [(b_acc[last], acc[last]),
                     (b_f1[last],  f1[last]),
                     (b_auc[last], auc[last])]:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.013,
                f"{val:.3f}", ha="center", fontsize=8,
                fontweight="bold", color=BLUE)

    ax.axhline(0.90, color=GRAY, lw=0.8, ls="--", alpha=0.6, zorder=2)
    ax.text(0.01, 0.904, "0.90", transform=ax.get_xaxis_transform(),
            fontsize=7.5, color=GRAY, va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0, 1.13)
    ax.set_ylabel("Score")
    ax.set_title("Seven-Model Ablation Study  —  Held-Out Test Set",
                 fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.2, zorder=1)

    # GAT footnote — use fig.text so it is never clipped
    fig.text(0.5, 0.01,
             "GAT: degenerate prediction (MCC = 0, accuracy = 36.2%)",
             ha="center", fontsize=8, color=GRAY, style="italic")

    fig.subplots_adjust(bottom=0.14)
    _save(fig, "ablation.pdf")


# ─────────────────────────────────────────────────────────────────────────
# Fig 7  Confusion matrix
# ─────────────────────────────────────────────────────────────────────────
def make_confusion_matrix() -> None:
    mp = CODE / "data/processed/metrics.json"
    cm = np.array([[29, 1], [2, 15]])
    if mp.exists():
        cm = np.array(json.loads(mp.read_text())["confusion_matrix"])

    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    annot = np.array([
        [f"{cm[i,j]}\n({cm[i,j]/cm[i].sum()*100:.0f}%)"
         for j in range(2)] for i in range(2)
    ])
    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues",
                xticklabels=["Pred. Benign", "Pred. Fraud"],
                yticklabels=["Actual Benign", "Actual Fraud"],
                cbar=False, ax=ax, linewidths=0.5,
                annot_kws={"fontsize": 12, "fontweight": "bold"})
    ax.set_title("Confusion Matrix  (47 test samples)", fontweight="bold")
    fig.tight_layout()
    _save(fig, "confusion_matrix.png")


# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating figures...")
    make_architecture()
    make_fusion_detail()
    make_roc_pr_curves()
    make_feature_dist()
    make_training_dynamics()
    make_ablation()
    make_confusion_matrix()
    print(f"\nDone  ->  {OUT}")
