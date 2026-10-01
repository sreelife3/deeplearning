# -*- coding: utf-8 -*-
"""Aggregate across seeds, then the results table, the Pareto plot and the
confusion matrix.

The aggregation is where the honesty lives. Four models within a point of each
other on a single seed is noise; if the seed spread is wider than the gap
between two models, they are indistinguishable and the report says so.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ACTIVITIES = ["WALKING", "WALK_UP", "WALK_DOWN", "SITTING", "STANDING", "LAYING"]

NAVY, TEAL, ORANGE, RED = "#17304f", "#0b7c86", "#e07b00", "#c62828"
COLOURS = {"rnn": "#8b95b8", "lstm": NAVY, "gru": TEAL, "cnn1d": ORANGE}


def aggregate(raw_csv="results/raw.csv", out_csv="results/summary.csv"):
    df = pd.read_csv(raw_csv)
    g = df.groupby(["model", "precision"])
    summary = g.agg(
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_min=("macro_f1", "min"),
        macro_f1_max=("macro_f1", "max"),
        accuracy_mean=("accuracy", "mean"),
        params=("params", "first"),
        size_kb=("size_kb", "mean"),
        latency_med_ms=("latency_med_ms", "median"),
        latency_p95_ms=("latency_p95_ms", "median"),
        n_seeds=("seed", "nunique"),
    ).reset_index()
    summary["macro_f1_range"] = (summary.macro_f1_max - summary.macro_f1_min).round(4)
    for c in ("macro_f1_mean", "accuracy_mean"):
        summary[c] = summary[c].round(4)
    summary["size_kb"] = summary.size_kb.round(1)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)
    return summary


def table(summary):
    """The markdown table the README leads with."""
    lines = ["| Model | Precision | Macro-F1 (mean ± range) | Params | Size | p50 ms | p95 ms |",
             "|---|---|---|---|---|---|---|"]
    for _, r in summary.sort_values(["model", "precision"]).iterrows():
        lines.append("| %s | %s | %.4f ± %.4f | %s | %.0f KB | %.2f | %.2f |"
                     % (r.model.upper(), r.precision, r.macro_f1_mean,
                        r.macro_f1_range, f"{int(r.params):,}", r.size_kb,
                        r.latency_med_ms, r.latency_p95_ms))
    return "\n".join(lines)


def indistinguishable(summary, precision="fp32"):
    """Pairs whose gap is inside the seed spread. These are the claims you are
    NOT allowed to make."""
    s = summary[summary.precision == precision]
    out = []
    for i, a in s.iterrows():
        for j, b in s.iterrows():
            if a.model >= b.model:
                continue
            gap = abs(a.macro_f1_mean - b.macro_f1_mean)
            spread = max(a.macro_f1_range, b.macro_f1_range)
            if gap < spread:
                out.append((a.model, b.model, round(gap, 4), round(spread, 4)))
    return out


def pareto_plot(summary, path="figures/pareto.png"):
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for _, r in summary.iterrows():
        col = COLOURS.get(r.model, NAVY)
        filled = r.precision == "fp32"
        ax.scatter(r.latency_med_ms, r.macro_f1_mean,
                   s=40 + r.size_kb * 0.45,
                   facecolor=col if filled else "white",
                   edgecolor=col, linewidth=1.8, zorder=3,
                   label="_nolegend_")
        ax.errorbar(r.latency_med_ms, r.macro_f1_mean,
                    yerr=r.macro_f1_range / 2, color=col, alpha=.45,
                    capsize=3, zorder=2)
        ax.annotate(f"{r.model} {r.precision}",
                    (r.latency_med_ms, r.macro_f1_mean),
                    textcoords="offset points", xytext=(9, 6),
                    fontsize=8.5, color=col, fontweight="bold")

    # the frontier: points nothing else beats on both axes at once
    pts = summary[["latency_med_ms", "macro_f1_mean"]].values
    keep = []
    for i, (x, y) in enumerate(pts):
        if not any((px <= x and py >= y) and (px < x or py > y)
                   for k, (px, py) in enumerate(pts) if k != i):
            keep.append(i)
    front = summary.iloc[keep].sort_values("latency_med_ms")
    ax.plot(front.latency_med_ms, front.macro_f1_mean, color="#c8d2e2",
            lw=1.6, ls=(0, (5, 4)), zorder=1)

    ax.set_xlabel("median latency per window, batch size 1, single thread (ms)")
    ax.set_ylabel("macro-F1 (mean over seeds, bar = range)")
    ax.set_title("Accuracy against latency — filled = FP32, hollow = INT8,\n"
                 "marker size = model size on disk", fontsize=10, color=NAVY)
    ax.set_xscale("log")
    ax.grid(alpha=.25, zorder=0)
    ax.spines[["right", "top"]].set_visible(False)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def confusion_plot(preds_npy, path="figures/confusion.png", title=""):
    from sklearn.metrics import confusion_matrix
    arr = np.load(preds_npy)
    cm = confusion_matrix(arr[1], arr[0], labels=range(6))
    cmn = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(6), ACTIVITIES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(6), ACTIVITIES, fontsize=8)
    for i in range(6):
        for j in range(6):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                        color="white" if cmn[i, j] > .5 else "#20252e")
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(title or "Confusion matrix", fontsize=10, color=NAVY)
    fig.colorbar(im, ax=ax, fraction=.046, label="row-normalised")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

    # SITTING is index 3, STANDING is 4 -- the pair that matters here
    return {"sitting_as_standing": int(cm[3, 4]),
            "standing_as_sitting": int(cm[4, 3]),
            "sit_stand_share_of_errors":
                round((cm[3, 4] + cm[4, 3]) / max(cm.sum() - np.trace(cm), 1), 3)}
