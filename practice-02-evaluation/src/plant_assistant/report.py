"""Comparison table and the quality / latency / cost chart."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

TABLE_COLUMNS = {
    "name": "setup",
    "valid_first_pct": "valid 1st try, %",
    "valid_final_pct": "valid final, %",
    "field_accuracy_pct": "field accuracy, %",
    "record_exact_pct": "all fields right, %",
    "invented_pct": "invented, %",
    "latency_median_s": "median latency, s",
    "cost_per_1000_usd": "USD per 1000 records",
}


def comparison_table(summaries: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(summaries)[list(TABLE_COLUMNS)].rename(columns=TABLE_COLUMNS)
    return df.reset_index(drop=True)


def per_field_table(summaries: list[dict[str, Any]]) -> pd.DataFrame:
    rows = {s["name"]: s["per_field_pct"] for s in summaries}
    return pd.DataFrame(rows).T


def plot_quality_latency_cost(summaries: list[dict[str, Any]], path: Path | None = None, title: str | None = None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=120)
    for s in summaries:
        x = max(s["latency_median_s"], 0.001)
        y = s["field_accuracy_pct"]
        cost = s["cost_per_1000_usd"]
        free = cost == 0
        ax.scatter(x, y, s=140, marker="o" if not free else "s",
                   color="#d1495b" if not free else "#2e86ab", edgecolor="black", zorder=3)
        label = f"{s['name']}\n" + ("free / self-hosted" if free else f"${cost:.2f} per 1000")
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(8, 6), fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("median latency per record, s (log scale)", fontsize=11)
    ax.set_ylabel("field accuracy, %", fontsize=11)
    ax.set_title(title or "Quality vs latency vs cost on eval_50", fontsize=13)
    ax.grid(True, which="both", alpha=0.3)
    ax.text(0.01, 0.01, "blue square = no per-token cost; red circle = paid API", transform=ax.transAxes,
            fontsize=8, color="gray")
    fig.tight_layout()
    if path:
        fig.savefig(path)
    return fig
