from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment_catalog import MAIN_DATASETS, MODEL_DISPLAY
from src.experiment_paths import DEFAULT_EXPERIMENT_VERSION, normalize_version, record_dir


MODEL_COLORS: Dict[str, str] = {
    "Plain": "#7f7f7f",
    "VerticalRes": "#1f77b4",
    "HorizontalRes": "#2ca02c",
    "MatrixRes": "#d62728",
    "MatrixResGated": "#9467bd",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper-facing suite figures from benchmark_summary.csv.")
    parser.add_argument("--version", default=DEFAULT_EXPERIMENT_VERSION)
    return parser.parse_args()


def load_rows(summary_path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "dataset": row["dataset"],
                    "model": row["model"],
                    "operator": row["operator"],
                    "mean_best_test_acc": float(row["mean_best_test_acc"]),
                }
            )
    return rows


def filtered_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    return [row for row in rows if row["operator"] == "GCNConv"]


def plot_main_bar(rows: List[Dict[str, object]], out_dir: Path) -> None:
    datasets = MAIN_DATASETS
    models = list(MODEL_DISPLAY.keys())
    x = np.arange(len(datasets))
    width = 0.9 / len(models)
    offsets = np.linspace(-(len(models) - 1) / 2, (len(models) - 1) / 2, len(models)) * width

    fig, ax = plt.subplots(figsize=(12, 6))
    for offset, model in zip(offsets, models):
        means = []
        for dataset in datasets:
            matched = [row for row in rows if row["dataset"] == dataset and row["model"] == model]
            means.append(matched[0]["mean_best_test_acc"] if matched else 0.0)
        bars = ax.bar(
            x + offset,
            means,
            width=width,
            label=MODEL_DISPLAY[model],
            color=MODEL_COLORS[model],
            edgecolor="black",
            linewidth=0.5,
        )
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=12)
    ax.set_ylabel("Mean best test accuracy", fontsize=12)
    ax.set_title("Main benchmark comparison (GCNConv)", fontsize=14, fontweight="bold")
    ax.legend(frameon=False, ncol=3)
    ax.set_ylim(0.0, max([row["mean_best_test_acc"] for row in rows], default=1.0) + 0.1)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_main_benchmark_gcnconv.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig_main_benchmark_gcnconv.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_model_wins(rows: List[Dict[str, object]], out_dir: Path) -> None:
    win_counts = {model: 0 for model in MODEL_DISPLAY}
    datasets = sorted(set(row["dataset"] for row in rows))
    for dataset in datasets:
        matched = [row for row in rows if row["dataset"] == dataset]
        if not matched:
            continue
        best = max(matched, key=lambda row: row["mean_best_test_acc"])
        win_counts[best["model"]] += 1

    models = list(MODEL_DISPLAY.keys())
    values = [win_counts[model] for model in models]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        [MODEL_DISPLAY[m] for m in models],
        values,
        color=[MODEL_COLORS[m] for m in models],
        edgecolor="black",
        linewidth=0.5,
    )
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{int(bar.get_height())}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylabel("Datasets won", fontsize=12)
    ax.set_title("Model win counts across summarized datasets", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_model_win_counts.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig_model_win_counts.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    version = normalize_version(args.version)
    summary_path = record_dir(ROOT, version) / "summaries" / "benchmark_summary.csv"
    rows = filtered_rows(load_rows(summary_path))
    out_dir = ROOT / "figures" / "exp"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_main_bar(rows, out_dir)
    plot_model_wins(rows, out_dir)
    print(out_dir)


if __name__ == "__main__":
    main()
