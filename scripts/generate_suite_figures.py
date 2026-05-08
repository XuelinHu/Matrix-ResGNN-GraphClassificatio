"""根据主 benchmark 汇总 CSV 生成论文使用的整体结果图。"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

# 仓库根目录：用于把脚本中的相对路径统一定位到项目根路径。
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment_catalog import MAIN_DATASETS, MODEL_DISPLAY
from src.experiment_paths import DEFAULT_EXPERIMENT_VERSION, normalize_version, record_dir
from scripts.plot_style import MODEL_COLORS, apply_paper_style, style_axis


def parse_args() -> argparse.Namespace:
    """解析命令行参数，返回当前脚本需要的实验配置。"""
    parser = argparse.ArgumentParser(description="Generate paper-facing suite figures from benchmark_summary.csv.")
    parser.add_argument("--version", default=DEFAULT_EXPERIMENT_VERSION)
    return parser.parse_args()


def load_rows(summary_path: Path) -> List[Dict[str, object]]:
    """读取输入结果文件，并转换成后续汇总需要的结构化行。"""
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
                    "std_best_test_acc": float(row["std_best_test_acc"]),
                }
            )
    return rows


def filtered_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """筛选绘图所需的 GCNConv 主结果行。"""
    return [row for row in rows if row["operator"] == "GCNConv"]


def plot_main_bar(rows: List[Dict[str, object]], out_dir: Path) -> None:
    """绘制 GCNConv 主 benchmark 柱状图。"""
    datasets = MAIN_DATASETS
    models = list(MODEL_DISPLAY.keys())
    x = np.arange(len(datasets))
    width = 0.9 / len(models)
    offsets = np.linspace(-(len(models) - 1) / 2, (len(models) - 1) / 2, len(models)) * width

    apply_paper_style()
    fig, ax = plt.subplots(figsize=(12, 5.8))
    for offset, model in zip(offsets, models):
        means = []
        stds = []
        for dataset in datasets:
            matched = [row for row in rows if row["dataset"] == dataset and row["model"] == model]
            means.append(matched[0]["mean_best_test_acc"] if matched else 0.0)
            stds.append(matched[0]["std_best_test_acc"] if matched else 0.0)
        bars = ax.bar(
            x + offset,
            means,
            yerr=stds,
            width=width,
            label=MODEL_DISPLAY[model],
            color=MODEL_COLORS[model],
            edgecolor="black",
            linewidth=0.7,
            capsize=3,
            error_kw={"elinewidth": 0.8, "capthick": 0.8},
            zorder=3,
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
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Mean best test accuracy")
    ax.set_title("Main benchmark comparison (GCNConv)", fontweight="bold")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    ax.set_ylim(0.0, max([row["mean_best_test_acc"] for row in rows], default=1.0) + 0.1)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_main_benchmark_gcnconv.pdf")
    fig.savefig(out_dir / "fig_main_benchmark_gcnconv.png")
    plt.close(fig)


def plot_model_wins(rows: List[Dict[str, object]], out_dir: Path) -> None:
    """绘制不同模型在数据集-算子组合上的胜出次数图。"""
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
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    bars = ax.bar(
        [MODEL_DISPLAY[m] for m in models],
        values,
        color=[MODEL_COLORS[m] for m in models],
        edgecolor="black",
        linewidth=0.7,
        zorder=3,
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
    ax.set_ylabel("Datasets won")
    ax.set_title("Model win counts across summarized datasets", fontweight="bold")
    ax.tick_params(axis="x", rotation=20)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_model_win_counts.pdf")
    fig.savefig(out_dir / "fig_model_win_counts.png")
    plt.close(fig)


def main() -> None:
    """脚本主入口，串联参数解析、数据读取、处理和结果写出。"""
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
