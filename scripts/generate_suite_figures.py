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

from src.experiment_catalog import ALL_ACTIVE_DATASETS, MODEL_DISPLAY
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
    datasets = ALL_ACTIVE_DATASETS
    models = list(MODEL_DISPLAY.keys())
    x = np.arange(len(datasets))
    width = 0.9 / len(models)
    offsets = np.linspace(-(len(models) - 1) / 2, (len(models) - 1) / 2, len(models)) * width

    apply_paper_style()
    fig, ax = plt.subplots(figsize=(14.2, 6.2))
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
        for bar, std in zip(bars, stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.008,
                f"{bar.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                rotation=75,
                rotation_mode="anchor",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Mean best test accuracy")
    ax.set_title("Main benchmark comparison (GCNConv)", fontweight="bold")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.14), columnspacing=1.1, handlelength=1.6)
    y_upper = max([row["mean_best_test_acc"] + row["std_best_test_acc"] for row in rows], default=0.8) + 0.04
    ax.set_ylim(0.2, max(0.86, y_upper))
    style_axis(ax)
    fig.tight_layout(rect=[0, 0.09, 1, 1])
    fig.savefig(out_dir / "fig_main_benchmark_gcnconv.pdf")
    fig.savefig(out_dir / "fig_main_benchmark_gcnconv.png")
    plt.close(fig)


def plot_model_wins(rows: List[Dict[str, object]], out_dir: Path) -> None:
    """绘制不同模型在数据集-算子组合上的胜出次数图。"""
    win_counts = {model: 0 for model in MODEL_DISPLAY}
    combo_keys = sorted({(row["dataset"], row["operator"]) for row in rows})
    for dataset, operator in combo_keys:
        matched = [row for row in rows if row["dataset"] == dataset and row["operator"] == operator]
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
    ax.set_ylabel("Dataset-operator combinations won")
    ax.set_title("Winner counts across 24 benchmark combinations", fontweight="bold")
    ax.tick_params(axis="x", rotation=18)
    ax.set_ylim(0, max(values, default=0) + 1.2)
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
    all_rows = load_rows(summary_path)
    gcn_rows = filtered_rows(all_rows)
    out_dir = ROOT / "figures" / "exp"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_main_bar(gcn_rows, out_dir)
    plot_model_wins(all_rows, out_dir)
    print(out_dir)


if __name__ == "__main__":
    main()
