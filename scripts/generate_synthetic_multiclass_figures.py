"""根据合成多分类 benchmark 汇总结果生成论文图表。"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams

# 仓库根目录：用于从任意工作目录运行脚本时定位数据和输出目录。
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_style import MODEL_COLORS, MODEL_LINESTYLES, MODEL_MARKERS, style_axis


MODEL_ORDER = ["Plain", "VerticalRes", "HorizontalRes", "MatrixRes", "MatrixResGated"]
MODEL_LABELS: Dict[str, str] = {
    "Plain": "Plain",
    "VerticalRes": "VerticalRes",
    "HorizontalRes": "HorizontalRes",
    "MatrixRes": "MatrixRes",
    "MatrixResGated": "MatrixResGated",
}


def parse_args() -> argparse.Namespace:
    """解析命令行参数，允许指定结果版本和输出目录。"""
    parser = argparse.ArgumentParser(description="Generate synthetic multiclass scaling figures.")
    parser.add_argument("--summary", default="records/SYN_MULTI_FULL/summaries/benchmark_summary.csv")
    parser.add_argument("--out-dir", default="figures/exp")
    return parser.parse_args()


def apply_compact_paper_style() -> None:
    """设置适合双栏论文结果图的紧凑字体和导出风格。"""
    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    rcParams["font.size"] = 10
    rcParams["axes.titlesize"] = 11
    rcParams["axes.labelsize"] = 10
    rcParams["xtick.labelsize"] = 9
    rcParams["ytick.labelsize"] = 9
    rcParams["legend.fontsize"] = 8.6
    rcParams["figure.dpi"] = 300
    rcParams["savefig.dpi"] = 300
    rcParams["figure.facecolor"] = "white"
    rcParams["axes.facecolor"] = "white"
    rcParams["savefig.facecolor"] = "white"
    rcParams["axes.linewidth"] = 0.9
    rcParams["legend.frameon"] = False
    rcParams["savefig.bbox"] = "tight"


def load_summary(summary_path: Path) -> pd.DataFrame:
    """读取合成多分类结果，并从数据集名称中解析类别数量。"""
    df = pd.read_csv(summary_path)
    df["class_count"] = df["dataset"].map(lambda value: int(re.search(r"_C(\d+)$", value).group(1)))
    return df


def aggregate_by_class(df: pd.DataFrame) -> pd.DataFrame:
    """按类别数和模型汇总所有图分布与算子的平均指标。"""
    return (
        df.groupby(["class_count", "model"], as_index=False)
        .agg(
            mean_best_test_acc=("mean_best_test_acc", "mean"),
            mean_test_macro_f1=("mean_test_macro_f1", "mean"),
            mean_test_normalized_acc=("mean_test_normalized_acc", "mean"),
        )
        .sort_values(["class_count", "model"])
    )


def plot_scaling_curves(class_summary: pd.DataFrame, out_dir: Path) -> None:
    """绘制 C2-C8 类别数增长下的准确率和归一化准确率曲线。"""
    apply_compact_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.15), sharex=True)
    metric_specs = [
        ("mean_best_test_acc", "Accuracy", "(a) Accuracy"),
        ("mean_test_normalized_acc", "Normalized accuracy", "(b) Normalized accuracy"),
    ]

    for ax, (metric, ylabel, title) in zip(axes, metric_specs):
        for model in MODEL_ORDER:
            sub = class_summary[class_summary["model"] == model].sort_values("class_count")
            ax.plot(
                sub["class_count"],
                sub[metric],
                label=MODEL_LABELS[model],
                color=MODEL_COLORS[model],
                linestyle=MODEL_LINESTYLES[model],
                marker=MODEL_MARKERS[model],
                linewidth=1.65,
                markersize=4.5,
            )
        ax.set_title(title)
        ax.set_xlabel("Number of classes")
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(2, 9))
        ax.set_ylim(0.50, 0.98)
        style_axis(ax, with_grid=True, grid_axis="y")

    axes[0].legend(
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(1.04, -0.24),
        columnspacing=1.1,
        handlelength=2.2,
    )
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig_synthetic_multiclass_scaling.pdf")
    fig.savefig(out_dir / "fig_synthetic_multiclass_scaling.png", dpi=600)
    plt.close(fig)


def main() -> None:
    """脚本主入口，完成数据读取、汇总和图像导出。"""
    args = parse_args()
    summary_path = ROOT / args.summary
    out_dir = ROOT / args.out_dir
    df = load_summary(summary_path)
    class_summary = aggregate_by_class(df)
    plot_scaling_curves(class_summary, out_dir)
    print(out_dir / "fig_synthetic_multiclass_scaling.pdf")


if __name__ == "__main__":
    main()
