"""根据机制汇总 CSV 生成准确率、多样性、相似度、CKA 和梯度的机制图。"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# 仓库根目录：用于把脚本中的相对路径统一定位到项目根路径。
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_style import MODEL_COLORS, MODEL_LINESTYLES, MODEL_MARKERS, apply_paper_style, style_axis


def parse_args() -> argparse.Namespace:
    """解析命令行参数，返回当前脚本需要的实验配置。"""
    parser = argparse.ArgumentParser(description="Generate mechanism compact figure.")
    parser.add_argument("--version", default="LATEST")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    """读取 CSV 文件并返回字典行列表，供绘图或汇总使用。"""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    """脚本主入口，串联参数解析、数据读取、处理和结果写出。"""
    args = parse_args()
    summaries = ROOT / "records" / args.version / "summaries"
    branch_rows = read_csv(summaries / "branch_ablation_summary.csv")
    mech_rows = read_csv(summaries / "mechanism_compact_summary.csv")

    metric_specs = [
        ("mean_best_test_acc", "Accuracy", branch_rows, "num_branches"),
        ("mean_pairwise_distance", "Branch Diversity", mech_rows, "branches"),
        ("mean_cosine_branch", "Branch Cosine", mech_rows, "branches"),
        ("mean_cka_branch", "Branch CKA", mech_rows, "branches"),
        ("mean_grad_norm", "Mean Gradient Norm", mech_rows, "branches"),
    ]
    datasets = ["PROTEINS", "DD"]
    apply_paper_style()
    fig, axes = plt.subplots(2, 5, figsize=(18, 7.2), sharex="col")

    for row_idx, dataset in enumerate(datasets):
        for col_idx, (metric, title, source_rows, branch_key) in enumerate(metric_specs):
            ax = axes[row_idx, col_idx]
            for model in ["HorizontalRes", "MatrixRes", "MatrixResGated"]:
                subset = [r for r in source_rows if r["dataset"] == dataset and r["model"] == model]
                if branch_key == "num_branches":
                    subset = sorted(subset, key=lambda r: int(r["num_branches"]))
                    xs = [int(r["num_branches"]) for r in subset]
                else:
                    subset = sorted(subset, key=lambda r: int(r["branches"].replace("B", "")))
                    xs = [int(r["branches"].replace("B", "")) for r in subset]
                ys = [float(r[metric]) for r in subset]
                ax.plot(
                    xs,
                    ys,
                    marker=MODEL_MARKERS[model],
                    linestyle=MODEL_LINESTYLES[model],
                    linewidth=2,
                    color=MODEL_COLORS[model],
                    label=model,
                )
            if row_idx == 0:
                ax.set_title(title, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(dataset)
            ax.set_xlabel("B")
            style_axis(ax)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.94), pad=1.1)

    out_dir = ROOT / "figures" / "exp"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "fig_mechanism_branch_dynamics.pdf"
    fig.savefig(target)
    fig.savefig(target.with_suffix(".png"))
    plt.close(fig)
    print(target)


if __name__ == "__main__":
    main()
