from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


MODEL_COLORS = {
    "VerticalRes": "#6a4c93",
    "HorizontalRes": "#d1495b",
    "MatrixRes": "#2e86ab",
    "MatrixResGated": "#3a7d44",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate branch-count and sensitivity figures.")
    parser.add_argument("--version", default="LATEST")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def generate_branch_count_figure(branch_rows: list[dict[str, str]], target: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, dataset in zip(axes, ["PROTEINS", "DD"]):
        subset = [r for r in branch_rows if r["dataset"] == dataset]
        for model in ["VerticalRes", "HorizontalRes", "MatrixRes", "MatrixResGated"]:
            rows = sorted(
                [r for r in subset if r["model"] == model],
                key=lambda r: int(r["num_branches"]),
            )
            xs = [int(r["num_branches"]) for r in rows]
            ys = [float(r["mean_best_test_acc"]) for r in rows]
            errs = [float(r["std_best_test_acc"]) for r in rows]
            ax.errorbar(
                xs,
                ys,
                yerr=errs,
                marker="o",
                linewidth=2,
                capsize=3,
                color=MODEL_COLORS[model],
                label=model,
            )
        ax.set_title(dataset)
        ax.set_xlabel("Branch Count B")
        ax.grid(alpha=0.25, linestyle="--")
    axes[0].set_ylabel("Mean Best Test Accuracy")
    axes[1].legend(frameon=False, loc="lower left")
    fig.suptitle("Branch-Count Ablation on PROTEINS and DD", y=1.02)
    fig.tight_layout()
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=300, bbox_inches="tight")
    fig.savefig(target.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_sensitivity_figure(rows: list[dict[str, str]], target: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
    panel_specs = [
        ("PROTEINS", "lr", "Learning Rate"),
        ("PROTEINS", "sparse_lambda", "Sparse Lambda"),
        ("DD", "lr", "Learning Rate"),
        ("DD", "gate_init", "Gate Init"),
    ]
    for ax, (dataset, sweep_name, title) in zip(axes.ravel(), panel_specs):
        subset = [
            r for r in rows
            if r["dataset"] == dataset
            and r["model"] == "MatrixResGated"
            and r["sweep_name"] in {sweep_name, "baseline"}
        ]
        baseline = next(r for r in subset if r["sweep_name"] == "baseline")
        sweep_rows = [r for r in subset if r["sweep_name"] == sweep_name]
        sweep_rows = sorted(sweep_rows, key=lambda r: float(r["sweep_value"]))
        xs = [str(r["sweep_value"]) for r in sweep_rows]
        ys = [float(r["mean_best_test_acc"]) for r in sweep_rows]
        ax.plot(xs, ys, marker="o", linewidth=2, color=MODEL_COLORS["MatrixResGated"])
        ax.axhline(
            float(baseline["mean_best_test_acc"]),
            color="#444444",
            linestyle="--",
            linewidth=1.5,
            label="baseline",
        )
        ax.set_title(f"{dataset}: {title}")
        ax.set_xlabel(sweep_name)
        ax.set_ylabel("Best Test Accuracy")
        ax.grid(alpha=0.25, linestyle="--")
        ax.legend(frameon=False, loc="best")
    fig.suptitle("MatrixResGated Sensitivity Slices", y=1.01)
    fig.tight_layout()
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=300, bbox_inches="tight")
    fig.savefig(target.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    base = ROOT / "records" / args.version / "summaries"
    fig_dir = ROOT / "figures" / "exp"
    branch_rows = read_csv(base / "branch_ablation_summary.csv")
    sensitivity_rows = read_csv(base / "parameter_sensitivity_summary.csv")
    generate_branch_count_figure(branch_rows, fig_dir / "fig_branch_count_ablation.pdf")
    generate_sensitivity_figure(sensitivity_rows, fig_dir / "fig_matrixresgated_sensitivity.pdf")
    print(fig_dir)


if __name__ == "__main__":
    main()
