from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment_paths import DEFAULT_EXPERIMENT_VERSION, log_dir, normalize_version, record_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize branch-count ablation runs.")
    parser.add_argument("--version", default=DEFAULT_EXPERIMENT_VERSION)
    parser.add_argument("--datasets", nargs="+", default=["PROTEINS", "DD"])
    parser.add_argument("--models", nargs="+", default=["VerticalRes", "HorizontalRes", "MatrixRes", "MatrixResGated"])
    parser.add_argument("--operator", default="GCNConv")
    return parser.parse_args()


def latest_results(active_log_dir: Path) -> Dict[str, Path]:
    latest: Dict[str, Path] = {}
    for path in sorted(active_log_dir.glob("result_*.json")):
        key = path.stem.rsplit("__", 1)[0]
        latest[key] = path
    return latest


def load_rows(active_log_dir: Path, datasets: List[str], models: List[str], operator: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for path in latest_results(active_log_dir).values():
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = payload["config"]
        if config["dataset"] not in datasets or config["model"] not in models or config["operator"] != operator:
            continue
        rows.append(
            {
                "dataset": config["dataset"],
                "model": config["model"],
                "operator": config["operator"],
                "fold": int(config["fold"]),
                "num_branches": int(config["num_branches"]),
                "residual_mode": config["residual_mode"],
                "best_epoch": int(payload["best_epoch"]) + 1,
                "best_test_acc": float(payload["best_test_acc"]),
                "test_loss": float(payload["test_loss"]),
                "runtime_seconds": float(payload["runtime_seconds"]),
                "total_params": int(payload["parameter_stats"]["total_params"]),
            }
        )
    return rows


def summarize(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    keys = sorted(set((r["dataset"], r["model"], r["num_branches"], r["residual_mode"]) for r in rows))
    summary_rows: List[Dict[str, object]] = []
    for dataset, model, num_branches, residual_mode in keys:
        subset = [
            r for r in rows
            if r["dataset"] == dataset and r["model"] == model and r["num_branches"] == num_branches and r["residual_mode"] == residual_mode
        ]
        if not subset:
            continue
        accs = [float(r["best_test_acc"]) for r in subset]
        losses = [float(r["test_loss"]) for r in subset]
        runtimes = [float(r["runtime_seconds"]) for r in subset]
        epochs = [int(r["best_epoch"]) for r in subset]
        summary_rows.append(
            {
                "dataset": dataset,
                "model": model,
                "num_branches": num_branches,
                "residual_mode": residual_mode,
                "folds": len(subset),
                "mean_best_test_acc": st.mean(accs),
                "std_best_test_acc": st.pstdev(accs) if len(accs) > 1 else 0.0,
                "mean_test_loss": st.mean(losses),
                "mean_runtime_seconds": st.mean(runtimes),
                "mean_best_epoch": st.mean(epochs),
                "total_params": int(subset[0]["total_params"]),
            }
        )
    return summary_rows


def write_csv(rows: List[Dict[str, object]], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        target.write_text("", encoding="utf-8")
        return
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    version = normalize_version(args.version)
    rows = load_rows(log_dir(ROOT, version), args.datasets, args.models, args.operator)
    summary_rows = summarize(rows)
    out_dir = record_dir(ROOT, version) / "summaries"
    write_csv(rows, out_dir / "branch_ablation_fold_rows.csv")
    write_csv(summary_rows, out_dir / "branch_ablation_summary.csv")
    print(out_dir / "branch_ablation_summary.csv")


if __name__ == "__main__":
    main()
