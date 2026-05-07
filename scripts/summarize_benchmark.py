from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment_catalog import ALL_ACTIVE_DATASETS, MAIN_MODELS
from src.experiment_paths import DEFAULT_EXPERIMENT_VERSION, log_dir, normalize_version, record_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize benchmark result JSON files into stable CSV tables.")
    parser.add_argument("--version", default=DEFAULT_EXPERIMENT_VERSION)
    return parser.parse_args()


def latest_results(active_log_dir: Path) -> Dict[str, Path]:
    latest: Dict[str, Path] = {}
    for path in sorted(active_log_dir.glob("result_*.json")):
        stem = path.stem
        if "__" not in stem:
            continue
        key = stem.rsplit("__", 1)[0]
        latest[key] = path
    return latest


def load_rows(active_log_dir: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for path in latest_results(active_log_dir).values():
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = payload["config"]
        rows.append(
            {
                "dataset": config["dataset"],
                "model": config["model"],
                "operator": config["operator"],
                "fold": int(config["fold"]),
                "num_branches": int(config["num_branches"]),
                "residual_mode": config["residual_mode"],
                "best_epoch": int(payload["best_epoch"]) + 1,
                "best_val_acc": float(payload["best_val_acc"]),
                "best_test_acc": float(payload["best_test_acc"]),
                "test_loss": float(payload["test_loss"]),
                "runtime_seconds": float(payload["runtime_seconds"]),
                "total_params": int(payload["parameter_stats"]["total_params"]),
            }
        )
    return rows


def summarize(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    summary_rows: List[Dict[str, object]] = []
    for dataset in ALL_ACTIVE_DATASETS:
        for model in MAIN_MODELS:
            for operator in ["GCNConv", "GATConv", "SAGEConv", "GINConv"]:
                subset = [
                    row
                    for row in rows
                    if row["dataset"] == dataset and row["model"] == model and row["operator"] == operator
                ]
                if not subset:
                    continue
                accs = [float(row["best_test_acc"]) for row in subset]
                losses = [float(row["test_loss"]) for row in subset]
                epochs = [int(row["best_epoch"]) for row in subset]
                runtimes = [float(row["runtime_seconds"]) for row in subset]
                summary_rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "operator": operator,
                        "folds": len(subset),
                        "mean_best_test_acc": st.mean(accs),
                        "std_best_test_acc": st.pstdev(accs) if len(accs) > 1 else 0.0,
                        "mean_test_loss": st.mean(losses),
                        "mean_best_epoch": st.mean(epochs),
                        "mean_runtime_seconds": st.mean(runtimes),
                        "total_params": int(subset[0]["total_params"]),
                        "num_branches": int(subset[0]["num_branches"]),
                        "residual_mode": subset[0]["residual_mode"],
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
    rows = load_rows(log_dir(ROOT, version))
    summary_rows = summarize(rows)
    out_dir = record_dir(ROOT, version) / "summaries"
    write_csv(rows, out_dir / "benchmark_fold_rows.csv")
    write_csv(summary_rows, out_dir / "benchmark_summary.csv")
    print(out_dir / "benchmark_summary.csv")


if __name__ == "__main__":
    main()
