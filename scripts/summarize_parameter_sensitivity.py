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

from src.experiment_paths import DEFAULT_EXPERIMENT_VERSION, log_dir, normalize_version, record_dir


BASELINE_DEFAULTS = {
    "lr": 0.003,
    "drop": 0.3,
    "dim": 64,
    "sparse_lambda": 0.05,
    "gate_init": 0.8,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize first-batch parameter sensitivity runs.")
    parser.add_argument("--version", default=DEFAULT_EXPERIMENT_VERSION)
    parser.add_argument("--datasets", nargs="+", default=["PROTEINS", "DD"])
    parser.add_argument("--models", nargs="+", default=["MatrixRes", "MatrixResGated"])
    parser.add_argument("--operator", default="GCNConv")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--num_branches", type=int, default=3)
    parser.add_argument("--ep", type=int, default=120)
    return parser.parse_args()


def latest_results(active_log_dir: Path) -> Dict[str, Path]:
    latest: Dict[str, Path] = {}
    for path in sorted(active_log_dir.glob("result_*.json")):
        key = path.stem.rsplit("__", 1)[0]
        latest[key] = path
    return latest


def latest_results_by_config(
    active_log_dir: Path,
    datasets: List[str],
    models: List[str],
    operator: str,
    fold: int,
    num_branches: int,
    ep: int,
) -> Dict[str, Dict[str, object]]:
    latest: Dict[str, Dict[str, object]] = {}
    for path in sorted(active_log_dir.glob("result_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = payload["config"]
        if (
            config["dataset"] not in datasets
            or config["model"] not in models
            or config["operator"] != operator
            or int(config["fold"]) != fold
            or int(config["num_branches"]) != num_branches
            or int(config["ep"]) != ep
        ):
            continue
        key = json.dumps(
            {
                "dataset": config["dataset"],
                "model": config["model"],
                "operator": config["operator"],
                "fold": int(config["fold"]),
                "num_branches": int(config["num_branches"]),
                "ep": int(config["ep"]),
                "residual_mode": config.get("residual_mode"),
                "lr": config.get("lr"),
                "drop": config.get("drop"),
                "dim": config.get("dim"),
                "sparse_lambda": config.get("sparse_lambda"),
                "gate_init": config.get("gate_init"),
            },
            sort_keys=True,
        )
        latest[key] = payload
    return latest


def infer_sweep(config: Dict[str, object]) -> tuple[str, object]:
    candidates: List[tuple[str, object]] = []
    for key, default in BASELINE_DEFAULTS.items():
        if key not in config:
            continue
        value = config[key]
        if value != default:
            candidates.append((key, value))
    if not candidates:
        return ("baseline", "default")
    if len(candidates) == 1:
        return candidates[0]
    label = "+".join(name for name, _ in candidates)
    value = "|".join(str(v) for _, v in candidates)
    return (label, value)


def load_rows(
    active_log_dir: Path,
    datasets: List[str],
    models: List[str],
    operator: str,
    fold: int,
    num_branches: int,
    ep: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for payload in latest_results_by_config(
        active_log_dir,
        datasets,
        models,
        operator,
        fold,
        num_branches,
        ep,
    ).values():
        config = payload["config"]
        sweep_name, sweep_value = infer_sweep(config)
        rows.append(
            {
                "dataset": config["dataset"],
                "model": config["model"],
                "operator": config["operator"],
                "fold": int(config["fold"]),
                "sweep_name": sweep_name,
                "sweep_value": sweep_value,
                "best_test_acc": float(payload["best_test_acc"]),
                "test_loss": float(payload["test_loss"]),
                "runtime_seconds": float(payload["runtime_seconds"]),
                "best_epoch": int(payload["best_epoch"]) + 1,
                "total_params": int(payload["parameter_stats"]["total_params"]),
            }
        )
    return rows


def summarize(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    keys = sorted(set((r["dataset"], r["model"], r["sweep_name"], str(r["sweep_value"])) for r in rows))
    summary_rows: List[Dict[str, object]] = []
    for dataset, model, sweep_name, sweep_value in keys:
        subset = [
            r for r in rows
            if r["dataset"] == dataset and r["model"] == model and r["sweep_name"] == sweep_name and str(r["sweep_value"]) == sweep_value
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
                "sweep_name": sweep_name,
                "sweep_value": sweep_value,
                "runs": len(subset),
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
    rows = load_rows(
        log_dir(ROOT, version),
        args.datasets,
        args.models,
        args.operator,
        args.fold,
        args.num_branches,
        args.ep,
    )
    summary_rows = summarize(rows)
    out_dir = record_dir(ROOT, version) / "summaries"
    write_csv(rows, out_dir / "parameter_sensitivity_fold_rows.csv")
    write_csv(summary_rows, out_dir / "parameter_sensitivity_summary.csv")
    print(out_dir / "parameter_sensitivity_summary.csv")


if __name__ == "__main__":
    main()
