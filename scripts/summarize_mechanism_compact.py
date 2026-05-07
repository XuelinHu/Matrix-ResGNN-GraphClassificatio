from __future__ import annotations

import argparse
import csv
import statistics as st
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment_paths import DEFAULT_EXPERIMENT_VERSION, normalize_version, record_dir


BASE_KEYS = ["dataset", "model", "operator", "fold", "branches", "residual_mode"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate mechanism summaries into compact tables.")
    parser.add_argument("--version", default=DEFAULT_EXPERIMENT_VERSION)
    parser.add_argument("--datasets", nargs="+", default=["PROTEINS", "DD", "ENZYMES"])
    parser.add_argument("--models", nargs="+", default=["HorizontalRes", "MatrixRes", "MatrixResGated", "VerticalRes", "Plain"])
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def latest_by_signature(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    latest: Dict[Tuple[str, ...], Dict[str, str]] = {}
    for row in rows:
        key = tuple(row.get(k, "") for k in BASE_KEYS)
        prev = latest.get(key)
        if prev is None or row.get("timestamp", "") >= prev.get("timestamp", ""):
            latest[key] = row
    return list(latest.values())


def index_metric(rows: List[Dict[str, str]], metric_columns: List[str]) -> Dict[Tuple[str, ...], Dict[str, float]]:
    indexed: Dict[Tuple[str, ...], Dict[str, float]] = {}
    for row in latest_by_signature(rows):
        key = tuple(row.get(k, "") for k in BASE_KEYS)
        payload: Dict[str, float] = {}
        for col in metric_columns:
            value = row.get(col, "")
            if value == "":
                continue
            payload[col] = float(value)
        indexed[key] = payload
    return indexed


def safe_mean(values: List[float]) -> float:
    return st.mean(values) if values else 0.0


def main() -> None:
    args = parse_args()
    version = normalize_version(args.version)
    base = record_dir(ROOT, version) / "mechanism_summaries"
    out_dir = record_dir(ROOT, version) / "summaries"

    branch_index = index_metric(read_csv(base / "branch_diversity_summary.csv"), ["mean_pairwise_distance", "max_pairwise_distance"])
    residual_index = index_metric(read_csv(base / "residual_stats_summary.csv"), ["residual_count_total", "residual_norm_pre_total", "residual_norm_post_total", "active_ratio_mean", "gate_mean"])
    gradient_index = index_metric(read_csv(base / "gradient_summary.csv"), ["mean_grad_norm", "grad_max", "grad_min"])
    cosine_branch_index = index_metric(read_csv(base / "cosine_branch_summary.csv"), ["mean_cosine_branch"])
    cosine_depth_index = index_metric(read_csv(base / "cosine_depth_summary.csv"), ["mean_cosine_depth"])
    cka_branch_index = index_metric(read_csv(base / "cka_branch_summary.csv"), ["mean_cka_branch"])
    cka_depth_index = index_metric(read_csv(base / "cka_depth_summary.csv"), ["mean_cka_depth"])

    all_keys = set()
    for index in [
        branch_index,
        residual_index,
        gradient_index,
        cosine_branch_index,
        cosine_depth_index,
        cka_branch_index,
        cka_depth_index,
    ]:
        all_keys.update(index.keys())

    fold_rows: List[Dict[str, object]] = []
    for key in sorted(all_keys):
        row = dict(zip(BASE_KEYS, key))
        if row["dataset"] not in args.datasets or row["model"] not in args.models:
            continue
        for index in [
            branch_index,
            residual_index,
            gradient_index,
            cosine_branch_index,
            cosine_depth_index,
            cka_branch_index,
            cka_depth_index,
        ]:
            row.update(index.get(key, {}))
        fold_rows.append(row)

    grouped: Dict[Tuple[str, str, str, str], List[Dict[str, object]]] = {}
    for row in fold_rows:
        gkey = (str(row["dataset"]), str(row["model"]), str(row["branches"]), str(row["residual_mode"]))
        grouped.setdefault(gkey, []).append(row)

    summary_rows: List[Dict[str, object]] = []
    metrics = [
        "mean_pairwise_distance",
        "max_pairwise_distance",
        "residual_count_total",
        "residual_norm_pre_total",
        "residual_norm_post_total",
        "active_ratio_mean",
        "gate_mean",
        "mean_grad_norm",
        "grad_max",
        "grad_min",
        "mean_cosine_branch",
        "mean_cosine_depth",
        "mean_cka_branch",
        "mean_cka_depth",
    ]
    for (dataset, model, branches, residual_mode), rows in sorted(grouped.items()):
        out: Dict[str, object] = {
            "dataset": dataset,
            "model": model,
            "branches": branches,
            "residual_mode": residual_mode,
            "folds": len(rows),
        }
        for metric in metrics:
            vals = [float(r[metric]) for r in rows if metric in r]
            if vals:
                out[metric] = safe_mean(vals)
        summary_rows.append(out)

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in [
        ("mechanism_compact_fold_rows.csv", fold_rows),
        ("mechanism_compact_summary.csv", summary_rows),
    ]:
        target = out_dir / name
        if not rows:
            target.write_text("", encoding="utf-8")
            continue
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(out_dir / "mechanism_compact_summary.csv")


if __name__ == "__main__":
    main()
