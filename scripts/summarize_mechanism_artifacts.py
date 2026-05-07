from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment_paths import DEFAULT_EXPERIMENT_VERSION, analysis_dir, normalize_version, record_dir


ARTIFACT_PREFIXES = [
    "branch_diversity",
    "residual_stats",
    "gradient_by_layer",
    "cosine_branch",
    "cosine_depth",
    "cka_branch",
    "cka_depth",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize saved mechanism-analysis artifacts into compact CSV tables.")
    parser.add_argument("--version", default=DEFAULT_EXPERIMENT_VERSION)
    return parser.parse_args()


def parse_stem(path: Path) -> Dict[str, str]:
    stem = path.stem
    artifact = ""
    suffix = stem
    for prefix in ARTIFACT_PREFIXES:
        token = f"{prefix}_"
        if stem.startswith(token):
            artifact = prefix
            suffix = stem[len(token):]
            break
    parts = suffix.split("__")
    return {
        "artifact": artifact or stem,
        "dataset": parts[0] if len(parts) > 0 else "",
        "model": parts[1] if len(parts) > 1 else "",
        "operator": parts[2] if len(parts) > 2 else "",
        "fold": parts[3].replace("fold", "") if len(parts) > 3 else "",
        "branches": parts[4] if len(parts) > 4 else "",
        "residual_mode": parts[5] if len(parts) > 5 else "",
        "timestamp": parts[6] if len(parts) > 6 else "",
    }


def write_csv(rows: List[Dict[str, object]], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        target.write_text("", encoding="utf-8")
        return
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_branch_diversity(active_analysis_dir: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for path in sorted(active_analysis_dir.glob("branch_diversity_*.json")):
        meta = parse_stem(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                **meta,
                "mean_pairwise_distance": payload["mean_pairwise_distance"],
                "max_pairwise_distance": payload["max_pairwise_distance"],
                "pair_count": len(payload["pairwise_distances"]),
            }
        )
    return rows


def summarize_residual_stats(active_analysis_dir: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for path in sorted(active_analysis_dir.glob("residual_stats_*.json")):
        meta = parse_stem(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        stats = payload.get("residual_stats", {})
        if not stats:
            continue
        residual_count = sum(float(item.get("residual_count", 0.0)) for item in stats.values())
        residual_norm_pre = sum(float(item.get("residual_norm_pre", 0.0)) for item in stats.values())
        residual_norm_post = sum(float(item.get("residual_norm_post", 0.0)) for item in stats.values())
        active_ratio = sum(float(item.get("active_ratio", 0.0)) for item in stats.values()) / max(len(stats), 1)
        gate_value = sum(float(item.get("gate", 0.0)) for item in stats.values()) / max(len(stats), 1)
        rows.append(
            {
                **meta,
                "residual_sites": len(stats),
                "residual_count_total": residual_count,
                "residual_norm_pre_total": residual_norm_pre,
                "residual_norm_post_total": residual_norm_post,
                "active_ratio_mean": active_ratio,
                "gate_mean": gate_value,
            }
        )
    return rows


def summarize_gradient(active_analysis_dir: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for path in sorted(active_analysis_dir.glob("gradient_by_layer_*.csv")):
        meta = parse_stem(path)
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            values = [float(row["grad_l2_norm"]) for row in reader]
        if not values:
            continue
        rows.append(
            {
                **meta,
                "grad_rows": len(values),
                "mean_grad_norm": sum(values) / len(values),
                "grad_max": max(values),
                "grad_min": min(values),
            }
        )
    return rows


def summarize_similarity(active_analysis_dir: Path, prefix: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for path in sorted(active_analysis_dir.glob(f"{prefix}_*.csv")):
        meta = parse_stem(path)
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = list(csv.reader(handle))
        values: List[float] = []
        for row in reader[1:]:
            for value in row[1:]:
                try:
                    values.append(float(value))
                except ValueError:
                    pass
        if not values:
            continue
        rows.append(
            {
                **meta,
                "matrix_name": prefix,
                f"mean_{prefix}": sum(values) / len(values),
                f"max_{prefix}": max(values),
                f"min_{prefix}": min(values),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    version = normalize_version(args.version)
    active_analysis_dir = analysis_dir(ROOT, version)
    out_dir = record_dir(ROOT, version) / "mechanism_summaries"

    write_csv(summarize_branch_diversity(active_analysis_dir), out_dir / "branch_diversity_summary.csv")
    write_csv(summarize_residual_stats(active_analysis_dir), out_dir / "residual_stats_summary.csv")
    write_csv(summarize_gradient(active_analysis_dir), out_dir / "gradient_summary.csv")
    write_csv(summarize_similarity(active_analysis_dir, "cosine_branch"), out_dir / "cosine_branch_summary.csv")
    write_csv(summarize_similarity(active_analysis_dir, "cosine_depth"), out_dir / "cosine_depth_summary.csv")
    write_csv(summarize_similarity(active_analysis_dir, "cka_branch"), out_dir / "cka_branch_summary.csv")
    write_csv(summarize_similarity(active_analysis_dir, "cka_depth"), out_dir / "cka_depth_summary.csv")
    print(out_dir)


if __name__ == "__main__":
    main()
