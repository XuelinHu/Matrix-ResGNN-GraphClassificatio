from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark_protocols import build_protocol
from src.experiment_catalog import MAIN_DATASETS, MAIN_MODELS
from src.experiment_paths import DEFAULT_EXPERIMENT_VERSION, log_dir, normalize_version


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check benchmark completeness for expected dataset/model/operator/fold combinations.")
    parser.add_argument("--version", default=DEFAULT_EXPERIMENT_VERSION)
    parser.add_argument("--datasets", nargs="+", default=MAIN_DATASETS)
    parser.add_argument("--models", nargs="+", default=MAIN_MODELS)
    parser.add_argument("--operators", nargs="+", default=["GCNConv"])
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    return parser.parse_args()


def latest_results(active_log_dir: Path) -> Dict[str, Path]:
    latest: Dict[str, Path] = {}
    for path in sorted(active_log_dir.glob("result_*.json")):
        key = path.stem.rsplit("__", 1)[0]
        latest[key] = path
    return latest


def main() -> None:
    args = parse_args()
    active_log_dir = log_dir(ROOT, normalize_version(args.version))
    latest = latest_results(active_log_dir)
    missing: List[str] = []
    complete: List[str] = []

    for dataset in args.datasets:
        for model in args.models:
            for operator in args.operators:
                for fold in args.folds:
                    protocol = build_protocol(dataset=dataset, model=model, operator=operator)
                    residual_mode = str(protocol.get("residual_mode", "identity"))
                    num_branches = int(protocol.get("num_branches", 3))
                    key = f"result_{dataset}__{model}__{operator}__fold{fold}__B{num_branches}__{residual_mode}"
                    if key in latest:
                        complete.append(key)
                    else:
                        missing.append(key)

    payload = {
        "expected": len(args.datasets) * len(args.models) * len(args.operators) * len(args.folds),
        "complete": len(complete),
        "missing": len(missing),
        "missing_keys": missing,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
