from __future__ import annotations

import argparse
import concurrent.futures as cf
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark_protocols import build_protocol
from src.experiment_paths import DEFAULT_EXPERIMENT_VERSION, ensure_version_manifest


TUNED_CANDIDATES = [
    {
        "label": "PROTEINS_sparse_lambda_0.02",
        "dataset": "PROTEINS",
        "model": "MatrixResGated",
        "operator": "GCNConv",
        "overrides": {"sparse_lambda": 0.02},
    },
    {
        "label": "DD_lr_0.001",
        "dataset": "DD",
        "model": "MatrixResGated",
        "operator": "GCNConv",
        "overrides": {"lr": 0.001},
    },
    {
        "label": "DD_dim_32",
        "dataset": "DD",
        "model": "MatrixResGated",
        "operator": "GCNConv",
        "overrides": {"dim": 32},
    },
    {
        "label": "DD_gate_init_0.2",
        "dataset": "DD",
        "model": "MatrixResGated",
        "operator": "GCNConv",
        "overrides": {"gate_init": 0.2},
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run five-fold confirmations for selected tuned MatrixResGated candidates.")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--version", default=DEFAULT_EXPERIMENT_VERSION)
    parser.add_argument("--max_workers", type=int, default=4)
    return parser.parse_args()


def build_jobs(args: argparse.Namespace) -> list[list[str]]:
    jobs: list[list[str]] = []
    for candidate in TUNED_CANDIDATES:
        for fold in args.folds:
            dataset = str(candidate["dataset"])
            model = str(candidate["model"])
            operator = str(candidate["operator"])
            protocol = build_protocol(dataset=dataset, model=model, operator=operator)
            protocol.update(candidate["overrides"])  # type: ignore[arg-type]
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "run_single.py"),
                "--dataset",
                dataset,
                "--model",
                model,
                "--operator",
                operator,
                "--fold",
                str(fold),
                "--version",
                args.version,
            ]
            for key, value in protocol.items():
                cmd.extend([f"--{key}", str(value)])
            jobs.append(cmd)
    return jobs


def main() -> None:
    args = parse_args()
    ensure_version_manifest(ROOT)
    jobs = build_jobs(args)

    def run_job(cmd: list[str]) -> tuple[int, str]:
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
        tail = "\n".join(proc.stdout.strip().splitlines()[-4:]) if proc.stdout else ""
        if proc.returncode != 0:
            tail = f"{tail}\n{proc.stderr}".strip()
        return proc.returncode, tail

    with cf.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(run_job, cmd) for cmd in jobs]
        for index, future in enumerate(cf.as_completed(futures), 1):
            rc, tail = future.result()
            status = "OK" if rc == 0 else "FAIL"
            print(f"[{index:03d}/{len(jobs)}] {status}")
            if tail:
                print(tail, flush=True)
            if rc != 0:
                raise SystemExit(rc)


if __name__ == "__main__":
    main()

