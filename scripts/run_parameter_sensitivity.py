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


SWEEP_VALUES = {
    "lr": [0.001, 0.003, 0.005, 0.01],
    "drop": [0.1, 0.3, 0.5, 0.7],
    "dim": [32, 64, 128],
    "sparse_lambda": [0.01, 0.02, 0.05, 0.1],
    "gate_init": [0.2, 0.5, 0.8, 0.95],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run first-batch parameter sensitivity scans.")
    parser.add_argument("--datasets", nargs="+", default=["PROTEINS", "DD"])
    parser.add_argument("--models", nargs="+", default=["MatrixRes", "MatrixResGated"])
    parser.add_argument("--operator", default="GCNConv")
    parser.add_argument("--folds", nargs="+", type=int, default=[0])
    parser.add_argument("--sweeps", nargs="+", default=["lr", "drop", "dim", "sparse_lambda", "gate_init"])
    parser.add_argument("--version", default=DEFAULT_EXPERIMENT_VERSION)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--override_ep", type=int, default=120)
    return parser.parse_args()


def model_supports_sweep(model: str, sweep_name: str) -> bool:
    if sweep_name in {"sparse_lambda", "gate_init"}:
        return model == "MatrixResGated"
    return True


def build_jobs(args: argparse.Namespace) -> list[list[str]]:
    jobs: list[list[str]] = []
    for dataset in args.datasets:
        for model in args.models:
            for fold in args.folds:
                base = build_protocol(dataset=dataset, model=model, operator=args.operator)
                if args.override_ep is not None:
                    base["ep"] = args.override_ep
                for sweep_name in args.sweeps:
                    if not model_supports_sweep(model, sweep_name):
                        continue
                    for sweep_value in SWEEP_VALUES[sweep_name]:
                        cmd = [
                            sys.executable,
                            str(ROOT / "scripts" / "run_single.py"),
                            "--dataset",
                            dataset,
                            "--model",
                            model,
                            "--operator",
                            args.operator,
                            "--fold",
                            str(fold),
                            "--version",
                            args.version,
                        ]
                        protocol = dict(base)
                        protocol[sweep_name] = sweep_value
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
