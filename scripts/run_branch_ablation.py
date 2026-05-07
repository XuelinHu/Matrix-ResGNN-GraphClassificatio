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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run branch-count ablations for multi-branch residual models.")
    parser.add_argument("--dataset", default="PROTEINS")
    parser.add_argument("--models", nargs="+", default=["VerticalRes", "HorizontalRes", "MatrixRes", "MatrixResGated"])
    parser.add_argument("--operator", default="GCNConv")
    parser.add_argument("--branches", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--version", default=DEFAULT_EXPERIMENT_VERSION)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--override_ep", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_version_manifest(ROOT)
    jobs: list[list[str]] = []
    for model in args.models:
        for num_branches in args.branches:
            for fold in args.folds:
                cmd = [
                    sys.executable,
                    str(ROOT / "scripts" / "run_single.py"),
                    "--dataset",
                    args.dataset,
                    "--model",
                    model,
                    "--operator",
                    args.operator,
                    "--num_branches",
                    str(num_branches),
                    "--fold",
                    str(fold),
                    "--version",
                    args.version,
                ]
                protocol = build_protocol(dataset=args.dataset, model=model, operator=args.operator)
                if args.override_ep is not None:
                    protocol["ep"] = args.override_ep
                for key, value in protocol.items():
                    if key == "num_branches":
                        continue
                    cmd.extend([f"--{key}", str(value)])
                jobs.append(cmd)

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
