"""批量运行主 benchmark 的数据集、模型、算子和折数组合。"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import subprocess
import sys
from pathlib import Path

# 仓库根目录：用于把脚本中的相对路径统一定位到项目根路径。
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark_protocols import build_protocol
from src.experiment_catalog import ACTIVE_OPERATORS, MAIN_DATASETS, MAIN_MODELS, SUPPLEMENTARY_DATASETS
from src.experiment_paths import DEFAULT_EXPERIMENT_VERSION, ensure_version_manifest


def parse_args() -> argparse.Namespace:
    """解析命令行参数，返回当前脚本需要的实验配置。"""
    parser = argparse.ArgumentParser(description="Scaffold for the main benchmark runner.")
    parser.add_argument("--dataset_group", choices=["main", "supplementary", "all"], default="main")
    parser.add_argument("--models", nargs="+", default=MAIN_MODELS)
    parser.add_argument("--operators", nargs="+", default=ACTIVE_OPERATORS)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--version", default=DEFAULT_EXPERIMENT_VERSION)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--override_ep", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """脚本主入口，串联参数解析、数据读取、处理和结果写出。"""
    root = ROOT
    ensure_version_manifest(root)
    args = parse_args()
    if args.dataset_group == "main":
        datasets = MAIN_DATASETS
    elif args.dataset_group == "supplementary":
        datasets = SUPPLEMENTARY_DATASETS
    else:
        datasets = [*MAIN_DATASETS, *SUPPLEMENTARY_DATASETS]

    jobs = []
    for dataset in datasets:
        for model in args.models:
            for operator in args.operators:
                for fold in args.folds:
                    cmd = [
                        sys.executable,
                        str(root / "scripts" / "run_single.py"),
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
                    protocol = build_protocol(dataset=dataset, model=model, operator=operator)
                    if args.override_ep is not None:
                        protocol["ep"] = args.override_ep
                    for key, value in protocol.items():
                        cmd.extend([f"--{key}", str(value)])
                    jobs.append(cmd)

    def run_job(cmd: list[str]) -> tuple[int, str]:
        """run_job 函数的职责说明。"""
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=root)
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
