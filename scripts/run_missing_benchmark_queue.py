"""发现缺失 benchmark 任务并可按队列批量补跑。"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

# 仓库根目录：用于把脚本中的相对路径统一定位到项目根路径。
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark_protocols import build_protocol
from src.experiment_catalog import ALL_ACTIVE_DATASETS, ACTIVE_OPERATORS, MAIN_MODELS
from src.experiment_paths import DEFAULT_EXPERIMENT_VERSION, ensure_version_manifest, log_dir, normalize_version, record_dir


def parse_args() -> argparse.Namespace:
    """解析命令行参数，返回当前脚本需要的实验配置。"""
    parser = argparse.ArgumentParser(description="Run missing benchmark jobs with resumable progress tracking.")
    parser.add_argument("--version", default=DEFAULT_EXPERIMENT_VERSION)
    parser.add_argument("--datasets", nargs="+", default=ALL_ACTIVE_DATASETS)
    parser.add_argument("--models", nargs="+", default=MAIN_MODELS)
    parser.add_argument("--operators", nargs="+", default=ACTIVE_OPERATORS)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--queue_name", default="full_benchmark_6datasets_4ops")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def latest_results(active_log_dir: Path) -> Dict[str, Path]:
    """扫描日志目录，按实验配置键保留最新的 result JSON 文件路径。"""
    latest: Dict[str, Path] = {}
    for path in sorted(active_log_dir.glob("result_*.json")):
        key = path.stem.rsplit("__", 1)[0]
        latest[key] = path
    return latest


def expected_key(dataset: str, model: str, operator: str, fold: int) -> str:
    """根据数据集、模型、算子和折号构造预期日志键。"""
    protocol = build_protocol(dataset=dataset, model=model, operator=operator)
    residual_mode = str(protocol.get("residual_mode", "identity"))
    num_branches = int(protocol.get("num_branches", 3))
    return f"result_{dataset}__{model}__{operator}__fold{fold}__B{num_branches}__{residual_mode}"


def build_command(dataset: str, model: str, operator: str, fold: int, version: str) -> List[str]:
    """把任务配置转换为可执行的 run_single 命令。"""
    protocol = build_protocol(dataset=dataset, model=model, operator=operator)
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
        version,
    ]
    for key, value in protocol.items():
        cmd.extend([f"--{key}", str(value)])
    return cmd


def build_jobs(args: argparse.Namespace, completed: Dict[str, Path]) -> List[Dict[str, object]]:
    """根据实验网格构造待运行任务列表。"""
    jobs: List[Dict[str, object]] = []
    for dataset in args.datasets:
        for model in args.models:
            for operator in args.operators:
                for fold in args.folds:
                    key = expected_key(dataset, model, operator, fold)
                    if key in completed:
                        continue
                    jobs.append(
                        {
                            "key": key,
                            "dataset": dataset,
                            "model": model,
                            "operator": operator,
                            "fold": fold,
                            "cmd": build_command(dataset, model, operator, fold, normalize_version(args.version)),
                        }
                    )
    return jobs


def write_json(path: Path, payload: Dict[str, object]) -> None:
    """把任务状态或队列信息写入 JSON 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, payload: Dict[str, object]) -> None:
    """向 JSONL 文件追加一条任务记录，便于持续跟踪队列进度。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run_one(job: Dict[str, object]) -> Dict[str, object]:
    """执行单个补跑任务，并记录任务状态。"""
    start = time.time()
    proc = subprocess.run(job["cmd"], capture_output=True, text=True, cwd=ROOT)
    tail_lines = []
    if proc.stdout:
        tail_lines.extend(proc.stdout.strip().splitlines()[-5:])
    if proc.returncode != 0 and proc.stderr:
        tail_lines.extend(proc.stderr.strip().splitlines()[-10:])
    return {
        "key": job["key"],
        "dataset": job["dataset"],
        "model": job["model"],
        "operator": job["operator"],
        "fold": job["fold"],
        "returncode": proc.returncode,
        "runtime_seconds": round(time.time() - start, 3),
        "tail": "\n".join(tail_lines),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }


def summarize_scope(args: argparse.Namespace, completed_before: int, pending: int) -> Dict[str, object]:
    """统计当前队列范围内已完成和缺失的实验组合。"""
    expected = len(args.datasets) * len(args.models) * len(args.operators) * len(args.folds)
    return {
        "queue_name": args.queue_name,
        "version": normalize_version(args.version),
        "datasets": args.datasets,
        "models": args.models,
        "operators": args.operators,
        "folds": args.folds,
        "expected": expected,
        "completed_before": completed_before,
        "pending_at_start": pending,
        "max_workers": args.max_workers,
        "pid": os.getpid(),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }


def main() -> None:
    """脚本主入口，串联参数解析、数据读取、处理和结果写出。"""
    args = parse_args()
    ensure_version_manifest(ROOT)
    version = normalize_version(args.version)
    queue_dir = record_dir(ROOT, version) / "queue"
    status_path = queue_dir / f"{args.queue_name}_status.json"
    events_path = queue_dir / f"{args.queue_name}_events.jsonl"
    plan_path = queue_dir / f"{args.queue_name}_plan.json"

    completed = latest_results(log_dir(ROOT, version))
    jobs = build_jobs(args, completed)
    status = summarize_scope(args, completed_before=len(completed), pending=len(jobs))
    write_json(plan_path, {"scope": status, "jobs": [{k: v for k, v in job.items() if k != "cmd"} for job in jobs]})
    write_json(status_path, {**status, "completed_now": 0, "failed": 0, "remaining": len(jobs), "state": "dry_run" if args.dry_run else "running"})

    if args.dry_run:
        print(json.dumps({"plan": str(plan_path), "status": str(status_path), "pending": len(jobs)}, indent=2))
        return

    completed_now = 0
    failed = 0
    with cf.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(run_one, job) for job in jobs]
        for future in cf.as_completed(futures):
            result = future.result()
            ok = result["returncode"] == 0
            completed_now += int(ok)
            failed += int(not ok)
            append_jsonl(events_path, {"event": "job_finished", **result})
            write_json(
                status_path,
                {
                    **status,
                    "completed_now": completed_now,
                    "failed": failed,
                    "remaining": len(jobs) - completed_now - failed,
                    "state": "running" if failed == 0 else "failed",
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                    "events_path": str(events_path.relative_to(ROOT)),
                    "plan_path": str(plan_path.relative_to(ROOT)),
                },
            )
            if not ok:
                raise SystemExit(result["returncode"])

    final_completed = latest_results(log_dir(ROOT, version))
    write_json(
        status_path,
        {
            **status,
            "completed_now": completed_now,
            "failed": failed,
            "remaining": 0,
            "complete_in_logs": len(final_completed),
            "state": "completed",
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "events_path": str(events_path.relative_to(ROOT)),
            "plan_path": str(plan_path.relative_to(ROOT)),
        },
    )


if __name__ == "__main__":
    main()
