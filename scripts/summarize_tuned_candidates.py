"""汇总已确认调参候选的五折结果。"""
from __future__ import annotations

import csv
import json
import statistics as st
import sys
from pathlib import Path
from typing import Dict, List

# 仓库根目录：用于把脚本中的相对路径统一定位到项目根路径。
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_tuned_candidates import TUNED_CANDIDATES
from src.experiment_paths import DEFAULT_EXPERIMENT_VERSION, log_dir, normalize_version, record_dir


def matches_candidate(config: Dict[str, object], candidate: Dict[str, object]) -> bool:
    """判断日志配置是否命中预定义的调参候选。"""
    if config["dataset"] != candidate["dataset"]:
        return False
    if config["model"] != candidate["model"]:
        return False
    if config["operator"] != candidate["operator"]:
        return False
    for key, value in candidate["overrides"].items():  # type: ignore[union-attr]
        if config.get(key) != value:
            return False
    return True


def load_rows(version: str) -> List[Dict[str, object]]:
    """读取输入结果文件，并转换成后续汇总需要的结构化行。"""
    latest: Dict[tuple[str, int], Dict[str, object]] = {}
    for path in sorted(log_dir(ROOT, normalize_version(version)).glob("result_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = payload["config"]
        for candidate in TUNED_CANDIDATES:
            if not matches_candidate(config, candidate):
                continue
            latest[(str(candidate["label"]), int(config["fold"]))] = {
                "candidate": candidate["label"],
                "dataset": config["dataset"],
                "model": config["model"],
                "operator": config["operator"],
                "fold": int(config["fold"]),
                "best_epoch": int(payload["best_epoch"]) + 1,
                "best_test_acc": float(payload["best_test_acc"]),
                "test_loss": float(payload["test_loss"]),
                "runtime_seconds": float(payload["runtime_seconds"]),
                "total_params": int(payload["parameter_stats"]["total_params"]),
            }
    return list(latest.values())


def summarize(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """对逐折实验记录做聚合，生成均值、标准差和运行统计。"""
    summary: List[Dict[str, object]] = []
    keys = sorted(set(str(row["candidate"]) for row in rows))
    for key in keys:
        subset = [row for row in rows if row["candidate"] == key]
        accs = [float(row["best_test_acc"]) for row in subset]
        losses = [float(row["test_loss"]) for row in subset]
        runtimes = [float(row["runtime_seconds"]) for row in subset]
        epochs = [int(row["best_epoch"]) for row in subset]
        summary.append(
            {
                "candidate": key,
                "dataset": subset[0]["dataset"],
                "model": subset[0]["model"],
                "operator": subset[0]["operator"],
                "folds": len(subset),
                "mean_best_test_acc": st.mean(accs),
                "std_best_test_acc": st.pstdev(accs) if len(accs) > 1 else 0.0,
                "mean_test_loss": st.mean(losses),
                "mean_runtime_seconds": st.mean(runtimes),
                "mean_best_epoch": st.mean(epochs),
                "total_params": int(subset[0]["total_params"]),
            }
        )
    return summary


def write_csv(rows: List[Dict[str, object]], target: Path) -> None:
    """把结构化行写入 CSV 文件，并自动创建父目录。"""
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        target.write_text("", encoding="utf-8")
        return
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """脚本主入口，串联参数解析、数据读取、处理和结果写出。"""
    version = DEFAULT_EXPERIMENT_VERSION
    rows = load_rows(version)
    out_dir = record_dir(ROOT, normalize_version(version)) / "summaries"
    write_csv(rows, out_dir / "tuned_candidate_fold_rows.csv")
    write_csv(summarize(rows), out_dir / "tuned_candidate_summary.csv")
    print(out_dir / "tuned_candidate_summary.csv")


if __name__ == "__main__":
    main()
