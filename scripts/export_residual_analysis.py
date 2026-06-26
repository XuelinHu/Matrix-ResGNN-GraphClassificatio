"""生成单个配置的残差分析导出计划文件。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.experiment_paths import DEFAULT_EXPERIMENT_VERSION, analysis_dir, ensure_version_manifest


def parse_args() -> argparse.Namespace:
    """解析命令行参数，返回当前脚本需要的实验配置。"""
    parser = argparse.ArgumentParser(description="Scaffold for exporting residual-analysis artifacts.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--operator", default="GCNConv")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--version", default=DEFAULT_EXPERIMENT_VERSION)
    return parser.parse_args()


def main() -> None:
    """脚本主入口，串联参数解析、数据读取、处理和结果写出。"""
    root = Path(__file__).resolve().parents[1]
    ensure_version_manifest(root)
    args = parse_args()
    out_dir = analysis_dir(root, args.version)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": args.dataset,
        "model": args.model,
        "operator": args.operator,
        "fold": args.fold,
        "status": "scaffold",
        "required_outputs": [
            "history.json",
            "checkpoint.pt",
            "predictions.csv",
            "embeddings.npz",
            "layer_states.pt",
            "residual_stats.json",
            "gradient_by_layer.csv",
            "cka_matrix.csv",
            "cosine_matrix.csv",
        ],
    }
    target = out_dir / f"{args.dataset}_{args.model}_{args.operator}_fold{args.fold}_analysis_plan.json"
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
