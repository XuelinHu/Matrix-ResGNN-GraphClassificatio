"""运行单个图分类实验配置，是其他批处理脚本的执行入口。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 仓库根目录：用于把脚本中的相对路径统一定位到项目根路径。
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment_paths import DEFAULT_EXPERIMENT_VERSION
from src.training import train_one_config


def parse_args() -> argparse.Namespace:
    """解析命令行参数，返回当前脚本需要的实验配置。"""
    parser = argparse.ArgumentParser(description="Run one Matrix-ResGNN graph-classification experiment.")
    parser.add_argument("--dataset", default="PROTEINS")
    parser.add_argument("--model", default="Plain", choices=["Plain", "VerticalRes", "HorizontalRes", "MatrixRes", "MatrixResGated"])
    parser.add_argument("--operator", default="GCNConv", choices=["GCNConv", "GATConv", "SAGEConv", "GINConv"])
    parser.add_argument("--version", default=DEFAULT_EXPERIMENT_VERSION)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--ep", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--drop", type=float, default=0.5)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--h_layer", type=int, default=4)
    parser.add_argument("--num_branches", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--min_delta", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=2.0)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--lr_patience", type=int, default=15)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--gate_init", type=float, default=0.8)
    parser.add_argument("--gate_mode", default="learnable", choices=["learnable", "fixed"])
    parser.add_argument("--fixed_gate_value", type=float, default=0.8)
    parser.add_argument("--residual_mode", default="identity", choices=["identity", "topk", "sparse"])
    parser.add_argument("--topk_ratio", type=float, default=0.5)
    parser.add_argument("--sparse_lambda", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    """脚本主入口，串联参数解析、数据读取、处理和结果写出。"""
    result = train_one_config(parse_args())
    print(result)


if __name__ == "__main__":
    main()
