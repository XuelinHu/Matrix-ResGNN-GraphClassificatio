"""生成 ER、BA、SBM、WS 和随机正则合成图分类数据集。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 仓库根目录：用于把脚本中的相对路径统一定位到项目根路径。
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.synthetic_graphs import SYNTHETIC_DATASETS, synthetic_profile_defaults, write_synthetic_dataset


def parse_args() -> argparse.Namespace:
    """解析命令行参数，返回合成数据生成配置。"""
    defaults = synthetic_profile_defaults("paper")
    parser = argparse.ArgumentParser(description="Generate distribution-controlled synthetic graph datasets.")
    parser.add_argument("--datasets", nargs="+", default=SYNTHETIC_DATASETS)
    parser.add_argument("--profile", default="paper", choices=["paper", "smoke"])
    parser.add_argument("--graphs_per_class", type=int, default=None)
    parser.add_argument("--min_nodes", type=int, default=None)
    parser.add_argument("--max_nodes", type=int, default=None)
    parser.add_argument("--feature_dim", type=int, default=None)
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    """脚本主入口，生成并缓存指定合成图数据集。"""
    args = parse_args()
    root = ROOT / "data" / "SyntheticGraphClassification"
    generated = []
    for dataset_name in args.datasets:
        path = write_synthetic_dataset(
            root=root,
            dataset_name=dataset_name,
            profile=args.profile,
            graphs_per_class=args.graphs_per_class,
            min_nodes=args.min_nodes,
            max_nodes=args.max_nodes,
            feature_dim=args.feature_dim,
            seed=args.seed,
            force=args.force,
        )
        generated.append(str(path.relative_to(ROOT)))
    print(json.dumps({"profile": args.profile, "datasets": args.datasets, "paths": generated}, indent=2))


if __name__ == "__main__":
    main()
 