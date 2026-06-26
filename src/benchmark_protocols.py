"""集中定义主实验、模型覆盖项和数据集覆盖项的训练协议配置。"""
from __future__ import annotations

from typing import Dict


# 基础训练协议：主实验默认使用的 epoch、学习率、分支数、早停和残差配置。
BASE_PROTOCOL: Dict[str, object] = {
    "ep": 200,
    "lr": 0.005,
    "weight_decay": 1e-4,
    "drop": 0.5,
    "dim": 64,
    "h_layer": 4,
    "num_branches": 3,
    "batch_size": 256,
    "patience": 60,
    "grad_clip": 2.0,
    "lr_factor": 0.5,
    "lr_patience": 15,
    "min_lr": 1e-5,
    "gate_init": 0.8,
    "gate_mode": "learnable",
    "fixed_gate_value": 0.8,
    "residual_mode": "identity",
    "topk_ratio": 0.5,
    "sparse_lambda": 0.05,
}


# 模型覆盖配置：针对不同残差拓扑微调训练步数、学习率和残差过滤方式。
MODEL_OVERRIDES: Dict[str, Dict[str, object]] = {
    "Plain": {
        "residual_mode": "identity",
    },
    "VerticalRes": {
        "residual_mode": "identity",
    },
    "HorizontalRes": {
        "residual_mode": "identity",
    },
    "MatrixRes": {
        "ep": 240,
        "patience": 80,
        "lr": 0.003,
        "weight_decay": 5e-5,
        "drop": 0.3,
    },
    "MatrixResGated": {
        "ep": 240,
        "patience": 80,
        "lr": 0.003,
        "weight_decay": 5e-5,
        "drop": 0.3,
        "residual_mode": "sparse",
        "sparse_lambda": 0.05,
    },
}


# 数据集覆盖配置：针对特定数据集调整层数、batch size 或 dropout。
DATASET_OVERRIDES: Dict[str, Dict[str, object]] = {
    "DD": {
        "h_layer": 3,
        "drop": 0.3,
    },
    "ENZYMES": {
        "batch_size": 128,
    },
}


def build_protocol(dataset: str, model: str, operator: str) -> Dict[str, object]:
    """合并基础协议、数据集覆盖项和模型覆盖项，得到最终训练配置。"""
    del operator
    protocol = dict(BASE_PROTOCOL)
    protocol.update(DATASET_OVERRIDES.get(dataset, {}))
    protocol.update(MODEL_OVERRIDES.get(model, {}))
    return protocol
