from __future__ import annotations

from typing import Dict


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
    del operator
    protocol = dict(BASE_PROTOCOL)
    protocol.update(DATASET_OVERRIDES.get(dataset, {}))
    protocol.update(MODEL_OVERRIDES.get(model, {}))
    return protocol
