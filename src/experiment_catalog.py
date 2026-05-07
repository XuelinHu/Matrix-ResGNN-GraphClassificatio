from __future__ import annotations

from typing import Dict, List


MAIN_DATASETS: List[str] = ["PROTEINS", "DD", "ENZYMES"]
SUPPLEMENTARY_DATASETS: List[str] = ["MUTAG", "AIDS", "Mutagenicity"]
ALL_ACTIVE_DATASETS: List[str] = [*MAIN_DATASETS, *SUPPLEMENTARY_DATASETS]

ACTIVE_OPERATORS: List[str] = ["GCNConv", "GATConv", "SAGEConv", "GINConv"]

MAIN_MODELS: List[str] = [
    "Plain",
    "VerticalRes",
    "HorizontalRes",
    "MatrixRes",
    "MatrixResGated",
]

MODEL_DISPLAY: Dict[str, str] = {
    "Plain": "Plain",
    "VerticalRes": "Vertical-Res",
    "HorizontalRes": "Horizontal-Res",
    "MatrixRes": "Matrix-Res",
    "MatrixResGated": "Matrix-Res (sparse/gated)",
}

DATASET_METADATA: Dict[str, Dict[str, str]] = {
    "PROTEINS": {
        "source": "PyG TUDataset",
        "family": "tu",
        "role": "main benchmark",
        "task_type": "graph classification",
        "split_protocol": "stratified 5-fold CV + inner validation split",
    },
    "DD": {
        "source": "PyG TUDataset",
        "family": "tu",
        "role": "main benchmark",
        "task_type": "graph classification",
        "split_protocol": "stratified 5-fold CV + inner validation split",
    },
    "ENZYMES": {
        "source": "PyG TUDataset",
        "family": "tu",
        "role": "supporting stress test",
        "task_type": "graph classification",
        "split_protocol": "stratified 5-fold CV + inner validation split",
    },
    "MUTAG": {
        "source": "PyG TUDataset",
        "family": "tu",
        "role": "supplementary robustness dataset",
        "task_type": "graph classification",
        "split_protocol": "stratified 5-fold CV + inner validation split",
    },
    "AIDS": {
        "source": "PyG TUDataset",
        "family": "tu",
        "role": "supplementary robustness dataset",
        "task_type": "graph classification",
        "split_protocol": "stratified 5-fold CV + inner validation split",
    },
    "Mutagenicity": {
        "source": "PyG TUDataset",
        "family": "tu",
        "role": "supplementary robustness dataset",
        "task_type": "graph classification",
        "split_protocol": "stratified 5-fold CV + inner validation split",
    },
}


def dataset_family(dataset_name: str) -> str:
    metadata = DATASET_METADATA.get(dataset_name)
    if metadata is not None:
        return metadata["family"]
    if dataset_name.startswith("ogbg-"):
        return "ogb_graphprop"
    return "tu"
