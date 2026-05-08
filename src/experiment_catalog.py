"""集中维护实验使用的数据集、算子、模型名称和数据集元信息。"""
from __future__ import annotations

from typing import Dict, List


# 主实验数据集：论文主体结果优先围绕这些图分类数据集展开。
MAIN_DATASETS: List[str] = ["PROTEINS", "DD", "ENZYMES"]
# 补充数据集：用于验证残差拓扑在分子图任务上的稳健性。
SUPPLEMENTARY_DATASETS: List[str] = ["MUTAG", "AIDS", "Mutagenicity"]
# 当前启用的数据集全集：批量 benchmark 会遍历该列表。
ALL_ACTIVE_DATASETS: List[str] = [*MAIN_DATASETS, *SUPPLEMENTARY_DATASETS]

# 当前启用的消息传递算子：用于四种 backbone 下的残差拓扑对比。
ACTIVE_OPERATORS: List[str] = ["GCNConv", "GATConv", "SAGEConv", "GINConv"]

# 主实验模型族：包含无残差、纵向、横向、矩阵和稀疏/门控矩阵残差。
MAIN_MODELS: List[str] = [
    "Plain",
    "VerticalRes",
    "HorizontalRes",
    "MatrixRes",
    "MatrixResGated",
]

# 论文图表显示名称：把代码模型名映射为图表中使用的模型名。
MODEL_DISPLAY: Dict[str, str] = {
    "Plain": "Plain",
    "VerticalRes": "Vertical-Res",
    "HorizontalRes": "Horizontal-Res",
    "MatrixRes": "Matrix-Res",
    "MatrixResGated": "Matrix-Res (sparse/gated)",
}

# 数据集元信息：记录来源、任务类型和划分协议，供论文和日志说明使用。
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
    """根据数据集名称返回加载器家族，用于选择数据读取和划分逻辑。"""
    metadata = DATASET_METADATA.get(dataset_name)
    if metadata is not None:
        return metadata["family"]
    if dataset_name.startswith("ogbg-"):
        return "ogb_graphprop"
    return "tu"
