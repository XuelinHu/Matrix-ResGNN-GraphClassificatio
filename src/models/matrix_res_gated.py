"""定义带稀疏或门控残差控制的 MatrixResGated 图分类模型。"""
from __future__ import annotations

from .common import BaseResidualGNN, ResidualConfig


class MatrixResGatedGNN(BaseResidualGNN):
    """在 MatrixRes 拓扑上增加稀疏或门控残差控制的模型。"""

    topology_mode = "matrix"

    def __init__(self, config: ResidualConfig, dataset: object):
        """初始化模块参数并调用父类构造逻辑。"""
        super().__init__(config=config, dataset=dataset)
