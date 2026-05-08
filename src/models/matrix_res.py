"""定义分支-层二维邻域复用的 MatrixRes 图分类模型。"""
from __future__ import annotations

from .common import BaseResidualGNN, ResidualConfig


class MatrixResGNN(BaseResidualGNN):
    """在分支-层二维局部邻域内复用残差信息的模型。"""

    topology_mode = "matrix"

    def __init__(self, config: ResidualConfig, dataset: object):
        """初始化模块参数并调用父类构造逻辑。"""
        super().__init__(config=config, dataset=dataset)
