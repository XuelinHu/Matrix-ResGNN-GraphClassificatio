"""定义同分支跨层复用的 VerticalRes 图分类模型。"""
from __future__ import annotations

from .common import BaseResidualGNN, ResidualConfig


class VerticalResGNN(BaseResidualGNN):
    """只沿深度方向复用同分支历史状态的模型。"""

    topology_mode = "vertical"

    def __init__(self, config: ResidualConfig, dataset: object):
        """初始化模块参数并调用父类构造逻辑。"""
        super().__init__(config=config, dataset=dataset)
