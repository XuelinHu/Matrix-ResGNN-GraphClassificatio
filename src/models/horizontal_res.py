"""定义同层跨分支交换的 HorizontalRes 图分类模型。"""
from __future__ import annotations

from .common import BaseResidualGNN, ResidualConfig


class HorizontalResGNN(BaseResidualGNN):
    """在同一层相邻分支之间交换残差信息的模型。"""

    topology_mode = "horizontal"

    def __init__(self, config: ResidualConfig, dataset: object):
        """初始化模块参数并调用父类构造逻辑。"""
        super().__init__(config=config, dataset=dataset)
