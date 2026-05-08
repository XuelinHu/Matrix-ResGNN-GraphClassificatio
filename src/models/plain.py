"""定义无残差交换的 Plain 基线图分类模型。"""
from __future__ import annotations

from .common import BaseResidualGNN, ResidualConfig


class PlainGNN(BaseResidualGNN):
    """无残差交换的基线模型。"""

    topology_mode = "plain"

    def __init__(self, config: ResidualConfig, dataset: object):
        """初始化模块参数并调用父类构造逻辑。"""
        super().__init__(config=config, dataset=dataset)
