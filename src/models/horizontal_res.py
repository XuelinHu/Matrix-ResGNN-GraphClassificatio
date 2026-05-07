from __future__ import annotations

from .common import BaseResidualGNN, ResidualConfig


class HorizontalResGNN(BaseResidualGNN):
    """Cross-branch, same-layer residual exchange."""

    topology_mode = "horizontal"

    def __init__(self, config: ResidualConfig, dataset: object):
        super().__init__(config=config, dataset=dataset)
