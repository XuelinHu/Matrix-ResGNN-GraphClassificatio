from __future__ import annotations

from .common import BaseResidualGNN, ResidualConfig


class VerticalResGNN(BaseResidualGNN):
    """Same-branch, cross-layer residual reuse."""

    topology_mode = "vertical"

    def __init__(self, config: ResidualConfig, dataset: object):
        super().__init__(config=config, dataset=dataset)
