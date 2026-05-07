from __future__ import annotations

from .common import BaseResidualGNN, ResidualConfig


class PlainGNN(BaseResidualGNN):
    """Baseline without residual exchange."""

    topology_mode = "plain"

    def __init__(self, config: ResidualConfig, dataset: object):
        super().__init__(config=config, dataset=dataset)
