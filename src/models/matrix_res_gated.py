from __future__ import annotations

from .common import BaseResidualGNN, ResidualConfig


class MatrixResGatedGNN(BaseResidualGNN):
    """Matrix residual with sparse or gated residual control."""

    topology_mode = "matrix"

    def __init__(self, config: ResidualConfig, dataset: object):
        super().__init__(config=config, dataset=dataset)
