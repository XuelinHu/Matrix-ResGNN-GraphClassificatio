from __future__ import annotations

from .common import BaseResidualGNN, ResidualConfig


class MatrixResGNN(BaseResidualGNN):
    """Structured 2D residual reuse on the branch x layer grid."""

    topology_mode = "matrix"

    def __init__(self, config: ResidualConfig, dataset: object):
        super().__init__(config=config, dataset=dataset)
