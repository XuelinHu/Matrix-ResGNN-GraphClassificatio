from .horizontal_res import HorizontalResGNN
from .matrix_res import MatrixResGNN
from .matrix_res_gated import MatrixResGatedGNN
from .plain import PlainGNN
from .vertical_res import VerticalResGNN

__all__ = [
    "PlainGNN",
    "VerticalResGNN",
    "HorizontalResGNN",
    "MatrixResGNN",
    "MatrixResGatedGNN",
]
