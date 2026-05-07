from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt
from matplotlib import rcParams


MODEL_COLORS: Dict[str, str] = {
    "Plain": "#4D4D4D",
    "VerticalRes": "#0072B2",
    "HorizontalRes": "#D55E00",
    "MatrixRes": "#009E73",
    "MatrixResGated": "#CC79A7",
}

MODEL_MARKERS: Dict[str, str] = {
    "Plain": "o",
    "VerticalRes": "s",
    "HorizontalRes": "^",
    "MatrixRes": "D",
    "MatrixResGated": "P",
}

MODEL_LINESTYLES: Dict[str, object] = {
    "Plain": "-",
    "VerticalRes": "--",
    "HorizontalRes": "-.",
    "MatrixRes": ":",
    "MatrixResGated": (0, (5, 1.5)),
}


def apply_paper_style() -> None:
    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    rcParams["font.size"] = 13
    rcParams["axes.titlesize"] = 16
    rcParams["axes.labelsize"] = 14
    rcParams["xtick.labelsize"] = 12
    rcParams["ytick.labelsize"] = 12
    rcParams["legend.fontsize"] = 12
    rcParams["figure.titlesize"] = 17
    rcParams["figure.dpi"] = 300
    rcParams["savefig.dpi"] = 300
    rcParams["savefig.facecolor"] = "white"
    rcParams["figure.facecolor"] = "white"
    rcParams["axes.facecolor"] = "white"
    rcParams["axes.edgecolor"] = "black"
    rcParams["axes.linewidth"] = 0.9
    rcParams["grid.color"] = "#D0D0D0"
    rcParams["grid.linewidth"] = 0.6
    rcParams["grid.alpha"] = 0.5
    rcParams["grid.linestyle"] = "--"
    rcParams["legend.frameon"] = False
    rcParams["savefig.bbox"] = "tight"


def style_axis(ax: plt.Axes, with_grid: bool = True, grid_axis: str = "y") -> None:
    if with_grid:
        ax.grid(axis=grid_axis, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
