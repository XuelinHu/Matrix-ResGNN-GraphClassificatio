"""集中定义论文图表的颜色、线型、标记和 Matplotlib 样式。"""
from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt
from matplotlib import rcParams


# 模型颜色配置：保证所有论文图中同一模型使用一致颜色。
MODEL_COLORS: Dict[str, str] = {
    "Plain": "#4D4D4D",
    "VerticalRes": "#0072B2",
    "HorizontalRes": "#D55E00",
    "MatrixRes": "#009E73",
    "MatrixResGated": "#CC79A7",
}

# 模型标记配置：保证折线图中不同模型容易区分。
MODEL_MARKERS: Dict[str, str] = {
    "Plain": "o",
    "VerticalRes": "s",
    "HorizontalRes": "^",
    "MatrixRes": "D",
    "MatrixResGated": "P",
}

# 模型线型配置：配合颜色和标记提升黑白打印下的可读性。
MODEL_LINESTYLES: Dict[str, object] = {
    "Plain": "-",
    "VerticalRes": "--",
    "HorizontalRes": "-.",
    "MatrixRes": ":",
    "MatrixResGated": (0, (5, 1.5)),
}


def apply_paper_style() -> None:
    """设置论文图表统一的字体、颜色、网格和导出风格。"""
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
    """为单个坐标轴应用统一网格和边框样式。"""
    if with_grid:
        ax.grid(axis=grid_axis, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
