"""matplotlib 中文主题（Tufte 极简：高 data-ink，去顶/右框）。"""

from __future__ import annotations

import matplotlib
from matplotlib import font_manager


def _cjk_font() -> str:
    for c in ["Microsoft YaHei", "SimHei", "SimSun", "DengXian"]:
        if c in {f.name for f in font_manager.fontManager.ttflist}:
            return c
    return "sans-serif"


def apply() -> None:
    matplotlib.rcParams.update({
        "font.sans-serif": [_cjk_font()],
        "axes.unicode_minus": False,
        "figure.dpi": 130,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#888888",
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "font.size": 10,
    })


# 存活/倒闭统一配色
OPEN_COLOR = "#2a9d8f"    # 营业-绿
CLOSED_COLOR = "#e76f51"  # 倒闭-红
