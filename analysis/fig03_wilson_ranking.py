"""核心图 4.1.3：Wilson Score 置信排名——「真实口碑」比裸均分更能区分生死。

小样本店的好评率天然虚高（5 条全好评 = 100%），Wilson 置信下界对样本量做
收缩惩罚，给出更稳健的"真实口碑"。本图：
  左：好评率(裸) vs Wilson 置信下界 散点，按评论数着色，展示小样本的收缩效应。
  右：分别按"裸好评率"与"Wilson 下界"分十分位，比较各分位的存活率——
      Wilson 分位对存活率更单调、区分力更强。
对应课程 L3（诚实统计）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from analysis import dataio, theme
from src import config

Z = 1.96
MIN_REVIEWS = 5


def wilson_lower(pos: np.ndarray, n: np.ndarray, z: float = Z) -> np.ndarray:
    p = pos / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return (center - margin) / denom


def main() -> str:
    theme.apply()
    config.ensure_dirs()

    rev = dataio.reviews_df(["business_id", "is_open", "stars_y"])
    rev = rev.dropna(subset=["is_open", "stars_y"])
    rev["pos"] = (rev["stars_y"] >= 4).astype(int)

    g = rev.groupby("business_id").agg(
        n=("pos", "size"), pos=("pos", "sum"),
        is_open=("is_open", "first")).reset_index()
    g = g[g["n"] >= MIN_REVIEWS].copy()
    g["naive"] = g["pos"] / g["n"]
    g["wilson"] = wilson_lower(g["pos"].values, g["n"].values)
    g["is_open"] = g["is_open"].astype(int)
    print("纳入商户:", len(g))

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(13.5, 5.8), gridspec_kw={"width_ratios": [1.15, 1]})

    # 左：裸好评率 vs Wilson 下界，按评论数着色
    sc = ax.scatter(g["naive"], g["wilson"], c=np.log10(g["n"]),
                    cmap="viridis", s=8, alpha=0.45, edgecolors="none")
    ax.plot([0, 1], [0, 1], color="#888", ls="--", lw=1)
    ax.set_xlabel("裸好评率（4★+ 占比）")
    ax.set_ylabel("Wilson 置信下界（真实口碑）")
    ax.set_title("小样本好评率被「收缩」回真实水平")
    ax.set_xlim(0.2, 1.02)
    ax.set_ylim(0, 1.0)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("评论数（log10）")
    ax.annotate("虚线下方=被惩罚\n（样本越少，下沉越多）",
                xy=(0.95, 0.55), xytext=(0.46, 0.18), fontsize=9, color="#444",
                arrowprops=dict(arrowstyle="->", color="#888"))

    # 右：两种排名的十分位存活率对比
    def decile_survival(score_col):
        q = pd.qcut(g[score_col], 10, labels=False, duplicates="drop")
        return g.groupby(q)["is_open"].mean()

    s_naive = decile_survival("naive")
    s_wilson = decile_survival("wilson")
    x = np.arange(len(s_wilson))
    ax2.plot(x, s_naive.values, marker="o", color="#bfc6cc", lw=2,
             label="按裸好评率分位")
    ax2.plot(x, s_wilson.values, marker="o", color=theme.OPEN_COLOR, lw=2.4,
             label="按 Wilson 下界分位")
    ax2.set_xlabel("口碑分位（0=最低 → 9=最高）")
    ax2.set_ylabel("该分位的存活率")
    ax2.set_title("Wilson 分位对存活更单调、区分力更强")
    ax2.legend(frameon=False, loc="upper left")
    spread_n = s_naive.max() - s_naive.min()
    spread_w = s_wilson.max() - s_wilson.min()
    ax2.text(0.98, 0.04,
             f"最高-最低分位存活率差：\n裸好评率 {spread_n:.1%}  ·  Wilson {spread_w:.1%}",
             transform=ax2.transAxes, ha="right", va="bottom", fontsize=9,
             color="#333")

    fig.suptitle("真实口碑（Wilson 置信下界）比裸均分更能区分餐厅生死",
                 fontsize=16, fontweight="bold")
    fig.tight_layout()
    out = config.FIG_DIR / "fig03_wilson_ranking.png"
    fig.savefig(out)
    plt.close(fig)
    print("saved:", out)
    print(f"区分力(存活率跨度): naive={spread_n:.3f}  wilson={spread_w:.3f}")
    return str(out)


if __name__ == "__main__":
    main()
