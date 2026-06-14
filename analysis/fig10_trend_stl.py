"""核心图 4.2.1 / 4.2.2：平台整体评论演化 + STL 时间序列分解。

全量评论按月聚合，刻画平台 2008–2021 的人气与口碑演化（含 2020 疫情冲击），
再用 STL 把"月度评论量"分解为 趋势 + 季节 + 残差 三分量。
对应课程 L8（时间序列可视化）与原理 3.3（STL）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from analysis import dataio, theme
from src import config

START, END = "2008-01", "2021-12"


def main() -> str:
    theme.apply()
    config.ensure_dirs()
    from statsmodels.tsa.seasonal import STL

    df = dataio.reviews_df(["date", "stars_y"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()

    m = df.groupby("month").agg(volume=("stars_y", "size"),
                                avg_stars=("stars_y", "mean"))
    m = m.loc[START:END]
    m = m.asfreq("MS")
    m["volume"] = m["volume"].interpolate()
    m["avg_stars"] = m["avg_stars"].interpolate()

    stl = STL(m["volume"], period=12, robust=True).fit()

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    # 1) 月度评论量 + 平均评分（双轴）
    ax = axes[0]
    ax.fill_between(m.index, m["volume"], color=theme.OPEN_COLOR, alpha=0.25)
    l1, = ax.plot(m.index, m["volume"], color=theme.OPEN_COLOR, lw=1.6,
                  label="月度评论量")
    ax.set_ylabel("月度评论量", color=theme.OPEN_COLOR)
    ax.tick_params(axis="y", colors=theme.OPEN_COLOR)
    axb = ax.twinx()
    l2, = axb.plot(m.index, m["avg_stars"], color="#9b5de5", lw=1.6,
                   label="月度平均评分")
    axb.set_ylabel("平均评分", color="#9b5de5", rotation=-90, labelpad=18)
    axb.tick_params(axis="y", colors="#9b5de5")
    axb.spines["top"].set_visible(False)
    covid = pd.Timestamp("2020-03-01")
    for a in (ax, axb):
        a.axvline(covid, color=theme.CLOSED_COLOR, ls="--", lw=1.3)
    ax.text(covid, ax.get_ylim()[1] * 0.92, " 2020 疫情冲击",
            color=theme.CLOSED_COLOR, fontsize=9, va="top")
    ax.legend([l1, l2], ["月度评论量", "月度平均评分"], frameon=False,
              loc="upper left", fontsize=9)
    ax.set_title("平台整体：评论量持续增长，2020 骤降；评分长期缓降", loc="left")

    # 2-4) STL 三联
    for ax_i, comp, name, color in [
        (axes[1], stl.trend, "趋势 Trend", "#264653"),
        (axes[2], stl.seasonal, "季节 Seasonal", "#e9a000"),
        (axes[3], stl.resid, "残差 Resid", "#9aa0a6"),
    ]:
        ax_i.plot(m.index, comp, color=color, lw=1.4)
        ax_i.set_ylabel(name)
        if name.startswith("残差"):
            ax_i.axhline(0, color="#bbb", lw=0.8)
        ax_i.axvline(covid, color=theme.CLOSED_COLOR, ls="--", lw=1)
    axes[3].set_xlabel("时间")

    fig.suptitle("平台评论演化与 STL 分解：长期增长 + 稳定季节性 + 2020 异常残差",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out = config.FIG_DIR / "fig10_trend_stl.png"
    fig.savefig(out)
    plt.close(fig)
    print("saved:", out)
    print("峰值月:", m["volume"].idxmax().date(), "量=", int(m["volume"].max()))
    return str(out)


if __name__ == "__main__":
    main()
