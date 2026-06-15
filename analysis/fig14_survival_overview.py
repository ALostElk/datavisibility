"""核心图 4.1.1：存活全景——倒闭率的「市场规模」与「人气」格局。

研究问题 RQ1/RQ5 的开篇全景，用两种互补编码呈现"什么市场、什么人气的店在倒闭"：
  左：主要城市树图（矩形面积 ∝ 餐厅数，颜色 ∝ 倒闭率）——一眼看清"哪些大市场更危险"。
  右：按评论数（人气）分桶的倒闭率折线——人气越低越易倒闭，为后续"评论速度"伏笔。
对应课程 L5/L6（基本图形 + 复杂图形：树图/分桶趋势）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import squarify

sys.path.append(str(Path(__file__).resolve().parent.parent))
from analysis import dataio, theme
from src import config

TOP_CITIES = 18
MIN_CITY_BIZ = 150


def main() -> str:
    theme.apply()
    config.ensure_dirs()

    df = dataio.business_df().dropna(subset=["is_open"]).copy()
    df["is_open"] = df["is_open"].astype(int)
    overall = (1 - df["is_open"].mean()) * 100

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(14.5, 6.4), gridspec_kw={"width_ratios": [1.4, 1]})

    # 左：城市树图（面积=餐厅数，颜色=倒闭率）
    city = df.groupby("city").agg(
        n=("business_id", "size"), closure=("is_open", lambda s: (1 - s.mean()) * 100)
    ).reset_index()
    city = city[city["n"] >= MIN_CITY_BIZ].sort_values("n", ascending=False).head(TOP_CITIES)
    norm = plt.Normalize(city["closure"].min(), city["closure"].max())
    cmap = plt.cm.RdYlGn_r
    colors = [cmap(norm(c)) for c in city["closure"]]
    labels = [f"{c}\n{n:,}家\n{cl:.0f}%"
              for c, n, cl in zip(city["city"], city["n"], city["closure"])]
    squarify.plot(sizes=city["n"], label=labels, color=colors, ax=axL,
                  pad=True, text_kwargs={"fontsize": 8, "color": "#1d2125"},
                  ec="white")
    axL.axis("off")
    axL.set_title("主要城市：面积∝餐厅数，颜色∝倒闭率（越红越危险）")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=axL, fraction=0.046, pad=0.02)
    cb.set_label("倒闭率 (%)")

    # 右：人气（评论数）分桶倒闭率
    df["rev_bucket"] = pd.cut(
        df["review_count"],
        bins=[0, 10, 25, 50, 100, 250, 1e9],
        labels=["≤10", "11–25", "26–50", "51–100", "101–250", ">250"])
    rate = df.groupby("rev_bucket", observed=True).agg(
        closure=("is_open", lambda s: (1 - s.mean()) * 100),
        n=("business_id", "size")).reset_index()
    x = np.arange(len(rate))
    axR.plot(x, rate["closure"], color=theme.CLOSED_COLOR, lw=2.4, marker="o", ms=7)
    axR.fill_between(x, rate["closure"], overall, color=theme.CLOSED_COLOR, alpha=0.12)
    axR.axhline(overall, color="#333", ls="--", lw=1, label=f"总体 {overall:.0f}%")
    for xi, (cl, nn) in enumerate(zip(rate["closure"], rate["n"])):
        axR.text(xi, cl + 1.0, f"{cl:.0f}%", ha="center", fontsize=9,
                 color=theme.CLOSED_COLOR, fontweight="bold")
    axR.set_xticks(x)
    axR.set_xticklabels(rate["rev_bucket"])
    axR.set_xlabel("评论数（人气）分桶")
    axR.set_ylabel("倒闭率 (%)")
    axR.set_title("人气越低，倒闭率越高")
    axR.set_ylim(0, rate["closure"].max() * 1.2)
    axR.legend(frameon=False, loc="lower left")

    fig.suptitle("存活全景：大市场竞争更激烈、低人气店更易倒闭",
                 fontsize=16, fontweight="bold")
    fig.tight_layout()
    out = config.FIG_DIR / "fig14_survival_overview.png"
    fig.savefig(out)
    plt.close(fig)
    print("saved:", out)
    print("总体倒闭率 %.1f%%" % overall)
    print(rate.to_string(index=False))
    return str(out)


if __name__ == "__main__":
    main()
