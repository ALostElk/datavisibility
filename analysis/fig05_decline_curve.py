"""核心图 4.2.1：关门前的"评论衰退曲线"。

把每家店的时间轴对齐到"最后一条评论"（倒闭店≈关门时点），回看 24 个月，
对比【倒闭】与【营业】两个队列的：
  左：平均每月评论速度（人气是否在熄火）
  右：平均评分（口碑是否在滑坡）

为保证可比，仅纳入"评论跨度≥24个月且评论数≥20"的店，使每条相对月曲线
的分母（合格商户数）在窗口内恒定。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from analysis import dataio, theme
from src import config

WINDOW = config.DECLINE_WINDOW_MONTHS   # 24
MIN_REVIEWS = 20


def main() -> str:
    theme.apply()
    config.ensure_dirs()

    df = dataio.reviews_df(["business_id", "is_open", "stars_y", "date"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "is_open"])
    df["midx"] = df["date"].dt.year * 12 + (df["date"].dt.month - 1)

    g = df.groupby("business_id")
    last = g["midx"].transform("max")
    first = g["midx"].transform("min")
    df["rel"] = last - df["midx"]          # 0 = 最后一条评论所在月
    df["span"] = last - first
    n_rev = g["business_id"].transform("size")

    # 合格队列：跨度≥WINDOW 且 评论数≥MIN_REVIEWS
    qual = df[(df["span"] >= WINDOW) & (n_rev >= MIN_REVIEWS) &
              (df["rel"] <= WINDOW)].copy()
    qual["is_open"] = qual["is_open"].astype(int)

    n_biz = qual.groupby("is_open")["business_id"].nunique()
    print("合格商户数：", dict(n_biz))

    agg = qual.groupby(["is_open", "rel"]).agg(
        n_reviews=("business_id", "size"),
        avg_stars=("stars_y", "mean"),
    ).reset_index()
    agg["velocity"] = agg.apply(
        lambda r: r["n_reviews"] / n_biz[r["is_open"]], axis=1)

    # 剔除两端构造性假象：rel=0(必含末条评论而虚高) 与 rel=WINDOW(跨度筛选开业峰)
    PLOT_MIN, PLOT_MAX = 1, WINDOW - 1
    agg = agg[(agg["rel"] >= PLOT_MIN) & (agg["rel"] <= PLOT_MAX)].copy()

    # 速度归一化到"较早稳定期"基线(rel∈[13,WINDOW-1])，比较衰退形状
    base_mask = agg["rel"].between(13, PLOT_MAX)
    base = agg[base_mask].groupby("is_open")["velocity"].mean()
    agg["velocity_norm"] = agg.apply(
        lambda r: r["velocity"] / base[r["is_open"]], axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    styles = {1: ("营业", theme.OPEN_COLOR), 0: ("倒闭", theme.CLOSED_COLOR)}

    for ax, col, title, ylab in [
        (axes[0], "velocity_norm", "人气在熄火：关门前评论速度持续走低",
         "相对评论速度（早期基线=1.0）"),
        (axes[1], "avg_stars", "口碑也在滑坡：关门前评分走低", "平均评分"),
    ]:
        for status, (label, color) in styles.items():
            sub = agg[agg["is_open"] == status].sort_values("rel")
            ax.plot(-sub["rel"], sub[col], color=color, lw=2.2,
                    marker="o", ms=3, label=label)
        ax.set_title(title)
        ax.set_xlabel("距最后一条评论的月数（越靠右越接近关门）")
        ax.set_ylabel(ylab)
        ax.legend(frameon=False)
    axes[0].axhline(1.0, color="#999", ls=":", lw=1)

    fig.suptitle("关门前的衰退曲线：评论停更 + 评分滑坡 是最强预警信号",
                 fontsize=16, fontweight="bold")
    fig.tight_layout()
    out = config.FIG_DIR / "fig05_decline_curve.png"
    fig.savefig(out)
    plt.close(fig)
    print("saved:", out)
    return str(out)


if __name__ == "__main__":
    main()
