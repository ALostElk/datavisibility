"""核心图 4.1.2：星级"失效"——高分≠存活（破直觉核心图）。

并排两图（提琴 + 折线，避免条形堆叠）：
  左：营业 vs 倒闭 餐厅店均星级的【提琴+箱线】分布——形态几乎重合、均值仅差 0.02。
  右：按星级细分箱的【倒闭率折线】——不随星级单调下降，高分区间倒闭率仍然不低。
结论：平均星级几乎无法区分餐厅生死。对应课程 L3（诚实分布对比）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append(str(Path(__file__).resolve().parent.parent))
from analysis import dataio, theme
from src import config


def main() -> str:
    sns.set_style("white")
    theme.apply()
    config.ensure_dirs()

    df = dataio.business_df().dropna(subset=["is_open", "stars_x"]).copy()
    df["is_open"] = df["is_open"].astype(int)
    df["状态"] = df["is_open"].map({1: "营业", 0: "倒闭"})
    open_m = df.loc[df["is_open"] == 1, "stars_x"].mean()
    closed_m = df.loc[df["is_open"] == 0, "stars_x"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))

    # 左：提琴 + 箱线
    ax = axes[0]
    palette = {"倒闭": theme.CLOSED_COLOR, "营业": theme.OPEN_COLOR}
    sns.violinplot(data=df, x="状态", y="stars_x", hue="状态",
                   order=["倒闭", "营业"], palette=palette, inner="box",
                   cut=0, ax=ax, legend=False)
    for i, (lab, mu) in enumerate([("倒闭", closed_m), ("营业", open_m)]):
        ax.scatter(i, mu, color="white", edgecolor="#222", zorder=5, s=45)
        ax.text(i, mu + 0.12, f"均值 {mu:.2f}", ha="center", fontsize=9,
                fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("店均星级 stars_x")
    ax.set_title(f"星级分布高度重合：均值仅差 {open_m - closed_m:.2f}")

    # 右：分星级倒闭率折线
    ax = axes[1]
    bins = np.arange(1, 5.6, 0.5)
    df["star_bin"] = pd.cut(df["stars_x"], bins=bins, include_lowest=True)
    rate = df.groupby("star_bin", observed=True)["is_open"].apply(
        lambda s: (1 - s.mean()) * 100)
    centers = [iv.mid for iv in rate.index]
    overall = (1 - df["is_open"].mean()) * 100
    ax.plot(centers, rate.values, color="#5b7fa6", lw=2.4, marker="o", ms=7)
    ax.axhline(overall, color="#e76f51", ls="--", lw=1.2,
               label=f"总体倒闭率 {overall:.0f}%")
    for x, y in zip(centers, rate.values):
        ax.text(x, y + 1.2, f"{y:.0f}%", ha="center", fontsize=9)
    ax.set_title("各星级区间倒闭率：无明显单调下降")
    ax.set_xlabel("店均星级")
    ax.set_ylabel("倒闭率 (%)")
    ax.set_ylim(0, max(rate.values) * 1.25)
    ax.legend(frameon=False)

    fig.suptitle("高分为什么也会倒闭？——星级几乎无法区分餐厅生死",
                 fontsize=16, fontweight="bold")
    fig.tight_layout()
    out = config.FIG_DIR / "fig01_stars_vs_survival.png"
    fig.savefig(out)
    plt.close(fig)
    print("saved:", out)
    print(f"营业均值 {open_m:.3f} vs 倒闭均值 {closed_m:.3f}")
    return str(out)


if __name__ == "__main__":
    main()
