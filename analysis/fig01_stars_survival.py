"""核心图 4.1.2：星级"失效"——高分≠存活。

并排两图：
  左：营业 vs 倒闭 餐厅的店均星级分布（归一化直方 + 均值线），高度重叠。
  右：按星级分箱的倒闭率（柱状），看星级是否单调降低倒闭率。
结论：平均星级几乎无法区分餐厅生死。
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


def main() -> str:
    theme.apply()
    config.ensure_dirs()

    df = dataio.business_df().dropna(subset=["is_open", "stars_x"])
    open_s = df.loc[df["is_open"] == 1, "stars_x"]
    closed_s = df.loc[df["is_open"] == 0, "stars_x"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 左：星级分布对比
    ax = axes[0]
    bins = np.arange(1, 5.6, 0.5)
    ax.hist(open_s, bins=bins, density=True, alpha=0.55,
            color=theme.OPEN_COLOR, label=f"营业 (n={len(open_s):,})")
    ax.hist(closed_s, bins=bins, density=True, alpha=0.55,
            color=theme.CLOSED_COLOR, label=f"倒闭 (n={len(closed_s):,})")
    ax.axvline(open_s.mean(), color=theme.OPEN_COLOR, lw=2, ls="--")
    ax.axvline(closed_s.mean(), color=theme.CLOSED_COLOR, lw=2, ls="--")
    ax.set_title(f"星级分布高度重叠：均值 {closed_s.mean():.2f}(倒闭) vs {open_s.mean():.2f}(营业)")
    ax.set_xlabel("店均星级 stars_x")
    ax.set_ylabel("密度")
    ax.legend(frameon=False)

    # 右：分星级的倒闭率
    ax = axes[1]
    df["star_bin"] = pd.cut(df["stars_x"], bins=bins, include_lowest=True)
    rate = df.groupby("star_bin", observed=True)["is_open"].apply(
        lambda s: 1 - s.mean())
    centers = [iv.mid for iv in rate.index]
    ax.bar(centers, rate.values, width=0.4, color="#5b7fa6")
    ax.axhline(1 - df["is_open"].mean(), color="#666", ls=":",
               label=f"总体倒闭率 {1-df['is_open'].mean():.1%}")
    ax.set_title("各星级区间倒闭率：无明显单调下降")
    ax.set_xlabel("店均星级")
    ax.set_ylabel("倒闭率")
    ax.set_ylim(0, max(rate.values) * 1.25)
    for x, y in zip(centers, rate.values):
        ax.text(x, y + 0.01, f"{y:.0%}", ha="center", fontsize=9)
    ax.legend(frameon=False)

    fig.suptitle("高分为什么也会倒闭？——星级几乎无法区分餐厅生死",
                 fontsize=16, fontweight="bold")
    fig.tight_layout()
    out = config.FIG_DIR / "fig01_stars_vs_survival.png"
    fig.savefig(out)
    plt.close(fig)
    print("saved:", out)
    return str(out)


if __name__ == "__main__":
    main()
