"""核心图 4.2.4：口碑衰退「预警指标卡」——把衰退曲线提炼成可操作阈值。

由评论序列为每家店派生 4 个可观测预警指标，对比倒闭 vs 营业店的分布（提琴图），
并标注建议阈值与"倒闭店命中率"，构成项目最终落地的预警指标卡：
  1) 评论停更时长（月）           —— 距今多久没有新评论
  2) 近 12 月评论速度变化         —— (近12月条数 − 前12月条数) / 前12月
  3) 近 12 月评分变化             —— 近12月均分 − 前12月均分
  4) 近 12 月差评占比(≤2★)
对应课程 L6（多维分布对比）与项目结论（6.1 预警框架）。
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

MIN_REVIEWS = 15
MIN_SPAN = 24       # 月：需覆盖近/前各 12 个月

# (列名, 中文, 阈值, 方向)  方向 'low'=低于阈值预警, 'high'=高于阈值预警
INDICATORS = [
    ("months_since_last", "评论停更时长（月）", 12, "high"),
    ("velocity_change", "近12月评论速度变化", -0.3, "low"),
    ("stars_delta", "近12月评分变化", -0.2, "low"),
    ("neg_ratio", "近12月差评占比(≤2★)", 0.30, "high"),
]


def build_features() -> pd.DataFrame:
    df = dataio.reviews_df(["business_id", "is_open", "stars_y", "date"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "is_open", "stars_y"])
    df["m"] = df["date"].dt.year * 12 + (df["date"].dt.month - 1)
    gmax = df["m"].max()

    g = df.groupby("business_id")
    base = g.agg(last=("m", "max"), first=("m", "min"),
                 n=("m", "size"), is_open=("is_open", "first"))
    base = base[(base["n"] >= MIN_REVIEWS) &
                (base["last"] - base["first"] >= MIN_SPAN)]
    df = df[df["business_id"].isin(base.index)].copy()
    df = df.join(base["last"], on="business_id")
    df["rel"] = df["last"] - df["m"]
    df["neg"] = (df["stars_y"] <= 2).astype(float)

    rec = df[df["rel"] < 12].groupby("business_id").agg(
        n_recent=("m", "size"), stars_recent=("stars_y", "mean"),
        neg_recent=("neg", "mean"))
    pri = df[(df["rel"] >= 12) & (df["rel"] < 24)].groupby("business_id").agg(
        n_prior=("m", "size"), stars_prior=("stars_y", "mean"))

    f = base.join(rec).join(pri)
    f["n_prior"] = f["n_prior"].fillna(0)
    f["months_since_last"] = gmax - f["last"]
    f["velocity_change"] = (f["n_recent"] - f["n_prior"]) / (f["n_prior"] + 1)
    f["stars_delta"] = f["stars_recent"] - f["stars_prior"]
    f["neg_ratio"] = f["neg_recent"]
    f["状态"] = f["is_open"].map({1: "营业", 0: "倒闭"})
    return f.dropna(subset=["stars_delta"])


def main() -> str:
    sns.set_style("white")
    theme.apply()        # 必须在 set_style 之后，避免中文字体被重置
    config.ensure_dirs()

    f = build_features()
    print("纳入商户:", len(f), dict(f["状态"].value_counts()))

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))
    palette = {"倒闭": theme.CLOSED_COLOR, "营业": theme.OPEN_COLOR}

    for ax, (col, label, thr, direction) in zip(axes.ravel(), INDICATORS):
        lo, hi = f[col].quantile([0.02, 0.98])
        d = f[(f[col] >= lo) & (f[col] <= hi)]
        sns.violinplot(data=d, x="状态", y=col, hue="状态", order=["倒闭", "营业"],
                       palette=palette, inner="quartile", cut=0, ax=ax,
                       legend=False)
        ax.axhline(thr, color="#333", ls="--", lw=1.3)
        ax.text(1.5, thr, f" 阈值 {thr:g}", color="#333", fontsize=9,
                va="bottom", ha="right")

        if direction == "high":
            hit = (f.loc[f["is_open"] == 0, col] >= thr).mean()
            safe = (f.loc[f["is_open"] == 1, col] >= thr).mean()
        else:
            hit = (f.loc[f["is_open"] == 0, col] <= thr).mean()
            safe = (f.loc[f["is_open"] == 1, col] <= thr).mean()
        ax.set_title(f"{label}\n倒闭店命中 {hit:.0%}  vs  营业店误报 {safe:.0%}",
                     fontsize=12)
        ax.set_xlabel("")
        ax.set_ylabel("")

    fig.suptitle("口碑衰退预警指标卡：「停更·降速」是强信号，「评分·差评」几乎无差异（再证星级失效）",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = config.FIG_DIR / "fig15_warning_card.png"
    fig.savefig(out)
    plt.close(fig)
    print("saved:", out)
    for col, label, thr, d in INDICATORS:
        print(f"  {label}: 倒闭中位 {f.loc[f.is_open==0, col].median():.2f} | "
              f"营业中位 {f.loc[f.is_open==1, col].median():.2f}")
    return str(out)


if __name__ == "__main__":
    main()
