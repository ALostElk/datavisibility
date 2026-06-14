"""核心图 4.1.3：经营属性的"存活 ROI"。

哑铃图：每个二元属性"有 vs 无"两组的存活率对比，按差距(百分点)排序，
直观回答"配置哪些属性最能续命"。右侧小图：价格档与存活率关系。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from analysis import dataio, theme
from src import config

ATTRS = [
    ("reservations", "接受预订"),
    ("outdoor", "户外座位"),
    ("delivery", "提供配送"),
    ("takeout", "提供外卖"),
    ("groups", "适合团体"),
    ("kids", "适合儿童"),
]
NO_COLOR = "#bfc6cc"
YES_COLOR = "#2a9d8f"


def main() -> str:
    theme.apply()
    config.ensure_dirs()
    df = dataio.business_attrs_df().dropna(subset=["is_open"])
    overall = df["is_open"].mean()

    rows = []
    for col, label in ATTRS:
        sub = df.dropna(subset=[col])
        grp = sub.groupby(col)["is_open"].agg(["mean", "size"])
        if True in grp.index and False in grp.index:
            rows.append({
                "label": label,
                "yes": grp.loc[True, "mean"], "n_yes": int(grp.loc[True, "size"]),
                "no": grp.loc[False, "mean"], "n_no": int(grp.loc[False, "size"]),
            })
    res = sorted(rows, key=lambda r: r["yes"] - r["no"])

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(13.5, 5.6), gridspec_kw={"width_ratios": [2.1, 1]})

    y = np.arange(len(res))
    for i, r in enumerate(res):
        ax.plot([r["no"], r["yes"]], [i, i], color="#cdd3d8", lw=2.5, zorder=1)
        ax.scatter(r["no"], i, s=110, color=NO_COLOR, zorder=2,
                   edgecolor="white", linewidth=1)
        ax.scatter(r["yes"], i, s=130, color=YES_COLOR, zorder=3,
                   edgecolor="white", linewidth=1)
        gap = (r["yes"] - r["no"]) * 100
        xmid = max(r["yes"], r["no"]) + 0.012
        ax.text(xmid, i, f"{gap:+.1f}pp", va="center", ha="left", fontsize=10,
                color=("#2a9d8f" if gap >= 0 else "#e76f51"), fontweight="bold")
    ax.axvline(overall, color="#e76f51", ls="--", lw=1.2, zorder=0)
    ax.text(overall, len(res) - 0.4, f"  总体存活率 {overall:.0%}",
            color="#e76f51", fontsize=9, va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in res])
    ax.set_xlabel("存活率（餐厅仍在营业的比例）")
    ax.set_xlim(min(r["no"] for r in res) - 0.05, max(r["yes"] for r in res) + 0.09)
    ax.set_title("属性 ROI：配置哪些属性最能「续命」")
    # 图例
    ax.scatter([], [], s=110, color=NO_COLOR, label="无该属性")
    ax.scatter([], [], s=130, color=YES_COLOR, label="有该属性")
    ax.legend(frameon=False, loc="lower right")

    # 右：价格档 vs 存活率
    pr = df.dropna(subset=["price_range"])
    pr = pr[pr["price_range"].between(1, 4)]
    rate = pr.groupby("price_range")["is_open"].agg(["mean", "size"])
    bars = ax2.bar(rate.index, rate["mean"], width=0.62,
                   color=["#8ecae6", "#5b9bd5", "#3a7ca5", "#264653"])
    ax2.axhline(overall, color="#e76f51", ls="--", lw=1.2)
    for x, (m, n) in zip(rate.index, rate[["mean", "size"]].values):
        ax2.text(x, m + 0.012, f"{m:.0%}", ha="center", fontsize=9)
    ax2.set_xticks([1, 2, 3, 4])
    ax2.set_xticklabels([r"\$", r"\$\$", r"\$\$\$", r"\$\$\$\$"])
    ax2.set_ylim(0, max(rate["mean"]) * 1.2)
    ax2.set_xlabel("价格档")
    ax2.set_ylabel("存活率")
    ax2.set_title("价格档与存活率")

    fig.suptitle("能「续命」的不是高价，而是便利性属性",
                 fontsize=16, fontweight="bold")
    fig.tight_layout()
    out = config.FIG_DIR / "fig03_attribute_roi.png"
    fig.savefig(out)
    plt.close(fig)
    print("saved:", out)
    print("overall survival: %.3f" % overall)
    return str(out)


if __name__ == "__main__":
    main()
