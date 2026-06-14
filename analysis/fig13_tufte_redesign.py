"""叙事图（第五章 / L3·L10）：可视化「改进前 → 改进后」对照。

同一份数据（主要菜系的倒闭率），左边是常见的"误导/低 data-ink"设计，
右边是遵循 Tufte 原则的诚实重设计，直观展示可视化评估要点：
  · 诚实坐标轴（y 轴从 0 起，不夸大差异）
  · 高 data-ink（去 3D/渐变/重网格等 chartjunk）
  · 有序编码 + 直接标注 + 参照线（总体倒闭率）
对应课程 L3（图形评估）与 L10（评估与叙事）。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from analysis import dataio, theme
from src import config

CUISINES = {
    "Pizza": "披萨", "Mexican": "墨西哥", "Chinese": "中餐",
    "Italian": "意餐", "American (Traditional)": "美式传统",
    "Japanese": "日料", "Thai": "泰餐", "Burgers": "汉堡",
    "Sushi Bars": "寿司", "Breakfast & Brunch": "早午餐",
}


def main() -> str:
    theme.apply()
    config.ensure_dirs()

    df = dataio.business_df().dropna(subset=["is_open", "categories"])
    df["is_open"] = df["is_open"].astype(int)
    overall = (1 - df["is_open"].mean()) * 100

    rows = []
    for key, cn in CUISINES.items():
        m = df["categories"].str.contains(re.escape(key), case=False, na=False)
        sub = df[m]
        if len(sub) >= 300:
            rows.append((cn, (1 - sub["is_open"].mean()) * 100, len(sub)))
    rows.sort(key=lambda r: r[1])
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    ns = [r[2] for r in rows]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14.5, 6.2))

    # ---------- 左：误导 / 低 data-ink 设计 ----------
    x = np.arange(len(labels))
    ymin = min(vals) - 1.5            # 截断坐标轴，人为放大差异
    bars = axL.bar(x, vals, width=0.7,
                   color=plt.cm.rainbow(np.linspace(0, 1, len(vals))),
                   edgecolor="black", linewidth=1.2, zorder=3)
    # 伪 3D 阴影
    for b in bars:
        axL.bar(b.get_x() + 0.06, b.get_height(), width=0.7, bottom=0,
                color="#00000018", zorder=2)
    axL.set_ylim(ymin, max(vals) + 0.5)
    axL.set_xticks(x)
    axL.set_xticklabels(labels, rotation=45, ha="right")
    axL.grid(True, which="both", color="#bbb", lw=0.8, zorder=0)
    axL.set_facecolor("#eef2f7")
    for sp in axL.spines.values():
        sp.set_visible(True)
        sp.set_color("#333")
    axL.set_ylabel("倒闭率 (%)")
    axL.set_title("改进前：截断 y 轴 + 3D + 彩虹色 + 重网格",
                  color=theme.CLOSED_COLOR)
    axL.text(0.5, -0.42, "问题：y 轴不从 0 起 → 夸大差异；彩虹色无序、阴影/网格降低 data-ink",
             transform=axL.transAxes, ha="center", fontsize=9,
             color=theme.CLOSED_COLOR)

    # ---------- 右：Tufte 诚实重设计 ----------
    y = np.arange(len(labels))
    colors = [theme.CLOSED_COLOR if v > overall else theme.OPEN_COLOR
              for v in vals]
    axR.barh(y, vals, color=colors, zorder=3)
    axR.axvline(overall, color="#333", ls="--", lw=1.2, zorder=4,
                label=f"总体倒闭率 {overall:.1f}%")
    axR.set_yticks(y)
    axR.set_yticklabels(labels)
    axR.set_xlim(0, max(vals) * 1.16)
    for yi, (v, nn) in enumerate(zip(vals, ns)):
        axR.text(v + 0.4, yi, f"{v:.1f}%  (n={nn:,})", va="center", fontsize=9)
    axR.set_xlabel("倒闭率 (%)")
    axR.set_title("改进后：y 轴从 0、有序条形 + 直接标注 + 参照线",
                  color=theme.OPEN_COLOR)
    axR.legend(frameon=False, loc="lower right")
    axR.text(0.5, -0.2,
             "优点：诚实比例、按值排序便于比较、颜色仅编码高于/低于总体、直接标注省去查表",
             transform=axR.transAxes, ha="center", fontsize=9,
             color=theme.OPEN_COLOR)

    fig.suptitle("可视化评估与叙事：同一组「菜系倒闭率」的改进前 / 改进后对照",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = config.FIG_DIR / "fig13_tufte_redesign.png"
    fig.savefig(out)
    plt.close(fig)
    print("saved:", out)
    print("总体倒闭率 %.1f%%" % overall)
    for cn, v, nn in rows:
        print(f"  {cn}: {v:.1f}%  n={nn:,}")
    return str(out)


if __name__ == "__main__":
    main()
