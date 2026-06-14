"""核心图 4.4：各州餐厅倒闭率地图（交互式 choropleth）。

按美国州聚合倒闭率（1 - 存活率），输出交互式 HTML；若安装了 kaleido
另导出静态 PNG 供报告插图。仅纳入商户数≥阈值的州以避免小样本噪声。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import plotly.express as px

sys.path.append(str(Path(__file__).resolve().parent.parent))
from analysis import dataio, theme
from src import config

MIN_BIZ = 50
US_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID",
    "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS",
    "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK",
    "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV",
    "WI", "WY", "DC",
}


def main() -> str:
    config.ensure_dirs()
    df = dataio.business_df().dropna(subset=["is_open", "state"])
    df = df[df["state"].isin(US_STATES)]

    agg = df.groupby("state").agg(
        n=("business_id", "size"),
        survival=("is_open", "mean"),
    ).reset_index()
    agg = agg[agg["n"] >= MIN_BIZ]
    agg["closure_rate"] = (1 - agg["survival"]) * 100

    fig = px.choropleth(
        agg, locations="state", locationmode="USA-states",
        color="closure_rate", scope="usa",
        color_continuous_scale="Reds",
        range_color=(agg["closure_rate"].min(), agg["closure_rate"].max()),
        hover_name="state",
        hover_data={"closure_rate": ":.1f", "n": ":,", "state": False},
        labels={"closure_rate": "倒闭率(%)", "n": "餐厅数"},
    )
    fig.update_layout(
        title_text="美国各州餐厅倒闭率（颜色越深越高）",
        title_x=0.5, font=dict(family="Microsoft YaHei", size=14),
        geo=dict(lakecolor="white"), margin=dict(l=10, r=10, t=60, b=10),
    )

    html = config.INTERACTIVE_DIR / "fig04_geo_closure.html"
    fig.write_html(str(html))
    print("saved html:", html)

    # 静态报告图：各州倒闭率排序条形（避开 kaleido 在 Windows 的挂起问题）
    png = _static_bar(agg)

    top = agg.sort_values("closure_rate", ascending=False).head(5)
    print("倒闭率最高的州:\n", top[["state", "closure_rate", "n"]].to_string(index=False))
    return str(png)


def _static_bar(agg) -> str:
    theme.apply()
    d = agg.sort_values("closure_rate")
    overall = d["closure_rate"].mean()
    fig, ax = plt.subplots(figsize=(7.5, max(5, len(d) * 0.33)))
    norm = plt.Normalize(d["closure_rate"].min(), d["closure_rate"].max())
    colors = plt.cm.Reds(norm(d["closure_rate"]))
    ax.barh(d["state"], d["closure_rate"], color=colors)
    ax.axvline(overall, color="#333", ls="--", lw=1,
               label=f"各州均值 {overall:.1f}%")
    for y, (st, v, n) in enumerate(d[["state", "closure_rate", "n"]].values):
        ax.text(v + 0.3, y, f"{v:.0f}%", va="center", fontsize=8)
    ax.set_xlabel("倒闭率 (%)")
    ax.set_title("美国各州餐厅倒闭率（按高低排序）", fontweight="bold")
    ax.legend(frameon=False, loc="lower right")
    ax.set_xlim(0, d["closure_rate"].max() * 1.12)
    fig.tight_layout()
    out = config.FIG_DIR / "fig04_geo_closure_bar.png"
    fig.savefig(out)
    plt.close(fig)
    print("saved png:", out)
    return str(out)


if __name__ == "__main__":
    main()
