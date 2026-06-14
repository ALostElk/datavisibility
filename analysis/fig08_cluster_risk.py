"""核心图 4.1.5：KMeans 商户分群与「高危画像」。

用店均星级、评论规模、价格档、便利性属性等做标准化后 KMeans 聚类，
比较各群存活率，刻画"高危店"长什么样：
  左：PCA 二维投影散点（按簇着色），点出各簇存活率。
  右：各簇标准化特征画像热力图 + 存活率条形，识别最高危/最稳健群体。
对应课程 L6（多维数据 / 聚类）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from analysis import dataio, theme
from src import config

K = 5
FEATURES = [
    ("stars_x", "店均星级"),
    ("log_reviews", "评论规模(log)"),
    ("price_range", "价格档"),
    ("takeout", "外卖"),
    ("delivery", "配送"),
    ("reservations", "预订"),
    ("outdoor", "户外座"),
]


def main() -> str:
    theme.apply()
    config.ensure_dirs()

    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    df = dataio.business_features_df().dropna(subset=["is_open", "stars_x"]).copy()
    df["log_reviews"] = np.log10(df["review_count"].clip(lower=1))
    df["price_range"] = df["price_range"].fillna(df["price_range"].median())
    for c in ["takeout", "delivery", "reservations", "outdoor"]:
        df[c] = df[c].map({True: 1.0, False: 0.0}).fillna(0.0)

    cols = [c for c, _ in FEATURES]
    X = StandardScaler().fit_transform(df[cols].values)
    km = KMeans(n_clusters=K, random_state=config.RANDOM_SEED, n_init=10)
    df["cluster"] = km.fit_predict(X)

    surv = df.groupby("cluster")["is_open"].mean()
    size = df.groupby("cluster").size()
    order = surv.sort_values().index.tolist()        # 低存活=高危在前
    overall = df["is_open"].mean()

    fig = plt.figure(figsize=(14.5, 6.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.25, 0.9], wspace=0.34)

    # 左：PCA 散点
    ax = fig.add_subplot(gs[0])
    pca = PCA(n_components=2, random_state=config.RANDOM_SEED)
    P = pca.fit_transform(X)
    samp = df.sample(min(12000, len(df)), random_state=config.RANDOM_SEED).index
    si = df.index.get_indexer(samp)
    palette = plt.cm.tab10(np.linspace(0, 1, K))
    for c in range(K):
        m = df.loc[samp, "cluster"].values == c
        ax.scatter(P[si][m, 0], P[si][m, 1], s=5, alpha=0.35,
                   color=palette[c], label=f"群{c}（存活{surv[c]:.0%}）")
    ax.set_xlabel("主成分 1")
    ax.set_ylabel("主成分 2")
    ax.set_title("KMeans 商户分群（PCA 投影）")
    ax.legend(frameon=True, fontsize=7.5, loc="upper left",
              framealpha=0.85, edgecolor="none", labelspacing=0.3,
              handletextpad=0.3, borderpad=0.3)

    # 中：簇画像热力图（标准化中心，按存活率排序）
    ax2 = fig.add_subplot(gs[1])
    centers = km.cluster_centers_[order]
    im = ax2.imshow(centers.T, cmap="RdBu_r", aspect="auto",
                    vmin=-np.abs(centers).max(), vmax=np.abs(centers).max())
    ax2.set_xticks(range(K))
    ax2.set_xticklabels([f"群{c}" for c in order])
    ax2.set_yticks(range(len(FEATURES)))
    ax2.set_yticklabels([lab for _, lab in FEATURES])
    ax2.set_title("各群特征画像（标准化，红高蓝低）")
    vmax = np.abs(centers).max()
    for i in range(K):
        for j in range(len(FEATURES)):
            v = centers[i, j]
            txt_color = "white" if abs(v) > 0.6 * vmax else "#222"
            ax2.text(i, j, f"{v:+.1f}", ha="center", va="center",
                     fontsize=7.5, color=txt_color)
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    # 右：各群存活率条形（高危在上）
    ax3 = fig.add_subplot(gs[2])
    y = np.arange(K)
    vals = [surv[c] for c in order]
    colors = [theme.CLOSED_COLOR if v < overall else theme.OPEN_COLOR for v in vals]
    ax3.barh(y, vals, color=colors)
    ax3.axvline(overall, color="#333", ls="--", lw=1,
                label=f"总体 {overall:.0%}")
    ax3.set_yticks(y)
    ax3.set_yticklabels([f"群{c}\n(n={size[c]:,})" for c in order], fontsize=8)
    ax3.invert_yaxis()
    ax3.set_xlabel("存活率")
    ax3.set_title("各群存活率")
    for yi, v in zip(y, vals):
        ax3.text(v + 0.005, yi, f"{v:.0%}", va="center", fontsize=8.5)
    ax3.set_xlim(0, max(vals) * 1.18)
    ax3.legend(frameon=False, loc="lower right", fontsize=8)

    fig.suptitle("KMeans 高危画像：低星级 + 评论稀少 + 缺便利性属性的群体最易倒闭",
                 fontsize=15, fontweight="bold")
    out = config.FIG_DIR / "fig08_cluster_risk.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("saved:", out)
    for c in order:
        print(f"  群{c}: 存活{surv[c]:.1%}  n={size[c]:,}")
    return str(out)


if __name__ == "__main__":
    main()
