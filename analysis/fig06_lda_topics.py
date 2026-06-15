"""核心图 4.3.2：LDA 主题——倒闭店 vs 存活店在「谈论什么」上的系统差异。

对采样评论做 LDA 得到若干主题，按"主导评论"统计每个主题在【倒闭店】与【存活店】
评论中的占比，画发散棒棒糖：
  右(红)=该主题在倒闭店评论中占比更高（潜在死亡主题，如 service/wait）
  左(绿)=在存活店中更突出（好评主题）
每行标注主题关键词。对应课程 L7（文本主题可视化）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from analysis import dataio, theme
from src import config

N_TOPICS = 8
N_DOCS = 60_000
TOPN_WORDS = 7


def main() -> str:
    theme.apply()
    config.ensure_dirs()

    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer

    df = dataio.sample_reviews_text(n_per_class=N_DOCS // 2).reset_index(drop=True)
    df["is_open"] = df["is_open"].astype(int)
    print("LDA 文档数:", len(df))

    vec = CountVectorizer(max_features=4000, stop_words="english",
                          min_df=30, ngram_range=(1, 2))
    X = vec.fit_transform(df["text"])
    lda = LatentDirichletAllocation(n_components=N_TOPICS, learning_method="online",
                                    random_state=config.RANDOM_SEED, max_iter=15,
                                    batch_size=4096)
    doc_topic = lda.fit_transform(X)

    names = np.array(vec.get_feature_names_out())
    keywords = {k: ", ".join(names[comp.argsort()[-TOPN_WORDS:][::-1]])
                for k, comp in enumerate(lda.components_)}

    df["topic"] = doc_topic.argmax(axis=1)
    df["avg_stars"] = df["stars_y"]
    # 各主题在倒闭/存活店评论中的占比
    share = (df.groupby(["is_open", "topic"]).size()
             / df.groupby("is_open").size()).unstack(0)
    share.columns = ["closed", "open"]
    share["diff"] = (share["closed"] - share["open"]) * 100   # 百分点
    stars = df.groupby("topic")["stars_y"].mean()
    share = share.sort_values("diff")

    fig, ax = plt.subplots(figsize=(13, 6.6))
    y = np.arange(len(share))
    for yi, (tid, row) in zip(y, share.iterrows()):
        color = theme.CLOSED_COLOR if row["diff"] > 0 else theme.OPEN_COLOR
        ax.plot([0, row["diff"]], [yi, yi], color=color, lw=2.5, zorder=1)
        ax.scatter(row["diff"], yi, s=120, color=color, zorder=2,
                   edgecolor="white")
        ha = "left" if row["diff"] > 0 else "right"
        off = 0.05 if row["diff"] > 0 else -0.05
        ax.text(row["diff"] + off, yi,
                f"{keywords[tid]}  (均分{stars[tid]:.1f})",
                va="center", ha=ha, fontsize=8.5, color="#222")
    ax.axvline(0, color="#333", lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([f"主题{int(t)}" for t in share.index])
    mx = max(abs(share["diff"])) * 2.4
    ax.set_xlim(-mx, mx)
    ax.set_xlabel("主题占比差（倒闭店 − 存活店，百分点）  ←存活更多 | 倒闭更多→")
    ax.set_title("倒闭店 vs 存活店：评论主题占比差异（红=死亡主题）",
                 fontweight="bold")
    ax.text(0.99, 0.02, "红点偏右=倒闭店更常谈及", transform=ax.transAxes,
            ha="right", color=theme.CLOSED_COLOR, fontsize=9)

    fig.tight_layout()
    out = config.FIG_DIR / "fig06_lda_topics.png"
    fig.savefig(out)
    plt.close(fig)
    print("saved:", out)
    for tid, row in share.iterrows():
        print(f"  主题{tid} 差{row['diff']:+.1f}pp 均分{stars[tid]:.2f}: {keywords[tid]}")
    return str(out)


if __name__ == "__main__":
    main()
