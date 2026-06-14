"""核心图 4.3.2：LDA 主题建模——顾客在谈论什么，哪些主题驱动差评。

对采样评论做 LDA，得到若干主题；按"主题主导评论的平均星级"排序，
红色低分=投诉主题，绿色高分=好评主题。每行标注主题关键词与占比。
对应课程 L7 文本数据可视化。
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
N_DOCS = 50_000
TOPN_WORDS = 8


def main() -> str:
    theme.apply()
    config.ensure_dirs()

    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer

    df = dataio.sample_reviews_text(n_per_class=N_DOCS // 2).reset_index(drop=True)
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

    dom = doc_topic.argmax(axis=1)
    df["topic"] = dom
    stats = df.groupby("topic").agg(
        avg_stars=("stars_y", "mean"), n=("text", "size")).reset_index()
    stats["share"] = stats["n"] / len(df)
    stats = stats.sort_values("avg_stars")

    fig, ax = plt.subplots(figsize=(13, 6.5))
    norm = plt.Normalize(1, 5)
    cmap = plt.cm.RdYlGn
    y = np.arange(len(stats))
    bars = ax.barh(y, stats["avg_stars"], color=cmap(norm(stats["avg_stars"])),
                   edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels([f"主题{int(t)}" for t in stats["topic"]])
    ax.set_xlim(1, 5.4)
    ax.set_xlabel("该主题主导评论的平均星级（越低越是投诉主题）")
    ax.set_title("顾客在谈论什么：LDA 主题与其情感倾向", fontweight="bold")
    ax.axvline(df["stars_y"].mean(), color="#333", ls="--", lw=1,
               label=f"总体均分 {df['stars_y'].mean():.2f}")
    for yi, row in zip(y, stats.itertuples()):
        ax.text(row.avg_stars + 0.04, yi,
                f"{keywords[row.topic]}  ·  占比{row.share:.0%}",
                va="center", fontsize=8.5, color="#222")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    out = config.FIG_DIR / "fig06_lda_topics.png"
    fig.savefig(out)
    plt.close(fig)
    print("saved:", out)
    for row in stats.itertuples():
        print(f"  主题{row.topic} 均分{row.avg_stars:.2f} 占比{row.share:.0%}: {keywords[row.topic]}")
    return str(out)


if __name__ == "__main__":
    main()
