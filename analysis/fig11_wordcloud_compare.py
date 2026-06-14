"""核心图 4.3.1：倒闭店 vs 存活店评论「判别词」对比词云。

直接做高频词云会两边雷同（都是 food/good/place）。本图改用 Monroe 等(2008)的
"加权 log-odds + 信息先验"得到每个词的 z 分数：先验用全语料词频，使 food/place
这类泛词自动趋零、稀有偶发词被收缩，只有"系统性偏向某一类"的词获得高 |z|。
词号大小 ∝ |z|，从而让"死亡信号词"与"好评信号词"稳健浮现。对应课程 L7（文本可视化）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from analysis import dataio, theme
from src import config

N_PER_CLASS = 60_000
TOPN = 110
A0 = 1000.0     # 信息先验总质量

# 高频但无判别价值的填充词（口语/指代/泛化评价），从词云剔除
FILLER = {
    "like", "really", "just", "got", "went", "came", "didn", "don", "ve",
    "did", "think", "going", "definitely", "told", "actually", "pretty",
    "quite", "ll", "good", "great", "nice", "place", "food", "ordered",
    "order", "time", "restaurant", "wasn", "isn", "doesn", "id", "im",
}


def main() -> str:
    theme.apply()
    config.ensure_dirs()
    from sklearn.feature_extraction.text import CountVectorizer
    from wordcloud import WordCloud

    df = dataio.sample_reviews_text(n_per_class=N_PER_CLASS)
    df["is_open"] = df["is_open"].astype(int)
    print("文档数:", len(df), dict(df["is_open"].value_counts()))

    stop = set(CountVectorizer(stop_words="english").get_stop_words()) | FILLER
    vec = CountVectorizer(max_features=3500, stop_words=list(stop),
                          min_df=50, token_pattern=r"[a-zA-Z]{3,}")
    X = vec.fit_transform(df["text"])
    words = np.array(vec.get_feature_names_out())
    is_open = df["is_open"].values

    y_open = np.asarray(X[is_open == 1].sum(axis=0)).ravel().astype(float)
    y_closed = np.asarray(X[is_open == 0].sum(axis=0)).ravel().astype(float)

    # 加权 log-odds + 信息 Dirichlet 先验（Monroe et al. 2008）
    total = y_open + y_closed
    alpha = A0 * total / total.sum()
    n_open, n_closed = y_open.sum(), y_closed.sum()
    a0_open, a0_closed = n_open + A0, n_closed + A0
    lo_closed = np.log((y_closed + alpha) / (a0_closed - y_closed - alpha))
    lo_open = np.log((y_open + alpha) / (a0_open - y_open - alpha))
    delta = lo_closed - lo_open                       # >0 偏倒闭
    var = 1.0 / (y_closed + alpha) + 1.0 / (y_open + alpha)
    z = delta / np.sqrt(var)                          # 标准化 z 分数

    def top_freqs(positive: bool):
        s = z if positive else -z
        idx = np.argsort(s)[::-1][:TOPN]
        idx = [i for i in idx if s[i] > 1.0]          # |z|>1 才纳入
        return {words[i]: float(s[i]) for i in idx}

    closed_freq = top_freqs(True)
    open_freq = top_freqs(False)

    def make_cloud(freqs, color):
        return WordCloud(width=900, height=620, background_color="white",
                         prefer_horizontal=0.95, max_words=TOPN,
                         color_func=lambda *a, **k: color,
                         relative_scaling=0.55).generate_from_frequencies(freqs)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    axes[0].imshow(make_cloud(closed_freq, theme.CLOSED_COLOR))
    axes[0].set_title("倒闭店评论更突出的词（潜在「死亡信号」）",
                      color=theme.CLOSED_COLOR, fontweight="bold")
    axes[1].imshow(make_cloud(open_freq, theme.OPEN_COLOR))
    axes[1].set_title("存活店评论更突出的词", color=theme.OPEN_COLOR,
                      fontweight="bold")
    for ax in axes:
        ax.axis("off")

    fig.suptitle("评论里的判别词：倒闭店 vs 存活店", fontsize=16, fontweight="bold")
    fig.tight_layout()
    out = config.FIG_DIR / "fig11_wordcloud_compare.png"
    fig.savefig(out)
    plt.close(fig)
    print("saved:", out)
    print("倒闭 Top:", list(closed_freq)[:15])
    print("存活 Top:", list(open_freq)[:15])
    return str(out)


if __name__ == "__main__":
    main()
