"""核心图 4.3.4：评论情感分类（TF-IDF + 逻辑回归）。

标签来自评分：>=4 正面 / <=2 负面（丢弃中性 3 分）。
左：最具判别力的正/负情感词（diverging 条形）；右：测试集混淆矩阵。
此图无时间泄漏，词汇可推广，对应课程 L7 文本 + L3 模型评估。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from analysis import dataio, theme
from src import config

TOPN = 16


def main() -> str:
    theme.apply()
    config.ensure_dirs()

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix, f1_score
    from sklearn.model_selection import train_test_split

    raw = dataio.sample_reviews_text(n_per_class=80_000)
    raw = raw[raw["stars_y"] != 3].copy()
    raw["label"] = np.where(raw["stars_y"] >= 4, "positive", "negative")
    # 平衡正负
    m = raw["label"].value_counts().min()
    df = (raw.groupby("label", group_keys=False)
          .apply(lambda g: g.sample(min(len(g), m), random_state=config.RANDOM_SEED)))
    print("样本:", len(df), dict(df["label"].value_counts()))

    Xtr, Xte, ytr, yte = train_test_split(
        df["text"], df["label"], test_size=0.2,
        random_state=config.RANDOM_SEED, stratify=df["label"])
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                          stop_words="english", min_df=20, sublinear_tf=True)
    Xtr_v = vec.fit_transform(Xtr)
    Xte_v = vec.transform(Xte)
    clf = LogisticRegression(max_iter=1000, C=4.0)
    clf.fit(Xtr_v, ytr)
    pred = clf.predict(Xte_v)
    acc = (pred == yte).mean()
    f1 = f1_score(yte, pred, pos_label="positive")
    print("测试集 acc=%.3f  F1=%.3f" % (acc, f1))

    names = np.array(vec.get_feature_names_out())
    coef = clf.coef_[0]            # 正类=positive
    order = coef.argsort()
    neg_idx = order[:TOPN]
    pos_idx = order[-TOPN:][::-1]

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(14, 6.5), gridspec_kw={"width_ratios": [1.7, 1]})

    words = list(names[neg_idx][::-1]) + list(names[pos_idx][::-1])
    vals = list(coef[neg_idx][::-1]) + list(coef[pos_idx][::-1])
    colors = [theme.CLOSED_COLOR if v < 0 else theme.OPEN_COLOR for v in vals]
    yy = np.arange(len(words))
    ax.barh(yy, vals, color=colors)
    ax.set_yticks(yy)
    ax.set_yticklabels(words, fontsize=9)
    ax.axvline(0, color="#333", lw=0.8)
    ax.set_xlabel("逻辑回归系数（←负面   正面→）")
    ax.set_title("最具判别力的情感词")
    ax.text(0.98, 0.98, "正面", transform=ax.transAxes, color=theme.OPEN_COLOR,
            fontweight="bold", va="top", ha="right")
    ax.text(0.02, 0.02, "负面", transform=ax.transAxes, color=theme.CLOSED_COLOR,
            fontweight="bold", va="bottom")

    # 右：混淆矩阵
    cm = confusion_matrix(yte, pred, labels=["negative", "positive"])
    cmn = cm / cm.sum(axis=1, keepdims=True)
    im = ax2.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(["负面", "正面"])
    ax2.set_yticks([0, 1]); ax2.set_yticklabels(["负面", "正面"])
    ax2.set_xlabel("预测"); ax2.set_ylabel("真实")
    ax2.set_title(f"混淆矩阵（acc={acc:.2f}, F1={f1:.2f}）")
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, f"{cmn[i,j]:.0%}\n({cm[i,j]:,})", ha="center",
                     va="center", color="white" if cmn[i, j] > 0.5 else "#333",
                     fontsize=11)
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    fig.suptitle("评论情感分类：什么词在表扬，什么词在差评",
                 fontsize=16, fontweight="bold")
    fig.tight_layout()
    out = config.FIG_DIR / "fig05_sentiment_words.png"
    fig.savefig(out)
    plt.close(fig)
    print("saved:", out)
    print("负面 Top:", list(names[neg_idx][:12]))
    print("正面 Top:", list(names[pos_idx][:12]))
    return str(out)


if __name__ == "__main__":
    main()
