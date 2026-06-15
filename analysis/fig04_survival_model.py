"""核心图 4.1.4：可解释的存活预测——逻辑回归输出「风险/续命因子」。

仅用商户结构化特征（星级、评论规模、价格档、便利性属性）做标准化后逻辑回归，
预测 `is_open`。系数即可解释的因子方向与强度：正=利于存活，负=倒闭风险。
  左：标准化系数条形（风险因子排序）。
  右上：ROC 曲线 + AUC；右下：混淆矩阵。
对应课程 L3（可解释模型与评估）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from analysis import dataio, theme
from src import config

FEATURES = [
    ("stars_x", "店均星级"),
    ("log_reviews", "评论规模(log)"),
    ("price_range", "价格档"),
    ("takeout", "提供外卖"),
    ("delivery", "提供配送"),
    ("reservations", "接受预订"),
    ("outdoor", "户外座位"),
    ("groups", "适合团体"),
    ("kids", "适合儿童"),
]


def main() -> str:
    theme.apply()
    config.ensure_dirs()

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    df = dataio.business_features_df().dropna(subset=["is_open", "stars_x"]).copy()
    df["log_reviews"] = np.log10(df["review_count"].clip(lower=1))
    df["price_range"] = df["price_range"].fillna(df["price_range"].median())
    for c in ["takeout", "delivery", "reservations", "outdoor", "groups", "kids"]:
        df[c] = df[c].map({True: 1.0, False: 0.0}).fillna(0.0)

    cols = [c for c, _ in FEATURES]
    labels = [lab for _, lab in FEATURES]
    scaler = StandardScaler()
    X = scaler.fit_transform(df[cols].values)
    y = df["is_open"].astype(int).values

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=config.RANDOM_SEED, stratify=y)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    auc = roc_auc_score(yte, proba)
    acc = (pred == yte).mean()
    print(f"AUC={auc:.3f}  acc={acc:.3f}")

    coef = clf.coef_[0]
    o = np.argsort(coef)
    sc, sl = coef[o], np.array(labels)[o]

    fig = plt.figure(figsize=(14, 6.3))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1], height_ratios=[1, 1],
                          hspace=0.5, wspace=0.28)

    # 左：系数条形（占两行）
    ax = fig.add_subplot(gs[:, 0])
    colors = [theme.CLOSED_COLOR if v < 0 else theme.OPEN_COLOR for v in sc]
    yy = np.arange(len(sc))
    ax.barh(yy, sc, color=colors)
    ax.set_yticks(yy)
    ax.set_yticklabels(sl)
    ax.axvline(0, color="#333", lw=0.8)
    ax.set_xlabel("标准化逻辑回归系数（←倒闭风险   利于存活→）")
    ax.set_title("存活/倒闭风险因子（系数越正越「续命」）")
    for yi, v in zip(yy, sc):
        ax.text(v + (0.01 if v >= 0 else -0.01), yi, f"{v:+.2f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=8.5)
    pad = max(abs(sc)) * 0.25
    ax.set_xlim(sc.min() - pad, sc.max() + pad)

    # 右上：ROC
    ax1 = fig.add_subplot(gs[0, 1])
    fpr, tpr, _ = roc_curve(yte, proba)
    ax1.plot(fpr, tpr, color=theme.OPEN_COLOR, lw=2.2, label=f"AUC={auc:.3f}")
    ax1.plot([0, 1], [0, 1], color="#999", ls="--", lw=1)
    ax1.set_xlabel("假正率")
    ax1.set_ylabel("真正率")
    ax1.set_title("ROC 曲线")
    ax1.legend(frameon=False, loc="lower right")

    # 右下：混淆矩阵（行归一化）
    ax2 = fig.add_subplot(gs[1, 1])
    cm = confusion_matrix(yte, pred, labels=[0, 1])
    cmn = cm / cm.sum(axis=1, keepdims=True)
    im = ax2.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(["倒闭", "营业"])
    ax2.set_yticks([0, 1]); ax2.set_yticklabels(["倒闭", "营业"])
    ax2.set_xlabel("预测"); ax2.set_ylabel("真实")
    ax2.set_title(f"混淆矩阵（acc={acc:.2f}）")
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, f"{cmn[i,j]:.0%}", ha="center", va="center",
                     color="white" if cmn[i, j] > 0.5 else "#333", fontsize=11)

    fig.suptitle("用结构化特征可解释地预测餐厅存活：评论规模与星级是最强续命因子",
                 fontsize=15, fontweight="bold")
    out = config.FIG_DIR / "fig04_survival_model.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("saved:", out)
    for lab, v in sorted(zip(labels, coef), key=lambda t: t[1]):
        print(f"  {lab}: {v:+.3f}")
    return str(out)


if __name__ == "__main__":
    main()
