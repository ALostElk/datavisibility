"""核心图 4.3.2（加分项）：差评投诉词共现网络——死亡信号如何抱团。

聚焦低分评论（≤2★），围绕一组与就餐体验相关的投诉词，统计它们在同一条评论中
的共现（Jaccard），用 networkx 弹簧布局成网络：节点大小∝词频，边宽∝共现强度，
按社区着色揭示"服务/等待""出品质量""订单与价格"等抱团的投诉主题。
对应课程 L9（图/关系数据可视化）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from analysis import dataio, theme
from src import config

N_NEG = 90_000
MIN_EDGE = 0.05      # 共现 Jaccard 下限（显出骨架，避免毛球）
MAX_DEG = 6          # 每节点最多保留的最强边数（k 近邻式剪枝）

# 与就餐体验强相关的投诉词典（可解释、聚焦"死亡信号"）
LEXICON = [
    # 服务 / 态度
    "service", "staff", "server", "waiter", "waitress", "manager",
    "rude", "attitude", "ignored", "unprofessional",
    # 等待 / 速度
    "wait", "waited", "minutes", "hour", "slow", "forever", "long",
    # 出品质量
    "cold", "bland", "dry", "overcooked", "undercooked", "raw", "frozen",
    "soggy", "greasy", "stale", "burnt", "flavor", "quality",
    # 订单 / 价格
    "wrong", "missing", "forgot", "mistake", "refund", "charged",
    "overpriced", "expensive", "price", "money",
    # 卫生 / 总体
    "dirty", "disappointed", "worst", "terrible", "awful", "horrible",
    "mediocre", "bad", "never",
]


def main() -> str:
    theme.apply()
    config.ensure_dirs()
    from sklearn.feature_extraction.text import CountVectorizer

    df = dataio.sample_reviews_text(n_per_class=80_000)
    neg = df[df["stars_y"] <= 2]
    if len(neg) > N_NEG:
        neg = neg.sample(N_NEG, random_state=config.RANDOM_SEED)
    print("差评文档数:", len(neg))

    vec = CountVectorizer(vocabulary=LEXICON, binary=True,
                          token_pattern=r"[a-zA-Z]{3,}")
    Xb = (vec.fit_transform(neg["text"]) > 0).astype(int)
    words = list(vec.get_feature_names_out())
    co = (Xb.T @ Xb).toarray()
    freq = np.diag(co).copy()
    np.fill_diagonal(co, 0)

    # 丢弃在差评中过于稀有的词，避免孤立噪声
    keep = [i for i, f in enumerate(freq) if f >= 200]
    words = [words[i] for i in keep]
    co = co[np.ix_(keep, keep)]
    freq = freq[keep]
    n = len(words)

    # 构图：Jaccard 边 + 每节点保留最强 MAX_DEG 条
    edges = {}
    for i in range(n):
        cand = []
        for j in range(n):
            if i == j:
                continue
            jac = co[i, j] / (freq[i] + freq[j] - co[i, j] + 1e-9)
            if jac >= MIN_EDGE:
                cand.append((jac, j))
        cand.sort(reverse=True)
        for jac, j in cand[:MAX_DEG]:
            a, b = sorted((i, j))
            edges[(a, b)] = max(edges.get((a, b), 0), jac)

    G = nx.Graph()
    for i, w in enumerate(words):
        G.add_node(w, freq=int(freq[i]))
    for (a, b), w in edges.items():
        G.add_edge(words[a], words[b], weight=w)
    G.remove_nodes_from([nd for nd in list(G) if G.degree(nd) == 0])
    print("节点:", G.number_of_nodes(), "边:", G.number_of_edges())

    comms = list(nx.algorithms.community.greedy_modularity_communities(
        G, weight="weight"))
    palette = plt.cm.Set2(np.linspace(0, 1, max(len(comms), 3)))
    node_color = {nd: palette[ci % len(palette)]
                  for ci, com in enumerate(comms) for nd in com}

    fig, ax = plt.subplots(figsize=(12.5, 9))
    pos = nx.spring_layout(G, k=0.9, seed=config.RANDOM_SEED,
                           weight="weight", iterations=200)
    sizes = np.array([G.nodes[nd]["freq"] for nd in G], dtype=float)
    sizes = 300 + 3000 * (sizes / sizes.max())
    ews = np.array([G[u][v]["weight"] for u, v in G.edges()])

    nx.draw_networkx_edges(G, pos, ax=ax, width=0.6 + 7 * ews,
                           edge_color="#cdd2d7", alpha=0.75)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes,
                           node_color=[node_color[nd] for nd in G],
                           edgecolors="white", linewidths=1.4)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10,
                            font_color="#1d2125")
    ax.set_axis_off()
    ax.set_title("差评投诉词共现网络：投诉信号抱团成「出品·价格 / 服务态度 / 等待速度」几类死亡主题",
                 fontsize=13.5, fontweight="bold")
    fig.tight_layout()
    out = config.FIG_DIR / "fig08_cooccur_network.png"
    fig.savefig(out)
    plt.close(fig)
    print("saved:", out)
    for ci, com in enumerate(comms):
        print(f"  社区{ci}:", sorted(com, key=lambda w: -G.nodes[w]['freq']))
    return str(out)


if __name__ == "__main__":
    main()
