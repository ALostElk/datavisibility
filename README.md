# 高分为什么也会倒闭？——Yelp 餐厅存活信号挖掘与口碑衰退预警

> 表格 · 序列 · 文本三维视角下的餐厅"衰退预警"框架
>
> 数据来源：HuggingFace [`Johnnyeee/Yelpdata_663`](https://huggingface.co/datasets/Johnnyeee/Yelpdata_663)
> （Yelp 官方 business+review 餐厅子集，**全量 4,722,066 条评论 / 52,268 家餐厅 / 2005–2022**）

围绕单一结果变量 **`is_open`（餐厅是否仍在营业，全量 33.1% 已倒闭）** 这一主线，
回答"在平均星级几乎无法区分生死（倒闭 3.50 vs 存活 3.52）的前提下，
哪些可观测信号预示餐厅衰退与倒闭"。在**表格 / 序列 / 文本**三维上应用
Wilson Score 置信排名、STL 时序分解、KMeans 聚类、LDA 主题建模、
TF-IDF + 逻辑回归（情感分类 / 可解释存活预测）五类方法，
产出覆盖课程 L2–L10 可视化考点的分析报告、可推广的运营建议与"衰退预警指标卡"。

## 项目结构

```
数据可视化项目/
├── README.md                  本文件
├── requirements.txt           依赖
├── 大纲.md                    论文/报告结构大纲（章节、方法、考点映射）
├── data/
│   ├── README.md              数据获取说明
│   ├── Yelpdata.py            HuggingFace 下载脚本
│   └── *.parquet              全量数据（>100MB，已 .gitignore，按 data/README.md 下载）
├── src/
│   ├── config.py              路径 / 字段 / 配色 / 分析常量
│   └── __init__.py
├── analysis/                  每个脚本产出报告一节的核心图
│   ├── dataio.py              数据加载与商户级聚合 / 文本分层抽样
│   ├── theme.py               matplotlib 中文主题（Tufte 极简、统一配色）
│   └── fig01–fig13_*.py       13 张分析图（见下方索引）
└── output/
    ├── figures/               静态图 PNG（报告插图）
    └── interactive/           交互式 HTML（如各州倒闭率地图，已 .gitignore）
```

## 运行流程

```bash
pip install -r requirements.txt

# 1. 下载全量数据（见 data/README.md，约 2.85GB，放入 data/ 下）
#    存活/序列/表格分析基于全量；文本建模(LDA/TF-IDF/词云)按 config.TEXT_SAMPLE_SIZE 分层抽样

# 2. 逐张生成分析图（每个脚本对应大纲一节，结果写入 output/figures）
python -m analysis.fig01_stars_survival      # ... 直到 fig13
```

## 图表索引（对应大纲章节与课程考点）

| 图 | 脚本 | 大纲节 | 方法 / 课程 Lecture | 一句话结论 |
|---|---|---|---|---|
| fig01 | `fig01_stars_survival` | 4.1.2 | 分布对比 · L3 | 星级分布高度重叠，几乎无法区分生死 |
| fig02 | `fig02_decline_curve` | 4.2.3 | 事件对齐折线 · L8 | 关门前评论停更 + 评分滑坡是最强预警 |
| fig03 | `fig03_attribute_roi` | 4.1.3 | 哑铃图/分组条形 · L6 | 续命靠便利性属性而非高价 |
| fig04 | `fig04_geo_closure` | 4.4 | choropleth 地图 · L8 | 各州倒闭率地理格局（交互 HTML + 静态条形）|
| fig05 | `fig05_sentiment_words` | 4.3.3 | TF-IDF+LR · L7/L3 | 情感判别词 + 混淆矩阵 |
| fig06 | `fig06_lda_topics` | 4.3.2 | LDA 主题 · L7 | 主题及其情感倾向（投诉主题低分）|
| fig07 | `fig07_wilson_ranking` | 4.1.4 | Wilson 置信下界 · L3 | 真实口碑比裸均分更能区分生死 |
| fig08 | `fig08_cluster_risk` | 4.1.5 | KMeans + PCA · L6 | 低星级+评论稀少+缺属性=高危画像 |
| fig09 | `fig09_survival_model` | 4.1.6 | 逻辑回归(可解释)+ROC · L3 | 评论规模与配送是最强续命因子(AUC≈0.76) |
| fig10 | `fig10_trend_stl` | 4.2.1/4.2.2 | STL 时序分解 · L8 | 长期增长+稳定季节性+2020异常残差 |
| fig11 | `fig11_wordcloud_compare` | 4.3.1 | 加权 log-odds 词云 · L7 | 倒闭店死亡信号词 vs 存活店判别词 |
| fig12 | `fig12_cooccur_network` | 4.3.4 | 共现网络(networkx) · L9 | 投诉信号抱团：出品·价格/服务态度/等待速度 |
| fig13 | `fig13_tufte_redesign` | 第五章 | 改进前后对照 · L3/L10 | 诚实坐标轴 + 高 data-ink 的重设计示范 |

覆盖课程 **L3 / L6 / L7 / L8 / L9 / L10**，全程围绕单一结果变量 `is_open`。

## 数据说明

- 商户级：店均分(`stars_x`)、评论数、价格档、经纬度、解析属性（外卖/配送/预订/户外/团体/儿童）、**`is_open`（结果变量）**。
- 评论级：评分(`stars_y`)、文本(`text`)、时间(`date`)，用于序列(STL/衰退曲线)与文本(LDA/情感/词云/网络)分析。
- 详细字段见 [`大纲.md`](大纲.md) 附录 A。

## 报告

完整结构见 [`大纲.md`](大纲.md)，已对齐课程优秀参考项目范式。
