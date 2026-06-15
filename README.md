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
python -m analysis.fig01_stars_survival      # ... 直到 fig15
```

## 图表索引（对应大纲章节与课程考点）

| 图 | 脚本 | 大纲节 | 图类型 / 课程 Lecture | 一句话结论 |
|---|---|---|---|---|
| fig14 | `fig14_survival_overview` | 4.1.1 | 树图 + 折线 · L5/L6 | 大市场竞争更激烈；评论≤10 倒闭率45% vs >250 仅14% |
| fig01 | `fig01_stars_survival` | 4.1.2 | 提琴+箱线 / 折线 · L3 | 星级分布高度重合（均值仅差0.03），无法区分生死 |
| fig03 | `fig03_attribute_roi` | 4.1.3 | 哑铃图 + 条形 · L6 | 续命靠便利性(外卖/配送)而非高价 |
| fig07 | `fig07_wilson_ranking` | 4.1.4 | 散点 + 折线 · L3 | 真实口碑(Wilson)比裸均分更能区分生死 |
| fig08 | `fig08_cluster_risk` | 4.1.5 | 散点 + 雷达 + 条形 · L6 | 外卖/配送群更稳健，高价·靠预订正餐群更高危 |
| fig09 | `fig09_survival_model` | 4.1.6 | 系数条形 + ROC/混淆矩阵 · L3 | 评论规模与配送是最强续命因子(AUC≈0.76) |
| fig10 | `fig10_trend_stl` | 4.2.1/4.2.2 | 面积+双线 / STL三联 · L8 | 长期增长+稳定季节性+2020异常残差 |
| fig02 | `fig02_decline_curve` | 4.2.3 | 事件对齐折线 · L8 | 关门前评论停更 + 评分滑坡是最强预警 |
| fig15 | `fig15_warning_card` | 4.2.4 | 提琴小多图 + 阈值线 · L6 | 预警指标卡：停更命中91% vs 误报9%（最强信号）|
| fig11 | `fig11_wordcloud_compare` | 4.3.1 | 加权 log-odds 词云 · L7 | 倒闭店:groupon/buffet/closed vs 存活店:amazing/best |
| fig06 | `fig06_lda_topics` | 4.3.2 | 发散棒棒糖 · L7 | 主题占比差异：存活店多好评主题，倒闭店多中性描述 |
| fig05 | `fig05_sentiment_words` | 4.3.3 | 判别词条形 + 混淆矩阵 · L7/L3 | 情感判别词 + 高精度好评/差评分类 |
| fig12 | `fig12_cooccur_network` | 4.3.4 | 关系网络(networkx) · L9 | 投诉抱团：出品·价格/服务态度/等待速度三类 |
| fig04 | `fig04_geo_closure` | 4.4 | 条形+散点+choropleth · L8 | CA最高(42%)；竞争越密集倒闭率越高(r≈0.17) |
| fig13 | `fig13_tufte_redesign` | 第五章 | 改进前后对照 · L3/L10 | 诚实坐标轴 + 高 data-ink 的重设计示范 |

> 表格按大纲叙事顺序排列（非文件编号顺序）。图类型涵盖提琴/树图/雷达/网络/地图/
> 散点/发散棒棒糖/哑铃/折线/词云等 10+ 种，覆盖课程 **L3/L5/L6/L7/L8/L9/L10**，
> 全程围绕单一结果变量 `is_open`。

## 数据说明

- 商户级：店均分(`stars_x`)、评论数、价格档、经纬度、解析属性（外卖/配送/预订/户外/团体/儿童）、**`is_open`（结果变量）**。
- 评论级：评分(`stars_y`)、文本(`text`)、时间(`date`)，用于序列(STL/衰退曲线)与文本(LDA/情感/词云/网络)分析。
- 详细字段见 [`大纲.md`](大纲.md) 附录 A。

## 报告

完整结构见 [`大纲.md`](大纲.md)，已对齐课程优秀参考项目范式。
