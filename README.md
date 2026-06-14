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
├── data/
│   ├── raw/                   原始数据（HF 全量 parquet + 抽样 yelp_sample.parquet）
│   ├── interim/               中间产物
│   ├── processed/             清洗后分析主表（reviews.parquet / shops.parquet）
│   └── external/              外部数据（如美国州 geojson）
├── src/
│   ├── config.py              路径/字段/配色/常量
│   ├── data/                  download(下载抽样) / clean(清洗派生) / load(加载)
│   ├── features/              tabular(Wilson) / sequence(STL) / text(TF-IDF)
│   ├── models/                cluster(KMeans) / sentiment(LR) / topic(LDA)
│   └── viz/                   theme + overview/ranking/cluster_viz/timeseries/
│                              geo_map/text_viz/network（每个对应报告一节）
├── notebooks/                 探索性分析 EDA
├── output/                    figures(静态) / interactive(html) / tables
└── report/
    ├── 大纲.md                论文/报告结构大纲
    └── figures/               报告引用图
```

## 运行流程

```bash
pip install -r requirements.txt

# 1. 下载全量数据（2.85GB，存活/序列/表格分析均基于全量）
python -m src.data.download --full

# 2. 清洗 + 派生字段（含 is_open 聚合、评论速度/评分轨迹、预警指标），写出 processed 主表
python -m src.data.clean

# 3. 后续：特征 / 建模 / 可视化（按 report/大纲.md 章节推进）
#    文本建模(LDA/TF-IDF)可对评论分层抽样（见 config.TEXT_SAMPLE_SIZE）
```

## 数据说明

- 评论级主表 `reviews.parquet`：评分(`stars_y`)、情感(派生)、文本、时间、菜系、属性、`is_open`。
- 商户级主表 `shops.parquet`：店均分(`stars_x`)、评论数、价格档、经纬度、**`is_open`（结果变量）**、
  及派生的预警指标（评论停更时长、近 12 月评论速度/评分斜率、差评占比等）。
- 详细字段见 `report/大纲.md` 附录 A。

## 报告

完整结构见 [`report/大纲.md`](report/大纲.md)，已对齐课程优秀参考项目范式。
