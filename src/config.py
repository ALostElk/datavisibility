"""全局配置：数据源、路径、字段、配色、分析常量。

数据源：HuggingFace `Johnnyeee/Yelpdata_663`
  - Yelp 官方 business + review 合并后的【餐厅子集】
  - train 3,778,658 行 / test 943,408 行（共 4,722,066 行 / 52,268 家餐厅，约 2.85GB）

研究方向：围绕结果变量 `is_open`（餐厅是否仍在营业，全量 33.1% 已倒闭）
  做"存活信号挖掘与口碑衰退预警"。分析口径：
  - 表格/序列/存活：使用【全量】52,268 家商户聚合数据
  - 文本建模(LDA/TF-IDF)：4.7M 条评论较重，可按 TEXT_SAMPLE_SIZE 分层抽样

核心三维度：
  - 表格 tabular：星级、评论数、价格档、属性(外卖/预订/停车/氛围/WiFi)、is_open、经纬度
  - 序列 sequence：评论日期(date) → 评论速度/评分轨迹 / STL / 关门前衰退曲线
  - 文本 text：评论正文(text) → TF-IDF / LDA / 情感与倒闭判别词
"""

from __future__ import annotations

from pathlib import Path

# ---------------- 路径 ----------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
TRAIN_PATH = DATA_DIR / "yelptrain_data.parquet"   # 全量 train (3,778,658 行)
TEST_PATH = DATA_DIR / "yelptest_data.parquet"     # 全量 test  (943,408 行)
OUTPUT_DIR = ROOT / "output"
FIG_DIR = OUTPUT_DIR / "figures"
INTERACTIVE_DIR = OUTPUT_DIR / "interactive"
TABLE_DIR = OUTPUT_DIR / "tables"

# 仅在需要写出产物时显式调用，避免导入即创建空目录
def ensure_dirs() -> None:
    for _d in (FIG_DIR, INTERACTIVE_DIR, TABLE_DIR):
        _d.mkdir(parents=True, exist_ok=True)

# ---------------- 数据源 ----------------
HF_REPO_ID = "Johnnyeee/Yelpdata_663"
TEXT_SAMPLE_SIZE = 100_000    # 文本建模(LDA/TF-IDF)时对评论的分层抽样上限
RANDOM_SEED = 42

# ---------------- 存活/倒闭分析 ----------------
TARGET_COL = "is_open"        # 结果变量：1=营业, 0=倒闭
# 关门前"衰退曲线"对齐：以末条评论为 t=0，回看的月数窗口
DECLINE_WINDOW_MONTHS = 24
# 预警指标候选（在 features 中计算）
WARNING_FEATURES = [
    "months_since_last_review",   # 评论停更时长
    "velocity_slope_12m",         # 近 12 月评论速度斜率
    "stars_slope_12m",            # 近 12 月评分斜率
    "neg_review_ratio_12m",       # 近 12 月差评占比
]

# 原始字段（HF 数据集列名）
COL_BIZ_STARS = "stars_x"     # 商户平均星级
COL_REV_STARS = "stars_y"     # 单条评论星级
TEXT_COL = "text"
DATE_COL = "date"

# 需要从 attributes(JSON 字符串) 解析出的结构化属性
ATTR_KEYS = [
    "RestaurantsPriceRange2",  # 价格档 1-4
    "Ambience",                # 氛围(嵌套)
    "WiFi",
    "BusinessParking",
    "OutdoorSeating",
    "RestaurantsTakeOut",
    "RestaurantsDelivery",
    "GoodForKids",
    "Alcohol",
    "NoiseLevel",
]

# ---------------- 情感（自建分类）----------------
# 评分→情感映射规则：>=4 正面，==3 中性，<=2 负面（用于训练标签 / 基线）
SENTIMENT_CN = {"positive": "正面", "neutral": "中性", "negative": "负面"}


def stars_to_sentiment(stars: float) -> str:
    if stars >= 4:
        return "positive"
    if stars <= 2:
        return "negative"
    return "neutral"


# ---------------- 配色 ----------------
SENTIMENT_COLORS = {
    "positive": "#2a9d8f",
    "neutral": "#adb5bd",
    "negative": "#e76f51",
}
SEQ_CMAP = "YlOrRd"
DIVERGING_CMAP = "RdYlGn"
PRIMARY = "#264653"
ACCENT = "#e9c46a"
