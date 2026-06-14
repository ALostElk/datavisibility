"""数据加载与商户级聚合（全量）。

读取 config.TRAIN_PATH / TEST_PATH 两个 parquet，按需选列以控内存。
- business_df(): 去重到商户级（is_open 为结果变量），用于存活/表格分析。
- review_iter_cols(): 评论级按需取列（用于序列/文本分析）。
"""

from __future__ import annotations

import pandas as pd

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import config


def _read_both(columns: list[str]) -> pd.DataFrame:
    tr = pd.read_parquet(config.TRAIN_PATH, columns=columns)
    te = pd.read_parquet(config.TEST_PATH, columns=columns)
    return pd.concat([tr, te], ignore_index=True)


def business_df() -> pd.DataFrame:
    """商户级表：每个 business_id 一行，含 is_open、店均星级、评论数、品类、州、经纬度。"""
    cols = ["business_id", "name", "is_open", "stars_x", "review_count",
            "state", "city", "categories", "latitude", "longitude"]
    df = _read_both(cols).drop_duplicates("business_id").reset_index(drop=True)
    df["is_open"] = df["is_open"].astype("Int64")
    return df


def reviews_df(columns: list[str] | None = None) -> pd.DataFrame:
    """评论级表（默认取常用列）。"""
    cols = columns or ["business_id", "is_open", "stars_y", "date", "text"]
    return _read_both(cols)


def sample_reviews_text(n_per_class: int = 60_000, seed: int | None = None):
    """按 is_open 平衡分层抽样评论文本（用于 TF-IDF / LDA / 词云）。
    返回列：business_id, is_open, stars_y, text。"""
    seed = seed if seed is not None else config.RANDOM_SEED
    df = _read_both(["business_id", "is_open", "stars_y", "text"])
    df = df.dropna(subset=["is_open", "text"])
    df = df[df["text"].str.len() > 10]
    df["is_open"] = df["is_open"].astype(int)
    parts = []
    for val, grp in df.groupby("is_open"):
        parts.append(grp.sample(min(n_per_class, len(grp)), random_state=seed))
    return pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed)


import ast


def _coerce_bool(v):
    if v in ("True", True):
        return True
    if v in ("False", False):
        return False
    return None  # None / 'None' / 缺失


def _parse_attr_row(raw) -> dict:
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        d = ast.literal_eval(raw)
        return d if isinstance(d, dict) else {}
    except (ValueError, SyntaxError):
        return {}


def business_attrs_df() -> pd.DataFrame:
    """商户级表 + 解析后的结构化属性列（价格档/外卖/预订/停车/户外/团体）。"""
    cols = ["business_id", "is_open", "review_count", "attributes"]
    tr = pd.read_parquet(config.TRAIN_PATH, columns=cols).drop_duplicates("business_id")
    te = pd.read_parquet(config.TEST_PATH, columns=cols).drop_duplicates("business_id")
    df = pd.concat([tr, te], ignore_index=True).drop_duplicates("business_id")
    df = df.reset_index(drop=True)

    parsed = df["attributes"].apply(_parse_attr_row)

    def price(d):
        v = d.get("RestaurantsPriceRange2")
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    df["price_range"] = parsed.apply(price)
    df["takeout"] = parsed.apply(lambda d: _coerce_bool(d.get("RestaurantsTakeOut")))
    df["delivery"] = parsed.apply(lambda d: _coerce_bool(d.get("RestaurantsDelivery")))
    df["reservations"] = parsed.apply(lambda d: _coerce_bool(d.get("RestaurantsReservations")))
    df["outdoor"] = parsed.apply(lambda d: _coerce_bool(d.get("OutdoorSeating")))
    df["groups"] = parsed.apply(lambda d: _coerce_bool(d.get("RestaurantsGoodForGroups")))
    df["kids"] = parsed.apply(lambda d: _coerce_bool(d.get("GoodForKids")))
    df["is_open"] = df["is_open"].astype("Int64")
    return df.drop(columns=["attributes"])


def business_features_df() -> pd.DataFrame:
    """商户级建模主表：合并基础表(星级/评论数/州)与解析属性(价格档/便利性布尔)。
    用于 KMeans 聚类与可解释存活预测。"""
    base = business_df()[["business_id", "is_open", "stars_x",
                          "review_count", "state"]]
    attrs = business_attrs_df()[["business_id", "price_range", "takeout",
                                 "delivery", "reservations", "outdoor",
                                 "groups", "kids"]]
    df = base.merge(attrs, on="business_id", how="left")
    df["is_open"] = df["is_open"].astype("Int64")
    return df


if __name__ == "__main__":
    b = business_df()
    print("businesses:", len(b))
    print(b["is_open"].value_counts(dropna=False))
    print("closed share: %.3f" % (1 - b["is_open"].mean()))
