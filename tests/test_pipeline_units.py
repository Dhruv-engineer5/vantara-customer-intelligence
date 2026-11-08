"""Unit tests for cleaning, validation, splitting, explanation text and segment names."""

import numpy as np
import pandas as pd
import pytest

from src.data.clean import clean_transactions
from src.data.validate import ValidationError, validate_clean, validate_raw
from src.explainability.explain import plain_language
from src.features.build import FEATURE_COLUMNS
from src.models.common import clf_metrics, make_split, reg_metrics
from src.segmentation.cluster import name_segments
from src.utils.config import load_config


def _raw():
    d = pd.Timestamp
    base = _raw_small()
    filler = pd.DataFrame(
        {
            "invoice": [f"F{i}" for i in range(30)],
            "stock_code": [f"Z{i}" for i in range(30)],
            "description": ["f"] * 30,
            "quantity": [2] * 30,
            "invoice_date": [d("2010-01-02")] * 30,
            "price": [1.0] * 30,
            "customer_id": pd.array([3] * 30, dtype="Int64"),
            "country": ["UK"] * 30,
        }
    )
    return pd.concat([base, filler], ignore_index=True)


def _raw_small():
    d = pd.Timestamp
    return pd.DataFrame(
        {
            "invoice": ["1", "2", "2", "C3", "4", "5", "6"],
            "stock_code": ["A", "X", "X", "A", "POST", "C", "Y"],
            "description": ["a", "b ", "b ", "a", "post", "c", "d"],
            "quantity": [1, 2, 2, -1, 1, 5000, 3],
            "invoice_date": [d("2010-01-01")] * 7,
            "price": [1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 0.0],
            "customer_id": pd.array([1, 1, 1, 1, 1, 2, None], dtype="Int64"),
            "country": ["UK"] * 7,
        }
    )


def test_cleaning_rules_counted():
    cfg = load_config()
    df, rep = clean_transactions(_raw(), cfg)
    assert rep["rows_missing_customer_id"] == 1
    assert rep["rows_non_product_stockcode"] == 1
    assert rep["rows_exact_duplicates"] == 1
    assert df["is_return"].sum() == 1  # returns kept as a signal
    assert (df["quantity"] == 5000).sum() == 1  # outlier rules are fitted later, on training customers only
    validate_clean(df, cfg)


def test_validation_fails_loudly():
    cfg = load_config()
    bad = _raw().drop(columns=["price"])
    with pytest.raises(ValidationError):
        validate_raw(bad, cfg)
    far = _raw()
    far["invoice_date"] = pd.Timestamp("2030-01-01")
    with pytest.raises(ValidationError):
        validate_raw(far, cfg)


def test_split_is_stratified_and_disjoint():
    cfg = load_config()
    t = pd.DataFrame({"churn": np.r_[np.ones(60), np.zeros(40)].astype(int)})
    s = make_split(t, cfg)
    allidx = np.concatenate(list(s.values()))
    assert len(set(allidx)) == len(allidx) == 100
    assert abs(t["churn"].iloc[s["train"]].mean() - 0.6) < 0.05
    assert make_split(t, cfg)["test"].tolist() == s["test"].tolist()  # fixed seed


def test_metrics():
    y = np.array([0, 0, 1, 1])
    m = clf_metrics(y, np.array([0.1, 0.4, 0.6, 0.9]))
    assert m["roc_auc"] == 1.0 and m["recall"] == 1.0
    assert reg_metrics(np.array([1.0, 2.0]), np.array([1.0, 2.0]))["r2"] == 1.0


def test_plain_language():
    sv = np.zeros(len(FEATURE_COLUMNS))
    i = FEATURE_COLUMNS.index("recency_days")
    sv[i] = 1.0
    x = pd.Series(np.zeros(len(FEATURE_COLUMNS)), index=FEATURE_COLUMNS)
    x["recency_days"] = 200
    s = plain_language(sv, x, 0.9)
    assert "high" in s and "200 days" in s and "raises risk" in s


def test_segment_names_unique():
    prof = pd.DataFrame({"monetary_total": [100, 5000, 900], "recency_days": [300, 20, 200], "frequency": [1, 12, 4]})
    names = name_segments(prof)
    assert len(set(names.values())) == 3
    assert names[1] == "Champions"
