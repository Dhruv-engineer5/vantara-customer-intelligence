"""Unit tests for feature engineering, with explicit leakage tests (brief Sec 7)."""

import numpy as np
import pandas as pd
import pytest

from src.features.build import (
    CATEGORIES,
    FEATURE_COLUMNS,
    assert_no_leakage,
    assign_category,
    build_features,
    build_sequences,
)

CUT = "2011-01-01"


def test_only_customers_with_pre_cutoff_purchases(tx):
    feats, _ = build_features(tx, CUT, 90)
    assert set(feats.index) == {1, 2}


def test_core_feature_values(tx):
    feats, _ = build_features(tx, CUT, 90)
    c1 = feats.loc[1]
    assert c1["frequency"] == 3
    assert c1["recency_days"] == 12  # 2010-12-20 to 2011-01-01
    assert c1["monetary_total"] == pytest.approx(10 + 5 + 10)
    assert c1["orders_last_30d"] == 1
    assert c1["gap_mean_days"] == pytest.approx((31 + 49) / 2)
    c2 = feats.loc[2]
    assert c2["return_rate"] == pytest.approx(0.5)
    assert c2["clv_historical"] == pytest.approx(10 - 2)  # returns reduce net value


def test_targets_use_only_future_window(tx):
    _, tg = build_features(tx, CUT, 90)
    assert tg.loc[1, "churn"] == 0 and tg.loc[2, "churn"] == 1
    assert tg.loc[1, "future_spend"] == pytest.approx(10.0)


def test_features_invariant_to_future_data(tx):
    """LEAKAGE TEST: changing any row on/after the cutoff must not change any feature."""
    base, _ = build_features(tx, CUT, 90)
    mutated = tx.copy()
    fut = mutated["invoice_date"] >= pd.Timestamp(CUT)
    mutated.loc[fut, "quantity"] = 999
    mutated.loc[fut, "line_total"] = mutated.loc[fut, "quantity"] * mutated.loc[fut, "price"]
    mutated.loc[fut, "price"] = 77.0
    after, _ = build_features(mutated, CUT, 90)
    pd.testing.assert_frame_equal(base, after)


def test_dropping_all_future_rows_changes_nothing(tx):
    base, _ = build_features(tx, CUT, 90)
    past_only, _ = build_features(tx[tx["invoice_date"] < pd.Timestamp(CUT)], CUT, 90)
    pd.testing.assert_frame_equal(base, past_only)


def test_sequences_use_only_past(tx):
    seq = build_sequences(tx, CUT, [1, 2], length=4)
    assert seq.shape == (2, 4, 3 + len(CATEGORIES))
    assert (seq[0, 0] == 0).all()  # left padded
    mutated = tx.copy()
    mutated.loc[mutated["invoice_date"] >= pd.Timestamp(CUT), "line_total"] = 1e6
    np.testing.assert_array_equal(seq, build_sequences(mutated, CUT, [1, 2], length=4))


def test_assert_no_leakage_raises():
    with pytest.raises(AssertionError):
        assert_no_leakage(pd.Series(pd.to_datetime(["2011-01-02"])), CUT)
    assert_no_leakage(pd.Series(pd.to_datetime(["2010-12-31"])), CUT)


def test_category_rules():
    out = assign_category(pd.Series(["CHRISTMAS BAUBLE", "PAPER BAG", "UNKNOWN THING", None]))
    assert out.iloc[0] == "christmas_seasonal"
    assert out.iloc[2] == "other" and out.iloc[3] == "other"
    assert set(out) <= set(CATEGORIES)


def test_schema_complete(tx):
    feats, _ = build_features(tx, CUT, 90)
    assert set(FEATURE_COLUMNS) <= set(feats.columns)
    assert not feats[FEATURE_COLUMNS].isna().any().any()
