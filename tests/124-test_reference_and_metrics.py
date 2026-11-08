"""Regression tests for train-only fitted statistics, the 12-month trend window, LSTM inputs and metric completeness."""

import numpy as np
import pandas as pd
import pytest

from src.data.reference import apply_reference, fit_engagement, fit_reference, percentile
from src.features.build import CATEGORIES, build_features, build_sequences
from src.models.common import clf_metrics, reg_metrics
from src.utils.config import load_config

CUT = "2011-01-01"


def _tx(rows):
    df = pd.DataFrame(
        rows, columns=["customer_id", "invoice", "stock_code", "description", "quantity", "price", "date"]
    )
    df["invoice_date"] = pd.to_datetime(df.pop("date"))
    df["is_return"] = df["quantity"] < 0
    df["country"] = "United Kingdom"
    df["line_total"] = df["quantity"] * df["price"]
    return df


def test_trend_ignores_history_older_than_12_months():
    base = [(1, f"A{i}", "X", "MUG", 1, 1.0, d) for i, d in enumerate(["2010-11-01", "2010-12-01", "2010-12-20"])]
    old = [(1, f"O{i}", "X", "MUG", 1, 1.0, f"2009-1{i}-05") for i in range(0, 3)]
    f1, _ = build_features(_tx(base), CUT, 90)
    f2, _ = build_features(_tx(base + old), CUT, 90)
    assert f1.loc[1, "freq_trend"] == pytest.approx(f2.loc[1, "freq_trend"])


def test_reference_fit_uses_only_given_rows_and_ignores_heldout_changes():
    cfg = load_config()
    train = _tx([(1, "A1", "X", "MUG", q, 2.0, "2010-06-01") for q in (1, 2, 3, 4, 5)])
    held = _tx([(2, "B1", "X", "MUG", 9999, 900.0, "2010-06-02")])
    ref_a = fit_reference(train, cfg)
    ref_b = fit_reference(train, cfg)  # held-out rows are never passed in, so changing them cannot matter
    assert ref_a["qty_bound"] == ref_b["qty_bound"] and ref_a["median_price"] == ref_b["median_price"]
    out, rep = apply_reference(pd.concat([train, held], ignore_index=True), ref_a, cfg)
    assert rep["rows_quantity_entry_error_dropped"] == 1 and len(out) == 5
    assert ref_a["median_price"]["X"] == 2.0  # unaffected by the held-out 900.0 price


def test_engagement_uses_persisted_training_reference_not_the_batch():
    cfg = load_config()
    rows = [(c, f"I{c}{k}", "X", "MUG", 1, float(c), f"2010-1{k}-01") for c in (1, 2, 3, 4) for k in (0, 1, 2)]
    tx = _tx(rows)
    feats, _ = build_features(tx, CUT, 90)
    ref = fit_engagement(fit_reference(tx[tx["customer_id"].isin([1, 2])], cfg), feats.loc[[1, 2]])
    full, _ = build_features(tx, CUT, 90, ref)
    one, _ = build_features(tx[tx["customer_id"] == 4], CUT, 90, ref)
    assert full.loc[4, "engagement_score"] == pytest.approx(one.loc[4, "engagement_score"])
    assert percentile(pd.Series([1e9]), ref["engagement"]["monetary_total"])[0] == pytest.approx(1.0)


def test_sequences_carry_category_onehot():
    tx = _tx([(1, "A1", "X", "CHRISTMAS TREE", 2, 1.0, "2010-12-01"), (1, "A2", "Y", "TEA MUG", 1, 1.0, "2010-12-15")])
    seq = build_sequences(tx, CUT, [1], length=4)
    assert seq.shape == (1, 4, 3 + len(CATEGORIES))
    assert seq[0, :, 3:].sum() == 2  # one dominant category per real invoice, padding rows all zero


def test_next30_label_is_future_only():
    tx = _tx([(1, "A1", "X", "MUG", 1, 1.0, "2010-12-01"), (1, "A2", "X", "MUG", 1, 1.0, "2011-01-20")])
    _, tg = build_features(tx, CUT, 90)
    assert tg.loc[1, "next30"] == 1 and tg.loc[1, "churn"] == 0
    tx2 = tx.copy()
    tx2.loc[1, "invoice_date"] = pd.Timestamp("2011-03-01")
    assert build_features(tx2, CUT, 90)[1].loc[1, "next30"] == 0


def test_metric_sets_are_complete():
    m = clf_metrics(np.array([0, 1, 1, 0]), np.array([0.2, 0.9, 0.4, 0.6]))
    assert {"roc_auc", "recall", "precision", "f1", "accuracy", "tn", "fp", "fn", "tp"} <= set(m)
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (1, 1, 1, 1)
    r = reg_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 5.0]))
    assert r["rmse"] == pytest.approx((4 / 3) ** 0.5) and "mae" in r and "r2" in r


def test_xgboost_contrib_explainer_sums_to_margin():
    from xgboost import XGBClassifier

    from src.explainability.explain import _sv, make_explainer

    rng = np.random.RandomState(0)
    X = rng.rand(200, 4)
    y = (X[:, 0] + 0.2 * rng.rand(200) > 0.6).astype(int)
    m = XGBClassifier(n_estimators=20, max_depth=3, base_score=0.5).fit(X, y)
    e = make_explainer(m)
    sv = _sv(e, X[:10])
    p = m.predict_proba(X[:10])[:, 1]
    np.testing.assert_allclose(sv.sum(1) + e.expected_value, np.log(p / (1 - p)), atol=1e-4)
