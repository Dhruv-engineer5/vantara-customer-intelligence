"""Statistics that must be learned from TRAINING customers only, then persisted and applied everywhere.

Fitted here: IQR bounds for quantity and price, the stock-code description lookup, per-product median prices
(for the discount proxy) and the percentile reference distributions behind the engagement score.
Nothing here uses labels, validation/test customers or any row on or after the cutoff.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ENGAGEMENT_INPUTS = ("neg_recency", "frequency", "monetary_total")


def fit_reference(train_pre: pd.DataFrame, cfg: dict) -> dict:
    """Learn data-dependent cleaning/feature statistics from training customers' pre-cutoff rows."""
    k = cfg["cleaning"]["outlier_iqr_k"]
    sales = train_pre[~train_pre["is_return"]]
    q1, q3 = sales["quantity"].quantile([0.25, 0.75])
    p1, p3 = sales["price"].quantile([0.25, 0.75])
    desc = train_pre[train_pre["description"] != "NAN"].groupby("stock_code")["description"]
    return {
        "qty_bound": float(q3 + k * (q3 - q1)),
        "price_bound": float(p3 + k * (p3 - p1)),
        "desc_lookup": desc.agg(lambda s: s.value_counts().index[0]).to_dict(),
        "median_price": sales.groupby("stock_code")["price"].median().to_dict(),
        # StockCode frequency encoding: log(1 + number of TRAINING customers who bought the product)
        "stock_freq": np.log1p(sales.groupby("stock_code")["customer_id"].nunique()).to_dict(),
        "n_train_rows": int(len(train_pre)),
    }


def fit_engagement(ref: dict, train_feats: pd.DataFrame) -> dict:
    """Store sorted training values of recency (negated), frequency and monetary total."""
    ref["engagement"] = {
        "neg_recency": np.sort((-train_feats["recency_days"]).to_numpy(dtype=float)),
        "frequency": np.sort(train_feats["frequency"].to_numpy(dtype=float)),
        "monetary_total": np.sort(train_feats["monetary_total"].to_numpy(dtype=float)),
    }
    return ref


def percentile(values: pd.Series, sorted_ref: np.ndarray) -> np.ndarray:
    """Mid-rank percentile of each value within the training reference distribution (0..1)."""
    v = values.to_numpy(dtype=float)
    lo = np.searchsorted(sorted_ref, v, side="left")
    hi = np.searchsorted(sorted_ref, v, side="right")
    return (lo + hi) / 2.0 / len(sorted_ref)


def apply_reference(df: pd.DataFrame, ref: dict, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Apply description lookup and the quantity/price outlier rules using persisted training statistics.

    A line above the IQR bound is a legitimate bulk/high-price line and is kept, unless it also reaches the domain
    floor (quantity >= bulk_quantity_floor, price >= price_error_floor), in which case it is treated as an entry error.
    """
    c = cfg["cleaning"]
    df = df.copy()
    df["description"] = df["stock_code"].map(ref["desc_lookup"]).fillna(df["description"])
    sales = ~df["is_return"]
    q_flag = sales & (df["quantity"] > ref["qty_bound"])
    p_flag = sales & (df["price"] > ref["price_bound"])
    q_err = q_flag & (df["quantity"] >= c["bulk_quantity_floor"])
    p_err = p_flag & (df["price"] >= c["price_error_floor"])
    err = q_err | p_err
    rep = {
        "iqr_quantity_bound_train_fit": ref["qty_bound"],
        "iqr_price_bound_train_fit": ref["price_bound"],
        "rows_qty_flagged_bulk_kept": int((q_flag & ~q_err).sum()),
        "rows_price_flagged_kept": int((p_flag & ~p_err).sum()),
        "rows_quantity_entry_error_dropped": int(q_err.sum()),
        "rows_price_entry_error_dropped": int(p_err.sum()),
    }
    df = df[~err].assign(line_total=lambda d: d["quantity"] * d["price"])
    return df.reset_index(drop=True), rep


def save_reference(ref: dict, path: Path) -> None:
    """Persist for inference and audit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(ref, path)


def load_reference(path: Path) -> dict:
    """Load persisted reference statistics."""
    return joblib.load(path)
