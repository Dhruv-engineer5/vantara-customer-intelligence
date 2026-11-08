"""Cleaning rules for Online Retail II. Every rule is counted in a cleaning report."""

from __future__ import annotations

import json

import pandas as pd

from src.data.load import load_raw
from src.data.validate import validate_clean, validate_raw
from src.utils.config import load_config, rpath
from src.utils.logging_setup import get_logger

log = get_logger(__name__)


def clean_transactions(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Apply the rule-based (not data-fitted) cleaning steps and report what each rule did."""
    rep: dict = {"rows_in": int(len(df))}
    df = df.copy()
    df["is_return"] = df["invoice"].astype(str).str.startswith("C") | (df["quantity"] < 0)
    # Missing Customer ID: excluded from customer-level modelling (kept only for product-level work).
    rep["rows_missing_customer_id"] = int(df["customer_id"].isna().sum())
    df = df[df["customer_id"].notna()]
    # Zero/negative price rows are adjustments and manual corrections: documented rule = drop.
    rep["rows_nonpositive_price"] = int((df["price"] <= 0).sum())
    df = df[df["price"] > 0]
    # Non-product administrative stock codes are not products.
    admin = set(cfg["data"]["non_product_stockcodes"])
    is_admin = df["stock_code"].astype(str).str.upper().isin(admin)
    rep["rows_non_product_stockcode"] = int(is_admin.sum())
    df = df[~is_admin]
    # Exact duplicate invoice lines (all fields). Repeated SKU lines with differing quantity stay.
    before = len(df)
    df = df.drop_duplicates()
    rep["rows_exact_duplicates"] = int(before - len(df))
    # Description text normalisation is row-wise (no fitted statistic). The stock-code lookup and the IQR outlier
    # bounds are data-dependent, so they are fitted on training customers only in src/data/reference.py.
    desc = df["description"].astype(str).str.upper().str.strip().str.replace(r"\s+", " ", regex=True)
    df = df.assign(description=desc)
    df = df.assign(line_total=df["quantity"] * df["price"])
    df = df.sort_values("invoice_date", kind="stable").reset_index(drop=True)
    df["customer_id"] = df["customer_id"].astype(int)
    rep["rows_out"] = int(len(df))
    rep["customers_out"] = int(df["customer_id"].nunique())
    return df, rep


def run(cfg: dict | None = None) -> pd.DataFrame:
    """Load, validate, clean, validate again, and persist interim data and the cleaning report."""
    cfg = cfg or load_config()
    raw = load_raw(cfg)
    validate_raw(raw, cfg)
    df, rep = clean_transactions(raw, cfg)
    validate_clean(df, cfg)
    out = rpath(cfg, "interim_rule_cleaned")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    rpath(cfg, "interim_rejects").write_text(json.dumps(rep, indent=2))
    log.info("cleaning report=%s", json.dumps(rep))
    return df


if __name__ == "__main__":
    run()
