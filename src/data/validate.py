"""Automated validation checks that run before feature engineering and fail loudly."""

from __future__ import annotations

import pandas as pd

from src.utils.logging_setup import get_logger

log = get_logger(__name__)

REQUIRED = [
    "invoice",
    "stock_code",
    "description",
    "quantity",
    "invoice_date",
    "price",
    "customer_id",
    "country",
]


class ValidationError(Exception):
    """Raised when a data validation check fails."""


def check_null_rates(df: pd.DataFrame, limits: dict) -> dict:
    """Fail if any column's share of null values exceeds its configured threshold; return the measured rates."""
    rates = {c: float(df[c].isna().mean()) for c in limits if c in df.columns}
    bad = {c: round(r, 4) for c, r in rates.items() if r > limits[c]}
    if bad:
        raise ValidationError(f"null rate above threshold: {bad}")
    return rates


def validate_raw(df: pd.DataFrame, cfg: dict) -> None:
    """Schema, date-range and null-rate checks on the raw combined table."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValidationError(f"missing columns: {missing}")
    lo = pd.Timestamp(cfg["data"]["expected_min_date"])
    hi = pd.Timestamp(cfg["data"]["expected_max_date"])
    if df["invoice_date"].min() < lo or df["invoice_date"].max() > hi:
        raise ValidationError("invoice_date outside expected range")
    if df["quantity"].isna().any() or df["price"].isna().any():
        raise ValidationError("null quantity or price present")
    check_null_rates(df, cfg["data"].get("max_null_rate_raw", {}))
    log.info("raw validation passed rows=%d", len(df))


def validate_clean(df: pd.DataFrame, cfg: dict) -> None:
    """Checks on the cleaned customer-level transaction table."""
    validate_raw(df, cfg)
    if df["customer_id"].isna().any():
        raise ValidationError("cleaned table still has null customer_id")
    if (df["price"] <= 0).any():
        raise ValidationError("non-positive price survived cleaning")
    if not df["invoice_date"].is_monotonic_increasing:
        raise ValidationError("not chronologically ordered")
    log.info("clean validation passed rows=%d customers=%d", len(df), df["customer_id"].nunique())
