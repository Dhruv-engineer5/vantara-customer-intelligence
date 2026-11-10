"""Processing pipeline for user-uploaded datasets (private workspaces).

Accepts a CSV in the same schema as the public training data (Online Retail II columns:
Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country),
cleans it with the same rule-based steps, builds customer features with the TRAINING
reference statistics (no statistics are fitted on user data), scores with the deployed
models, and returns rows ready for the user_* tables.

Honesty rules enforced here: hard caps keep free-tier memory honest, and uploads that are
too small or malformed are rejected with clear messages instead of being scored anyway.
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.data.reference import apply_reference
from src.features.build import FEATURE_COLUMNS, build_features
from src.recommend.recommender import fit as fit_recommender
from src.utils.logging_setup import get_logger

log = get_logger(__name__)

MAX_FILE_BYTES = 5_000_000
MAX_ROWS = 50_000
MIN_ROWS = 200
MIN_CUSTOMERS = 20
MIN_HISTORY_DAYS = 30
MAX_DATASETS_PER_USER = 3
# Global guard so public signups cannot fill the 1 GB free database: total rows stored
# across ALL user datasets (50k rows is roughly 25 MB including features and scores).
GLOBAL_MAX_STORED_ROWS = 1_000_000

# Accept both the raw Online Retail II headers and the internal snake-case names.
COLUMN_ALIASES = {
    "invoice": "invoice",
    "stockcode": "stock_code",
    "stock_code": "stock_code",
    "description": "description",
    "quantity": "quantity",
    "invoicedate": "invoice_date",
    "invoice_date": "invoice_date",
    "price": "price",
    "customer id": "customer_id",
    "customer_id": "customer_id",
    "customerid": "customer_id",
    "country": "country",
}
REQUIRED = ["invoice", "stock_code", "quantity", "invoice_date", "price", "customer_id", "country"]
EXPECTED_COLUMNS_TEXT = "Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country"


class UploadError(ValueError):
    """A user-facing validation failure (422)."""


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in COLUMN_ALIASES:
            renamed[col] = COLUMN_ALIASES[key]
    return df.rename(columns=renamed)


def _read_and_clean(raw: bytes, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Parse the CSV and apply the same rule-based cleaning as the training pipeline."""
    try:
        df = pd.read_csv(io.BytesIO(raw), dtype={"Invoice": str, "StockCode": str})
    except Exception as exc:  # noqa: BLE001
        raise UploadError(f"could not read the CSV: {exc}") from exc
    df = _normalise_columns(df)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise UploadError(
            f"missing columns: {missing}. The CSV must have these columns (same layout as the sample data): "
            f"{EXPECTED_COLUMNS_TEXT}"
        )
    rep: dict[str, Any] = {"rows_in": int(len(df))}
    if len(df) > MAX_ROWS:
        raise UploadError(f"the CSV has {len(df):,} rows; the limit is {MAX_ROWS:,} rows")
    df = df.copy()
    df["invoice"] = df["invoice"].astype(str).str.strip()
    df["stock_code"] = df["stock_code"].astype(str).str.strip()
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce", utc=True).dt.tz_localize(None)
    if "description" in df.columns:
        df["description"] = df["description"].astype(str)
    else:
        df["description"] = ""
    df["customer_id"] = df["customer_id"].astype(str).str.strip().replace({"nan": np.nan, "": np.nan, "None": np.nan})
    df["country"] = df["country"].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
    bad = df[["quantity", "price", "invoice_date"]].isna().any(axis=1)
    rep["rows_unparseable"] = int(bad.sum())
    df = df[~bad]
    # Same documented rules as src/data/clean.py, but customer ids stay strings (uploads are not the UCI dataset).
    df["is_return"] = df["invoice"].str.startswith("C") | (df["quantity"] < 0)
    rep["rows_missing_customer_id"] = int(df["customer_id"].isna().sum())
    df = df[df["customer_id"].notna()]
    rep["rows_nonpositive_price"] = int((df["price"] <= 0).sum())
    df = df[df["price"] > 0]
    admin = set(cfg["data"]["non_product_stockcodes"])
    df = df[~df["stock_code"].str.upper().isin(admin)]
    before = len(df)
    df = df.drop_duplicates()
    rep["rows_exact_duplicates"] = int(before - len(df))
    desc = df["description"].str.upper().str.strip().str.replace(r"\s+", " ", regex=True)
    df = df.assign(description=desc)
    df = df.assign(line_total=df["quantity"] * df["price"])
    df = df.sort_values("invoice_date", kind="stable").reset_index(drop=True)
    rep["rows_out"] = int(len(df))
    rep["customers_out"] = int(df["customer_id"].nunique())
    return df, rep


def _check_usable(tx: pd.DataFrame) -> None:
    """Reject uploads too small for the models to say anything honest about."""
    if len(tx) < MIN_ROWS:
        raise UploadError(f"only {len(tx)} usable rows after cleaning; need at least {MIN_ROWS:,}")
    if tx["customer_id"].nunique() < MIN_CUSTOMERS:
        raise UploadError(
            f"only {tx['customer_id'].nunique()} customers after cleaning; need at least {MIN_CUSTOMERS}"
        )
    span = (tx["invoice_date"].max() - tx["invoice_date"].min()).days
    if span < MIN_HISTORY_DAYS:
        raise UploadError(
            f"the data spans {span} days; the models need at least {MIN_HISTORY_DAYS} days of history per dataset"
        )
    if (tx["invoice_date"].max() - pd.Timestamp.now()) > pd.Timedelta(days=7):
        raise UploadError("invoice dates are in the future; check the InvoiceDate column")


def process_upload(state: Any, raw: bytes, cfg: dict) -> dict:
    """Clean, feature-build and score an uploaded CSV with the deployed models.

    Returns a dict with rows for the user_* tables plus the cleaning report.
    Raises UploadError (422) for user-fixable problems.
    """
    tx, rep = _read_and_clean(raw, cfg)
    _check_usable(tx)
    ref = state.ref
    tx, ref_rep = apply_reference(tx, ref, cfg)
    rep.update(ref_rep)
    # Features use everything strictly before the cutoff; the cutoff is just past the last invoice
    # so the customer's whole uploaded history counts (there is no future window to label against).
    cutoff = tx["invoice_date"].max() + pd.Timedelta(days=1)
    feats, _ = build_features(tx, cutoff, cfg["features"]["churn_horizon_days"], ref)
    ids = feats.index.astype(str).tolist()
    X = feats[FEATURE_COLUMNS].to_numpy(dtype=float)
    from api.deps import score_matrix

    sc = score_matrix(state, X)
    from src.segmentation.cluster import SEG_COLS

    seg_cols = [FEATURE_COLUMNS.index(c) for c in SEG_COLS]
    seg_frame = pd.DataFrame(X[:, seg_cols], columns=SEG_COLS)
    for c in ("frequency", "monetary_total", "avg_basket_size", "recency_days"):
        seg_frame[c] = np.log1p(seg_frame[c].clip(lower=0))
    seg_labels = state.seg["kmeans"].predict(state.seg["scaler"].transform(seg_frame.to_numpy()))
    segs = [state.seg["names"][int(i)] for i in seg_labels]
    country = (
        tx.groupby("customer_id")["country"].agg(lambda s: s.value_counts().index[0]).reindex(feats.index).tolist()
    )
    now = datetime.utcnow()
    model_name = state.metadata["production_churn_model"]
    sales = tx[~tx["is_return"]]
    reco = fit_recommender(sales[["customer_id", "stock_code"]])
    bought = sales.groupby("customer_id")["stock_code"].agg(set)
    reco["customer_products"] = {str(k): sorted(v) for k, v in bought.items()}
    reco["description"] = (
        sales.assign(d=sales["description"].str.title())
        .groupby("stock_code")["d"]
        .agg(lambda s: s.value_counts().index[0])
        .to_dict()
    )
    inv = (
        sales.groupby(["customer_id", "invoice"], observed=True)
        .agg(date=("invoice_date", "max"), amount=("line_total", "sum"), items=("quantity", "sum"))
        .reset_index()
    )
    ret = (
        tx[tx["is_return"]]
        .groupby(["customer_id", "invoice"], observed=True)
        .agg(date=("invoice_date", "max"), amount=("line_total", "sum"), items=("quantity", "sum"))
        .reset_index()
    )
    inv["is_return"] = False
    ret["is_return"] = True
    invoices = pd.concat([inv, ret], ignore_index=True)
    out = {
        "report": rep,
        "n_rows": int(len(tx)),
        "n_customers": int(len(feats)),
        "first_date": tx["invoice_date"].min().to_pydatetime(),
        "last_date": tx["invoice_date"].max().to_pydatetime(),
        "recommender": reco,
        "customers": [
            {
                "customer_id": cid,
                "country": str(cty),
                "recency_days": float(feats.loc[cid0, "recency_days"]),
                "frequency": float(feats.loc[cid0, "frequency"]),
                "monetary_total": float(feats.loc[cid0, "monetary_total"]),
                "clv_historical": float(feats.loc[cid0, "clv_historical"]),
                "engagement_score": float(feats.loc[cid0, "engagement_score"]),
                "features": {c: float(feats.loc[cid0, c]) for c in FEATURE_COLUMNS},
            }
            for cid, cid0, cty in zip(ids, feats.index, country)
        ],
        "predictions": [
            {
                "customer_id": cid,
                "churn_probability": float(r.churn_probability),
                "clv_predicted": float(r.clv_predicted_90d),
                "is_anomaly": bool(r.is_anomaly),
                "model_name": model_name,
                "scored_at": now,
            }
            for cid, r in zip(ids, sc.itertuples())
        ],
        "segments": [
            {"customer_id": cid, "segment_id": int(label), "segment_name": name}
            for cid, label, name in zip(ids, seg_labels, segs)
        ],
        "transactions": [
            {
                "invoice": str(r.invoice),
                "customer_id": str(r.customer_id),
                "invoice_date": r.date.to_pydatetime(),
                "amount": float(r.amount),
                "items": float(r.items),
                "is_return": bool(r.is_return),
            }
            for r in invoices.itertuples()
        ],
    }
    log.info("upload processed rows=%d customers=%d report=%s", out["n_rows"], out["n_customers"], rep)
    return out
