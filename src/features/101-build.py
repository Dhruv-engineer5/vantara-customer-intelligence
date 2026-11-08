"""Customer-level feature engineering with strict point-in-time cutoff discipline.

Every feature is computed ONLY from rows with invoice_date < cutoff. Targets use the window
[cutoff, cutoff + horizon). `assert_no_leakage` and the unit tests enforce this.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data.reference import percentile
from src.utils.config import load_config, rpath
from src.utils.logging_setup import get_logger

log = get_logger(__name__)

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "christmas_seasonal": ["CHRISTMAS", "XMAS", "SNOW", "ADVENT", "EASTER", "HALLOWEEN"],
    "kitchen_dining": ["MUG", "CUP", "TEA", "CAKE", "PLATE", "BOWL", "KITCHEN", "JAR", "SPOON", "BAKING"],
    "bags_storage": ["BAG", "BOX", "TIN", "BASKET", "STORAGE", "CASE"],
    "lights_candles": ["LIGHT", "CANDLE", "LANTERN", "LAMP", "T-LIGHT", "HOLDER"],
    "home_decor": ["HEART", "FRAME", "CUSHION", "SIGN", "CLOCK", "MIRROR", "WALL", "DOORMAT", "VINTAGE"],
    "toys_party": ["TOY", "GAME", "PARTY", "BIRTHDAY", "BALLOON", "CHILDREN", "KIDS", "PAPER"],
    "garden_outdoor": ["GARDEN", "BIRD", "PLANT", "FLOWER", "WATERING"],
}
CATEGORIES = list(CATEGORY_KEYWORDS) + ["other"]
_FCFG = load_config()["features"]  # thresholds and windows live in config/config.yaml
COUNTRY_GROUPS: list[str] = list(_FCFG["country_groups"])
NICHE_MAX_CUSTOMERS: int = int(_FCFG["niche_max_customers"])
DISCOUNT_PRICE_RATIO: float = float(_FCFG["discount_price_ratio"])
RECENT_WINDOWS: tuple[int, ...] = tuple(int(w) for w in _FCFG["recent_windows_days"])
TREND_WINDOW_DAYS: int = int(_FCFG["trend_window_days"])
assert 90 in RECENT_WINDOWS, "features.recent_windows_days must contain 90"
FEATURE_COLUMNS: list[str] = []  # filled at bottom after the builder defines the schema


def assign_category(desc: pd.Series) -> pd.Series:
    """Keyword rule mapping product descriptions to coarse categories (documented proxy: the
    source data has no category field)."""
    out = pd.Series("other", index=desc.index, dtype=object)
    d = desc.fillna("").astype(str).str.upper()
    for cat in reversed(list(CATEGORY_KEYWORDS)):
        pat = "|".join(CATEGORY_KEYWORDS[cat])
        out[d.str.contains(pat, regex=True)] = cat
    return out


def _slope(counts: pd.DataFrame) -> pd.Series:
    """Per-row OLS slope of a wide monthly-count matrix against month index."""
    x = np.arange(counts.shape[1], dtype=float)
    x = x - x.mean()
    y = counts.to_numpy(dtype=float)
    return pd.Series((y - y.mean(axis=1, keepdims=True)) @ x / (x @ x), index=counts.index)


def build_features(
    tx: pd.DataFrame, cutoff: str | pd.Timestamp, horizon_days: int = 90, ref: dict | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (features_df, targets_df) for customers with at least one purchase before cutoff.

    features_df is indexed by customer_id; targets_df has churn, future_spend.
    `ref` holds statistics fitted on training customers only (product median prices, engagement percentile
    reference). If it is None they are computed from the rows passed in; that fallback exists for unit tests and
    notebooks and must not be used for model evaluation.
    """
    cutoff = pd.Timestamp(cutoff)
    end = cutoff + pd.Timedelta(days=horizon_days)
    past = tx[tx["invoice_date"] < cutoff]
    fut = tx[(tx["invoice_date"] >= cutoff) & (tx["invoice_date"] < end)]
    sales = past[~past["is_return"]]
    if sales.empty:
        raise ValueError("no sales rows before cutoff")
    inv = (
        sales.groupby(["customer_id", "invoice"])
        .agg(
            date=("invoice_date", "max"),
            amount=("line_total", "sum"),
            items=("quantity", "sum"),
            lines=("stock_code", "size"),
        )
        .reset_index()
        .sort_values(["customer_id", "date"])
    )
    g = inv.groupby("customer_id")
    feats = pd.DataFrame(index=g.size().index)
    feats["recency_days"] = (cutoff - g["date"].max()).dt.days
    feats["frequency"] = g.size()
    feats["monetary_total"] = g["amount"].sum()
    feats["monetary_avg"] = g["amount"].mean()
    feats["tenure_days"] = (g["date"].max() - g["date"].min()).dt.days
    feats["avg_basket_size"] = g["items"].mean()
    feats["avg_lines_per_order"] = g["lines"].mean()
    feats["n_unique_products"] = sales.groupby("customer_id")["stock_code"].nunique()
    inv["gap"] = g["date"].diff().dt.total_seconds() / 86400.0
    gg = inv.groupby("customer_id")["gap"]
    feats["gap_mean_days"] = gg.mean().fillna(0.0)
    feats["gap_var_days"] = gg.var().fillna(0.0)
    # purchase frequency trend: slope of monthly order counts over the 12 months before cutoff
    last12 = inv[inv["date"] >= cutoff - pd.Timedelta(days=TREND_WINDOW_DAYS)].copy()  # only the window before cutoff
    last12["m"] = (cutoff - last12["date"]).dt.days // 30
    wide = last12.pivot_table(index="customer_id", columns="m", values="invoice", aggfunc="count", fill_value=0)
    wide = wide.reindex(columns=range(11, -1, -1), fill_value=0).reindex(feats.index, fill_value=0)
    feats["freq_trend"] = _slope(wide)
    # recent-activity windows (strictly before cutoff): orders and spend in the last 30/90/180 days
    for w in RECENT_WINDOWS:
        recent = inv[inv["date"] >= cutoff - pd.Timedelta(days=w)]
        feats[f"orders_last_{w}d"] = recent.groupby("customer_id").size().reindex(feats.index).fillna(0)
        feats[f"spend_last_{w}d"] = recent.groupby("customer_id")["amount"].sum().reindex(feats.index).fillna(0.0)
    feats["recent_share_90d"] = feats["orders_last_90d"] / feats["frequency"]
    feats["days_since_first_purchase"] = (cutoff - g["date"].min()).dt.days
    # seasonal concentration: share of spend in the customer's single biggest calendar month
    sales_m = sales.assign(mo=sales["invoice_date"].dt.month)
    by_mo = sales_m.pivot_table(index="customer_id", columns="mo", values="line_total", aggfunc="sum", fill_value=0.0)
    feats["seasonal_concentration"] = (
        (by_mo.max(axis=1) / by_mo.sum(axis=1).replace(0, np.nan)).reindex(feats.index).fillna(1.0)
    )
    # category affinity: share of lines per category
    cat = assign_category(sales["description"])
    aff = pd.crosstab(sales["customer_id"], cat).reindex(columns=CATEGORIES, fill_value=0)
    aff = aff.div(aff.sum(axis=1), axis=0).add_prefix("aff_")
    feats = feats.join(aff)
    # return rate: share of the customer's lines that are returns
    rr = past.groupby("customer_id")["is_return"].mean()
    feats["return_rate"] = rr.reindex(feats.index).fillna(0.0)
    # discount sensitivity: share of lines priced under 90% of the product's median price (past data only)
    if ref is not None and "median_price" in ref:
        med = sales["stock_code"].map(ref["median_price"]).fillna(sales["price"])  # unseen product: not discounted
    else:
        med = sales.groupby("stock_code")["price"].transform("median")
    feats["discount_sensitivity"] = (
        (sales["price"] < DISCOUNT_PRICE_RATIO * med)
        .groupby(sales["customer_id"])
        .mean()
        .reindex(feats.index)
        .fillna(0.0)
    )
    # StockCode frequency encoding (high cardinality): popularity of each product among TRAINING customers
    if ref is not None and "stock_freq" in ref:
        pop = sales["stock_code"].map(ref["stock_freq"]).fillna(0.0)  # unseen product: popularity 0
    else:
        pop = np.log1p(sales.groupby("stock_code")["customer_id"].transform("nunique"))
    g = pop.groupby(sales["customer_id"])
    feats["product_popularity_mean"] = g.mean().reindex(feats.index).fillna(0.0)
    feats["niche_product_share"] = (
        (pop < np.log1p(NICHE_MAX_CUSTOMERS)).groupby(sales["customer_id"]).mean().reindex(feats.index).fillna(0.0)
    )
    # country one-hot (low cardinality)
    ctry = past.groupby("customer_id")["country"].agg(lambda s: s.value_counts().index[0]).reindex(feats.index)
    for c in COUNTRY_GROUPS:
        feats["country_" + c.lower().replace(" ", "_")] = (ctry == c).astype(int)
    feats["country_other"] = (~ctry.isin(COUNTRY_GROUPS)).astype(int)
    # historical CLV = net revenue (sales minus returns) over the customer's observed history
    net = past.groupby("customer_id")["line_total"].sum()
    feats["clv_historical"] = net.reindex(feats.index).fillna(0.0)
    # engagement score: mean percentile of (inverted) recency, frequency, monetary
    if ref is not None and "engagement" in ref:  # percentiles against the TRAINING reference distributions
        e = ref["engagement"]
        r = pd.Series(percentile(-feats["recency_days"], e["neg_recency"]), index=feats.index)
        f = pd.Series(percentile(feats["frequency"], e["frequency"]), index=feats.index)
        m = pd.Series(percentile(feats["monetary_total"], e["monetary_total"]), index=feats.index)
    else:
        r = (-feats["recency_days"]).rank(pct=True)
        f = feats["frequency"].rank(pct=True)
        m = feats["monetary_total"].rank(pct=True)
    feats["engagement_score"] = 100 * (r + f + m) / 3
    # targets from the future window only
    fut_sales = fut[~fut["is_return"]]
    bought = set(fut_sales["customer_id"].unique())
    targets = pd.DataFrame(index=feats.index)
    targets["churn"] = (~feats.index.isin(bought)).astype(int)
    # next-purchase timing label for the sequence model: any purchase within 30 days after the cutoff
    soon = fut_sales[fut_sales["invoice_date"] < cutoff + pd.Timedelta(days=30)]
    targets["next30"] = feats.index.isin(set(soon["customer_id"].unique())).astype(int)
    # next-purchase timing and category (among customers who buy in the horizon; others stay censored/-1)
    fs = fut_sales.sort_values(["invoice_date", "invoice"])
    first_inv = fs.groupby("customer_id")["invoice"].first()
    first_day = fs.groupby("customer_id")["invoice_date"].min()
    days = ((first_day - pd.Timestamp(cutoff)).dt.total_seconds() / 86400.0).clip(lower=0, upper=horizon_days)
    targets["days_to_next"] = days.reindex(feats.index).fillna(float(horizon_days))
    lines = fs[fs["invoice"] == fs["customer_id"].map(first_inv)]
    dom = (
        lines.assign(cat=assign_category(lines["description"]))
        .groupby("customer_id")["cat"]
        .agg(lambda c: c.value_counts().index[0])
    )
    targets["next_category"] = (
        dom.reindex(feats.index).map({c: i for i, c in enumerate(CATEGORIES)}).fillna(-1).astype(int)
    )
    targets["future_spend"] = (
        fut.groupby("customer_id")["line_total"].sum().reindex(feats.index).fillna(0.0).clip(lower=0)
    )
    feats.index.name = targets.index.name = "customer_id"
    return feats, targets


def build_sequences(tx: pd.DataFrame, cutoff: Any, ids: Any, length: int = 8) -> np.ndarray:
    """Last `length` invoices per customer before cutoff as (log1p amount, log1p gap days, log1p items) plus a
    one-hot of the invoice's dominant product category (the brief asks for amount, gap and category).
    Left-padded with zeros. Shape (n_customers, length, 3 + len(CATEGORIES)), row order follows `ids`."""
    cutoff = pd.Timestamp(cutoff)
    sales = tx[(tx["invoice_date"] < cutoff) & (~tx["is_return"])]
    inv = (
        sales.groupby(["customer_id", "invoice"])
        .agg(
            date=("invoice_date", "max"),
            amount=("line_total", "sum"),
            items=("quantity", "sum"),
        )
        .reset_index()
        .sort_values(["customer_id", "date"])
    )
    line_cat = sales.assign(cat=assign_category(sales["description"]))
    dom = line_cat.groupby(["customer_id", "invoice", "cat"])["quantity"].sum().reset_index()
    dom = dom.sort_values("quantity", ascending=False).drop_duplicates(["customer_id", "invoice"])
    inv = inv.merge(dom[["customer_id", "invoice", "cat"]], on=["customer_id", "invoice"], how="left")
    inv["gap"] = inv.groupby("customer_id")["date"].diff().dt.total_seconds().div(86400).fillna(0.0)
    out = np.zeros((len(ids), length, 3 + len(CATEGORIES)), dtype=np.float32)
    cat_idx = {c: i for i, c in enumerate(CATEGORIES)}
    pos = {c: i for i, c in enumerate(ids)}
    for cid, grp in inv.groupby("customer_id"):
        if cid not in pos:
            continue
        a = grp.tail(length)
        onehot = np.zeros((len(a), len(CATEGORIES)), dtype=np.float32)
        for j, c in enumerate(a["cat"]):
            onehot[j, cat_idx.get(c, cat_idx.get("other", 0))] = 1.0
        arr = np.column_stack(
            [np.log1p(a["amount"].clip(lower=0)), np.log1p(a["gap"]), np.log1p(a["items"].clip(lower=0)), onehot]
        )
        out[pos[cid], length - len(a) :, :] = arr
    return out


def assert_no_leakage(features_index_dates: pd.Series, cutoff: Any) -> None:
    """Raise if any date used for features is not strictly before the cutoff."""
    if (features_index_dates >= pd.Timestamp(cutoff)).any():
        raise AssertionError("feature rows contain data on/after the cutoff")


def split_ids(ids: np.ndarray, churn: np.ndarray, cfg: dict) -> pd.Series:
    """Stratified train/val/test assignment of customers (same logic as models.common.make_split)."""
    from sklearn.model_selection import train_test_split

    s, seed = cfg["split"], cfg["seed"]
    idx = np.arange(len(ids))
    tr, rest = train_test_split(idx, train_size=s["train"], stratify=churn, random_state=seed)
    rel = s["val"] / (s["val"] + s["test"])
    va, te = train_test_split(rest, train_size=rel, stratify=churn[rest], random_state=seed)
    out = np.empty(len(ids), dtype=object)
    out[tr], out[va], out[te] = "train", "val", "test"
    return pd.Series(out, index=ids, name="data_split")


def run(cfg: dict | None = None) -> pd.DataFrame:
    """Split customers, fit reference statistics on TRAIN customers only, apply them, build and persist features."""
    import json

    from src.data.reference import apply_reference, fit_engagement, fit_reference, save_reference

    cfg = cfg or load_config()
    tx_rule = pd.read_parquet(rpath(cfg, "interim_rule_cleaned"))
    fc = cfg["features"]
    cutoff = pd.Timestamp(fc["cutoff_date"])
    # 1) customer set and churn label come from row-level facts only (no fitted statistic involved)
    _, lab = build_features(tx_rule, cutoff, fc["churn_horizon_days"])
    split = split_ids(lab.index.to_numpy(), lab["churn"].to_numpy(), cfg)
    train_ids = split.index[split == "train"]
    # 2) fit data-dependent statistics on training customers' pre-cutoff rows only
    train_pre = tx_rule[(tx_rule["invoice_date"] < cutoff) & tx_rule["customer_id"].isin(train_ids)]
    ref = fit_reference(train_pre, cfg)
    tx, rep = apply_reference(tx_rule, ref, cfg)
    # 3) features with train-fitted product medians, then the train-fitted engagement percentile reference
    feats, _ = build_features(tx, cutoff, fc["churn_horizon_days"], ref)
    ref = fit_engagement(ref, feats.loc[feats.index.intersection(train_ids)])
    feats, targets = build_features(tx, cutoff, fc["churn_horizon_days"], ref)
    save_reference(ref, rpath(cfg, "reference_stats"))
    table = feats.join(targets)
    table["data_split"] = split.reindex(table.index)
    assert table["data_split"].notna().all(), "every modelled customer must have a split assignment"
    table["country"] = (
        tx[tx["invoice_date"] < cutoff]
        .groupby("customer_id")["country"]
        .agg(lambda s: s.value_counts().index[0])
        .reindex(table.index)
    )
    tx.to_parquet(rpath(cfg, "interim_transactions"), index=False)
    rp = rpath(cfg, "interim_rejects")
    full = json.loads(rp.read_text()) if rp.exists() else {}
    full.update(rep)
    full["rows_out"], full["customers_out"] = int(len(tx)), int(tx["customer_id"].nunique())
    full["reference_fit_customers"], full["reference_fit_rows"] = int(len(train_ids)), ref["n_train_rows"]
    rp.write_text(json.dumps(full, indent=2))
    out = rpath(cfg, "features")
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out)
    seq = build_sequences(tx, fc["cutoff_date"], table.index.to_list(), fc["sequence_length"])
    np.savez_compressed(rpath(cfg, "sequences"), seq=seq, ids=table.index.to_numpy())
    log.info("features rows=%d cols=%d churn_rate=%.3f", len(table), table.shape[1], table["churn"].mean())
    return table


FEATURE_COLUMNS = (
    ["recency_days"]
    + [f"orders_last_{w}d" for w in RECENT_WINDOWS]
    + [f"spend_last_{w}d" for w in RECENT_WINDOWS]
    + [
        "recent_share_90d",
        "days_since_first_purchase",
        "frequency",
        "monetary_total",
        "monetary_avg",
        "tenure_days",
        "avg_basket_size",
        "avg_lines_per_order",
        "n_unique_products",
        "gap_mean_days",
        "gap_var_days",
        "freq_trend",
        "seasonal_concentration",
    ]
    + ["aff_" + c for c in CATEGORIES]
    + ["return_rate", "discount_sensitivity", "product_popularity_mean", "niche_product_share"]
    + ["country_" + c.lower().replace(" ", "_") for c in COUNTRY_GROUPS]
    + ["country_other"]
    + ["clv_historical", "engagement_score"]
)

if __name__ == "__main__":
    run()
