"""Item-item co-purchase recommender (collaborative filtering on purchase history).

Fitted on TRAINING customers' purchases made before the cutoff only. For a customer, each unseen product is scored by
the summed cosine similarity to the products the customer already bought. Customers without usable history fall back
to the most popular products. Offline check: for buyers in the test split, how often does a recommended product
appear among their purchases in the next 90 days, compared with a popularity-only baseline? That is a sanity check,
not proof of business impact (no live A/B test; the brief lists that as out of scope).
"""

from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from src.utils.config import load_config, rpath
from src.utils.logging_setup import get_logger

log = get_logger(__name__)

_RCFG = load_config()["recommender"]
MIN_CUSTOMERS_PER_PRODUCT: int = int(_RCFG["min_customers_per_product"])
NEIGHBORS: int = int(_RCFG["neighbors"])


def fit(sales_train: pd.DataFrame) -> dict[str, Any]:
    """Build the product neighbour lists and the popularity ranking from training customers' pre-cutoff sales."""
    pairs = sales_train[["customer_id", "stock_code"]].drop_duplicates()
    counts = pairs.groupby("stock_code")["customer_id"].nunique()
    keep = counts[counts >= MIN_CUSTOMERS_PER_PRODUCT].index
    pairs = pairs[pairs["stock_code"].isin(keep)]
    cust = pd.Index(pairs["customer_id"].unique())
    prod = pd.Index(sorted(keep))
    m = sparse.csr_matrix(
        (np.ones(len(pairs)), (cust.get_indexer(pairs["customer_id"]), prod.get_indexer(pairs["stock_code"]))),
        shape=(len(cust), len(prod)),
    )
    co = (m.T @ m).tocsr().astype(float)
    norm = np.sqrt(co.diagonal())
    neighbors: dict[str, list[tuple[str, float]]] = {}
    for i, code in enumerate(prod):
        row = co.getrow(i).toarray().ravel() / (norm[i] * norm + 1e-12)
        row[i] = 0.0
        top = np.argsort(-row)[:NEIGHBORS]
        neighbors[code] = [(prod[j], float(row[j])) for j in top if row[j] > 0]
    pop = counts[counts >= MIN_CUSTOMERS_PER_PRODUCT].sort_values(ascending=False)
    return {"neighbors": neighbors, "popularity": list(pop.index[:200])}


def recommend_for(model: dict[str, Any], bought: list[str], n: int = 5) -> list[tuple[str, float]]:
    """Top-n (stock_code, score) for a customer's purchased product list, excluding products already bought."""
    seen = set(bought)
    scores: dict[str, float] = {}
    for code in seen:
        for other, sim in model["neighbors"].get(code, []):
            if other not in seen:
                scores[other] = scores.get(other, 0.0) + sim
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:n]
    if len(ranked) < n:  # cold start or short history: fill with the most popular unseen products
        have = {c for c, _ in ranked}
        for code in model["popularity"]:
            if code not in seen and code not in have:
                ranked.append((code, 0.0))
                if len(ranked) == n:
                    break
    return ranked


def run(cfg: dict | None = None) -> dict[str, Any]:
    """Fit on training customers, evaluate offline on held-out test buyers, persist artifact and results."""
    cfg = cfg or load_config()
    tx = pd.read_parquet(rpath(cfg, "interim_transactions"))
    table = pd.read_parquet(rpath(cfg, "features"), columns=["data_split"])
    cutoff = pd.Timestamp(cfg["features"]["cutoff_date"])
    end = cutoff + pd.Timedelta(days=cfg["features"]["churn_horizon_days"])
    past = tx[(tx["invoice_date"] < cutoff) & ~tx["is_return"] & tx["customer_id"].isin(table.index)]
    fut = tx[(tx["invoice_date"] >= cutoff) & (tx["invoice_date"] < end) & ~tx["is_return"]]
    train_ids = set(table.index[table["data_split"] == "train"])
    model = fit(past[past["customer_id"].isin(train_ids)])
    desc = tx[tx["description"] != "NAN"].groupby("stock_code")["description"].agg(lambda s: s.value_counts().index[0])
    model["description"] = desc.to_dict()
    model["customer_products"] = {int(c): sorted(g.unique()) for c, g in past.groupby("customer_id")["stock_code"]}
    # offline evaluation on test buyers (their future purchases are never used to fit anything)
    test_ids = set(table.index[table["data_split"] == "test"])
    fut_sets = fut[fut["customer_id"].isin(test_ids)].groupby("customer_id")["stock_code"].agg(set)
    hits = pop_hits = n_eval = 0
    for cid, bought_future in fut_sets.items():
        history = model["customer_products"].get(int(cid), [])
        if not history:
            continue
        n_eval += 1
        rec = {c for c, _ in recommend_for(model, history, 10)}
        base = [c for c in model["popularity"] if c not in set(history)][:10]
        hits += bool(rec & bought_future)
        pop_hits += bool(set(base) & bought_future)
    result = {
        "method": "item-item cosine co-purchase, fitted on train customers",
        "k": 10,
        "test_buyers_evaluated": n_eval,
        "hit_rate_at_10": round(hits / max(n_eval, 1), 4),
        "popularity_baseline_hit_rate_at_10": round(pop_hits / max(n_eval, 1), 4),
        "products_in_model": len(model["neighbors"]),
        "note": "offline sanity check on the 90 days after the cutoff; no live A/B test",
    }
    art = rpath(cfg, "artifacts")
    joblib.dump(model, art / "recommender.joblib")
    (art / "recommender_eval.json").write_text(json.dumps(result, indent=2))
    log.info("recommender %s", result)
    return result


if __name__ == "__main__":
    run()
