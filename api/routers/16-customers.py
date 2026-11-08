"""Customer lookup, leaderboard, explanation and segment endpoints."""

from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.deps import risk_level
from src.db.models import Customer, Prediction, Segment
from src.explainability.explain import explain_row, lime_row
from src.features.build import FEATURE_COLUMNS

router = APIRouter(tags=["customers"])


def _row(c: Customer, p: Prediction, s: Segment) -> dict:
    return {
        "customer_id": c.customer_id,
        "country": c.country,
        "recency_days": c.recency_days,
        "frequency": c.frequency,
        "monetary_total": round(c.monetary_total, 2),
        "clv_historical": round(c.clv_historical, 2),
        "engagement_score": round(c.engagement_score, 1),
        "churn_probability": round(p.churn_probability, 4),
        "churn_risk_level": risk_level(p.churn_probability),
        "purchase_probability_90d": round(1.0 - p.churn_probability, 4),
        "clv_predicted_90d": round(p.clv_predicted, 2),
        "clv_total_projected": round(c.clv_historical + p.clv_predicted, 2),
        "is_anomaly": p.is_anomaly,
        "segment_name": s.segment_name,
        "data_split": p.data_split,
    }


def _base() -> Any:
    return (
        select(Customer, Prediction, Segment)
        .join(Prediction, Prediction.customer_id == Customer.customer_id)
        .join(Segment, Segment.customer_id == Customer.customer_id)
    )


@router.get("/customers")
def list_customers(
    request: Request,
    segment: str | None = None,
    country: str | None = None,
    value_tier: str | None = Query(None, pattern="^(high|medium|low)$"),
    risk: str | None = Query(None, pattern="^(high|moderate|low)$"),
    customer_id: int | None = None,
    limit: int = Query(200, ge=1, le=6000),
) -> list[dict]:
    """Filterable customer table (segment, country, value tier, risk level, id)."""
    q = _base()
    if segment:
        q = q.where(Segment.segment_name == segment)
    if country:
        q = q.where(Customer.country == country)
    if customer_id is not None:
        q = q.where(Customer.customer_id == customer_id)
    if risk:
        lo, hi = {"high": (0.7, 1.01), "moderate": (0.4, 0.7), "low": (0.0, 0.4)}[risk]
        q = q.where(Prediction.churn_probability >= lo, Prediction.churn_probability < hi)
    with Session(request.app.state.s.engine) as sess:
        rows = [_row(*r) for r in sess.execute(q.order_by(Customer.customer_id)).all()]
    if value_tier:  # tiers are terciles of historical CLV across all customers
        with Session(request.app.state.s.engine) as sess:
            allv = np.array([v for (v,) in sess.execute(select(Customer.clv_historical)).all()])
        lo_q, hi_q = np.quantile(allv, [1 / 3, 2 / 3])
        pick = {"high": lambda v: v >= hi_q, "medium": lambda v: lo_q <= v < hi_q, "low": lambda v: v < lo_q}[
            value_tier
        ]
        rows = [r for r in rows if pick(r["clv_historical"])]
    return rows[:limit]


@router.get("/customers/leaderboard")
def leaderboard(request: Request, n: int = Query(25, ge=1, le=200)) -> list[dict]:
    """Churn leaderboard ranked by value at risk (churn probability x historical CLV)."""
    with Session(request.app.state.s.engine) as sess:
        rows = [_row(*r) for r in sess.execute(_base()).all()]
    for r in rows:
        r["value_at_risk"] = round(r["churn_probability"] * max(r["clv_historical"], 0), 2)
    return sorted(rows, key=lambda r: -r["value_at_risk"])[:n]


@router.get("/customers/{customer_id}")
def get_customer(request: Request, customer_id: int) -> dict:
    """One customer with stored prediction and segment."""
    with Session(request.app.state.s.engine) as sess:
        r = sess.execute(_base().where(Customer.customer_id == customer_id)).first()
    if not r:
        raise HTTPException(404, f"customer {customer_id} not found")
    return _row(*r)


@router.get("/customers/{customer_id}/explain")
def explain_customer(request: Request, customer_id: int) -> dict:
    """SHAP factors and a plain-language sentence for one customer."""
    st = request.app.state.s
    with Session(st.engine) as sess:
        c = sess.get(Customer, customer_id)
    if c is None:
        raise HTTPException(404, f"customer {customer_id} not found")
    x = np.array([c.features[f] for f in FEATURE_COLUMNS], dtype=float)
    return {"customer_id": customer_id, **explain_row(st.churn_model, x, st.explainer)}


@router.get("/customers/{customer_id}/explain/lime")
def explain_customer_lime(request: Request, customer_id: int) -> dict:
    """LIME local explanation for one customer, next to the SHAP top factors for comparison."""
    st = request.app.state.s
    with Session(st.engine) as sess:
        c = sess.get(Customer, customer_id)
    if c is None:
        raise HTTPException(404, f"customer {customer_id} not found")
    x = np.array([c.features[f] for f in FEATURE_COLUMNS], dtype=float)
    shap_top = explain_row(st.churn_model, x, st.explainer)["top_factors"][:5]
    out = lime_row(st.churn_model, st.lime, x)
    names = [f["feature"] for f in shap_top]
    out["agree_with_shap_top5"] = [n for n in names if any(n in f["rule"] for f in out["lime_factors"][:5])]
    return {"customer_id": customer_id, "shap_top5": names, **out}


@router.get("/customers/{customer_id}/recommendations")
def recommendations(request: Request, customer_id: int, n: int = Query(5, ge=1, le=20)) -> dict:
    """Top-n products to recommend (item-item co-purchase model fitted on training customers)."""
    st = request.app.state.s
    if not st.recommender:
        raise HTTPException(503, "recommender artifact not loaded")
    bought = st.recommender["customer_products"].get(customer_id)
    if bought is None:
        raise HTTPException(404, f"customer {customer_id} not found")
    from src.recommend.recommender import recommend_for

    items = [
        {
            "stock_code": code,
            "description": st.recommender["description"].get(code, ""),
            "score": round(score, 4),
        }
        for code, score in recommend_for(st.recommender, bought, n)
    ]
    return {"customer_id": customer_id, "n_products_bought": len(bought), "recommendations": items}


@router.get("/segments")
def segments(request: Request) -> list[dict]:
    """Segment profiles: average RFM, predicted_* model outputs and observed_* retrospective outcomes."""
    import pandas as pd

    from src.utils.config import rpath

    prof = pd.read_csv(rpath(request.app.state.s.cfg, "reports") / "segment_profiles.csv")
    # The CSV's churn and future_spend are what actually happened after the cutoff (retrospective evaluation labels).
    prof = prof.rename(columns={"churn": "observed_churn_rate", "future_spend": "observed_future_spend"})
    with Session(request.app.state.s.engine) as sess:
        rows = sess.execute(
            select(Segment.segment_id, Prediction.churn_probability, Prediction.clv_predicted).join(
                Prediction, Prediction.customer_id == Segment.customer_id
            )
        ).all()
    pred = pd.DataFrame(rows, columns=["segment_id", "predicted_churn_probability", "predicted_clv_90d"])
    pred = pred.groupby("segment_id").mean().round(3).reset_index()
    return prof.merge(pred, on="segment_id", how="left").to_dict(orient="records")
