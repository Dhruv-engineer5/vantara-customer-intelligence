"""Dataset resolution and tenant-scoped query helpers.

The public sample workspace lives in the original tables (customers/predictions/segments/
transactions) and stays readable without sign-in. User uploads live in the user_* tables
keyed by dataset_id; every query through here first proves the dataset belongs to the
signed-in user (a missing or foreign dataset is a 404, so ids cannot be probed).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.auth import require_user
from src.db.models import UserCustomer, UserDataset, UserPrediction, UserSegment

SAMPLE = "sample"


@dataclass
class Ctx:
    """Which workspace a request is scoped to."""

    sample: bool
    dataset_id: str | None = None


def resolve(request: Request, dataset: str | None) -> Ctx:
    """Resolve the ?dataset= query parameter into an authorized workspace context."""
    if not dataset or dataset == SAMPLE:
        return Ctx(sample=True)
    user = require_user(request)
    with Session(request.app.state.s.engine) as sess:
        row = sess.execute(
            select(UserDataset.dataset_id, UserDataset.user_id, UserDataset.status, UserDataset.error).where(
                UserDataset.dataset_id == dataset
            )
        ).first()
    if row is None or row.user_id != user:
        raise HTTPException(404, "dataset not found")
    if row.status != "ready":
        detail = "dataset is still processing" if row.status == "processing" else f"dataset failed: {row.error}"
        raise HTTPException(409, detail)
    return Ctx(sample=False, dataset_id=row.dataset_id)


def user_base(ctx: Ctx) -> Any:
    """The user-table equivalent of customers.join(predictions).join(segments)."""
    return (
        select(UserCustomer, UserPrediction, UserSegment)
        .join(
            UserPrediction,
            (UserPrediction.dataset_id == UserCustomer.dataset_id)
            & (UserPrediction.customer_id == UserCustomer.customer_id),
        )
        .join(
            UserSegment,
            (UserSegment.dataset_id == UserCustomer.dataset_id)
            & (UserSegment.customer_id == UserCustomer.customer_id),
        )
        .where(UserCustomer.dataset_id == ctx.dataset_id)
    )


def user_row(c: UserCustomer, p: UserPrediction, s: UserSegment) -> dict:
    """Same response shape as the sample workspace's customer row."""
    from api.deps import risk_level

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
        "data_split": "user",
    }
