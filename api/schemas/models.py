"""Pydantic request/response models with clear validation errors."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CustomerFeatures(BaseModel):
    """Customer record for ad-hoc scoring. Unspecified optional fields use training-set medians."""

    model_config = ConfigDict(extra="forbid")
    recency_days: float = Field(ge=0, le=5000, description="Days since last purchase")
    frequency: float = Field(ge=1, le=5000, description="Number of distinct orders")
    monetary_total: float = Field(ge=0, description="Total spend before the cutoff")
    avg_basket_size: float | None = Field(default=None, ge=0)
    return_rate: float | None = Field(default=None, ge=0, le=1)
    orders_last_90d: float | None = Field(default=None, ge=0)
    spend_last_90d: float | None = Field(default=None, ge=0)
    tenure_days: float | None = Field(default=None, ge=0)
    gap_mean_days: float | None = Field(default=None, ge=0)
    extra_features: dict[str, float] = Field(default_factory=dict, description="Any other model feature by name")


class PredictionOut(BaseModel):
    """Scored output for one customer."""

    customer_id: int | None = None
    churn_probability: float
    churn_risk_level: str
    purchase_probability_90d: float
    clv_predicted_90d: float
    is_anomaly: bool
    segment_name: str | None = None
    explanation: str | None = None


class Health(BaseModel):
    """Health-check payload."""

    status: str
    database: str
    model_loaded: bool
