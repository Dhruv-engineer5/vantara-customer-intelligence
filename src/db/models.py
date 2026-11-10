"""SQLAlchemy schema: customers, transactions (invoice level), predictions, segments."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base."""


class Customer(Base):
    """One row per modelled customer with the full feature vector stored as JSON."""

    __tablename__ = "customers"
    customer_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    country: Mapped[str] = mapped_column(String(64))
    recency_days: Mapped[float] = mapped_column(Float)
    frequency: Mapped[float] = mapped_column(Float)
    monetary_total: Mapped[float] = mapped_column(Float)
    clv_historical: Mapped[float] = mapped_column(Float)
    engagement_score: Mapped[float] = mapped_column(Float)
    features: Mapped[dict] = mapped_column(JSON)


class Transaction(Base):
    """One row per invoice (line items are aggregated; raw lines stay in data/interim)."""

    __tablename__ = "transactions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    invoice: Mapped[str] = mapped_column(String(16), index=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.customer_id"), index=True)
    invoice_date: Mapped[datetime] = mapped_column(DateTime, index=True)
    amount: Mapped[float] = mapped_column(Float)
    items: Mapped[float] = mapped_column(Float)
    is_return: Mapped[bool] = mapped_column(Boolean)


class Prediction(Base):
    """Scored predictions so the dashboard never recomputes on page load."""

    __tablename__ = "predictions"
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.customer_id"), primary_key=True)
    churn_probability: Mapped[float] = mapped_column(Float, index=True)
    clv_predicted: Mapped[float] = mapped_column(Float)
    anomaly_score: Mapped[float] = mapped_column(Float)
    is_anomaly: Mapped[bool] = mapped_column(Boolean)
    data_split: Mapped[str] = mapped_column(String(8))
    model_name: Mapped[str] = mapped_column(String(32))
    scored_at: Mapped[datetime] = mapped_column(DateTime)


class Segment(Base):
    """K-Means segment assignment per customer."""

    __tablename__ = "segments"
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.customer_id"), primary_key=True)
    segment_id: Mapped[int] = mapped_column(Integer, index=True)
    segment_name: Mapped[str] = mapped_column(String(48))


class ScoreRun(Base):
    """One row per load of model scores into the database; artifact_hash detects stale scores."""

    __tablename__ = "score_runs"

    run_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    artifact_hash: Mapped[str] = mapped_column(String(64), index=True)
    model_name: Mapped[str] = mapped_column(String(32))
    n_customers: Mapped[int] = mapped_column(Integer)
    loaded_at: Mapped[datetime] = mapped_column(DateTime)


class PredictionHistory(Base):
    """Archived prediction rows from earlier score runs. Never deleted by a refresh."""

    __tablename__ = "prediction_history"

    customer_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    churn_probability: Mapped[float] = mapped_column(Float)
    clv_predicted: Mapped[float] = mapped_column(Float)
    anomaly_score: Mapped[float] = mapped_column(Float)
    is_anomaly: Mapped[bool] = mapped_column(Boolean)
    data_split: Mapped[str] = mapped_column(String(8))
    model_name: Mapped[str] = mapped_column(String(32))
    scored_at: Mapped[datetime] = mapped_column(DateTime)


class UserDataset(Base):
    """One uploaded workspace owned by a signed-in user (Supabase Auth user UUID)."""

    __tablename__ = "user_datasets"
    dataset_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    name: Mapped[str] = mapped_column(String(120))
    status: Mapped[str] = mapped_column(String(16))  # processing | ready | failed
    error: Mapped[str | None] = mapped_column(String(500), nullable=True)
    n_rows: Mapped[int] = mapped_column(Integer, default=0)
    n_customers: Mapped[int] = mapped_column(Integer, default=0)
    first_date: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_date: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    recommender: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class UserCustomer(Base):
    """One row per modelled customer in a user's uploaded dataset."""

    __tablename__ = "user_customers"
    dataset_id: Mapped[str] = mapped_column(ForeignKey("user_datasets.dataset_id"), primary_key=True)
    customer_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    country: Mapped[str] = mapped_column(String(64))
    recency_days: Mapped[float] = mapped_column(Float)
    frequency: Mapped[float] = mapped_column(Float)
    monetary_total: Mapped[float] = mapped_column(Float)
    clv_historical: Mapped[float] = mapped_column(Float)
    engagement_score: Mapped[float] = mapped_column(Float)
    features: Mapped[dict] = mapped_column(JSON)


class UserPrediction(Base):
    """Stored model scores for one customer in a user's dataset."""

    __tablename__ = "user_predictions"
    dataset_id: Mapped[str] = mapped_column(ForeignKey("user_datasets.dataset_id"), primary_key=True)
    customer_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    churn_probability: Mapped[float] = mapped_column(Float, index=True)
    clv_predicted: Mapped[float] = mapped_column(Float)
    is_anomaly: Mapped[bool] = mapped_column(Boolean)
    model_name: Mapped[str] = mapped_column(String(32))
    scored_at: Mapped[datetime] = mapped_column(DateTime)


class UserSegment(Base):
    """K-Means segment assignment for one customer in a user's dataset."""

    __tablename__ = "user_segments"
    dataset_id: Mapped[str] = mapped_column(ForeignKey("user_datasets.dataset_id"), primary_key=True)
    customer_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    segment_id: Mapped[int] = mapped_column(Integer)
    segment_name: Mapped[str] = mapped_column(String(48))


class UserTransaction(Base):
    """Invoice-level rows of a user's uploaded dataset (powers the revenue trend view)."""

    __tablename__ = "user_transactions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("user_datasets.dataset_id"), index=True)
    invoice: Mapped[str] = mapped_column(String(32))
    customer_id: Mapped[str] = mapped_column(String(64), index=True)
    invoice_date: Mapped[datetime] = mapped_column(DateTime)
    amount: Mapped[float] = mapped_column(Float)
    items: Mapped[float] = mapped_column(Float)
    is_return: Mapped[bool] = mapped_column(Boolean)
