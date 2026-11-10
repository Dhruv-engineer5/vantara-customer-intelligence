"""Shared application state: DB engine, trained artifacts and helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

from src.db.session import make_engine, seed_if_empty
from src.explainability.explain import make_explainer, make_lime
from src.features.build import FEATURE_COLUMNS
from src.models.ae_serve import NumpyAutoEncoder, recon_error
from src.data.reference import load_reference
from src.utils.config import database_url, load_config, rpath
from src.utils.logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class State:
    """Loaded once at startup."""

    cfg: dict
    engine: Engine
    churn_model: object
    clv_model: object
    explainer: object
    metadata: dict
    medians: pd.Series
    clv_cap: float
    ae: object = None
    ae_meta: dict = field(default_factory=dict)
    seg: dict = field(default_factory=dict)
    lime: object = None
    recommender: dict = field(default_factory=dict)
    ref: dict = field(default_factory=dict)


def load_state(engine: Engine | None = None) -> State:
    """Load config, models and (optionally) connect + seed the database."""
    log.info("startup: load_state begin")
    cfg = load_config()
    art = Path(rpath(cfg, "artifacts"))
    if engine is None:
        engine = make_engine(database_url(cfg))
    log.info("startup: engine ready")
    seed_if_empty(engine, cfg)
    log.info("startup: seed complete, loading artifacts")
    churn = joblib.load(art / "churn_model.joblib")
    meta = json.loads((art / "metadata.json").read_text())
    table = pd.read_parquet(rpath(cfg, "features"))
    ae_meta = joblib.load(art / "autoencoder_meta.joblib")
    ae = NumpyAutoEncoder(art / "autoencoder_weights.json")
    log.info("startup: artifacts loaded")
    return State(
        cfg=cfg,
        engine=engine,
        churn_model=churn,
        clv_model=joblib.load(art / "clv_model.joblib"),
        explainer=make_explainer(churn),
        metadata=meta,
        medians=table.loc[table["data_split"] == "train", FEATURE_COLUMNS].median(),
        clv_cap=float(np.log1p(table.loc[table["data_split"] == "train", "future_spend"].max())),
        ae=ae,
        ae_meta=ae_meta,
        seg=joblib.load(art / "segmentation.joblib"),
        recommender=joblib.load(art / "recommender.joblib") if (art / "recommender.joblib").exists() else {},
        lime=make_lime(table.loc[table["data_split"] == "train", FEATURE_COLUMNS].to_numpy(dtype=float), cfg["seed"]),
        ref=load_reference(Path(rpath(cfg, "reference_stats"))),
    )


def vector(state: State, feats: dict[str, float]) -> np.ndarray:
    """Order a feature dict into the model's column order, filling gaps with training medians."""
    row = state.medians.copy()
    for k, v in feats.items():
        if k not in row.index:
            raise ValueError(f"unknown feature: {k}")
        row[k] = v
    return row.to_numpy(dtype=float)


def score_matrix(state: State, X: np.ndarray) -> pd.DataFrame:
    """Churn probability, CLV, anomaly flag for rows of feature vectors."""
    proba = state.churn_model.predict_proba(X)[:, 1]
    clv = np.expm1(np.clip(state.clv_model.predict(X), 0, state.clv_cap))
    idx = [FEATURE_COLUMNS.index(c) for c in state.ae_meta["cols"]]
    err = recon_error(state.ae, state.ae_meta["scaler"].transform(np.log1p(np.abs(X[:, idx]))))
    return pd.DataFrame(
        {"churn_probability": proba, "clv_predicted_90d": clv, "is_anomaly": err > state.ae_meta["threshold"]}
    )


_RISK = load_config()["risk"]


def risk_level(p: float) -> str:
    """Bucket a probability into a business label."""
    r = _RISK
    return "high" if p >= r["high_churn"] else "moderate" if p >= r["moderate_churn"] else "low"


def segment_for(state: State, X: np.ndarray) -> list[str]:
    """Assign K-Means segment names to feature rows."""
    from src.segmentation.cluster import SEG_COLS

    cols = [FEATURE_COLUMNS.index(c) for c in SEG_COLS]
    frame = pd.DataFrame(X[:, cols], columns=SEG_COLS)
    for c in ("frequency", "monetary_total", "avg_basket_size", "recency_days"):
        frame[c] = np.log1p(frame[c].clip(lower=0))
    lab = state.seg["kmeans"].predict(state.seg["scaler"].transform(frame.to_numpy()))
    return [state.seg["names"][int(i)] for i in lab]
