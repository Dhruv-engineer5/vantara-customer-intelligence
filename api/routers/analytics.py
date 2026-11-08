"""Health, metadata and revenue trend endpoints."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Request
from sqlalchemy import text

from api.schemas.models import Health
from src.db.session import artifact_hash, current_score_run

router = APIRouter(tags=["system"])


@router.get("/health", response_model=Health)
def health(request: Request) -> Health:
    """Liveness plus a database round trip."""
    st = request.app.state.s
    try:
        with st.engine.connect() as c:
            c.execute(text("SELECT 1"))
        db = "ok"
    except Exception:  # noqa: BLE001
        db = "down"
    return Health(status="ok" if db == "ok" else "degraded", database=db, model_loaded=st.churn_model is not None)


def _recommender_eval(st: Any) -> dict:
    """Offline evaluation of the product recommender, if the pipeline produced it."""
    import json

    from src.utils.config import rpath

    p = rpath(st.cfg, "artifacts") / "recommender_eval.json"
    return json.loads(p.read_text()) if p.exists() else {}


@router.get("/model/metadata")
def model_metadata(request: Request) -> dict:
    """Training metadata, held-out metrics and which brief targets were met."""
    st = request.app.state.s
    m = st.metadata
    run = current_score_run(st.engine)
    file_hash = artifact_hash(st.cfg) if hasattr(st, "cfg") else None
    return {
        "score_run": {
            "db_artifact_hash": run.artifact_hash if run else None,
            "files_artifact_hash": file_hash,
            "in_sync": bool(run and file_hash and run.artifact_hash == file_hash),
            "loaded_at": run.loaded_at.isoformat() if run else None,
        },
        "recommender": _recommender_eval(st),
        **{
            k: m[k]
            for k in (
                "production_churn_model",
                "clv_model",
                "n_customers",
                "split_sizes",
                "churn_rate",
                "cutoff_date",
                "horizon_days",
                "churn_results",
                "clv_results",
                "targets",
                "targets_met",
                "best_model_by_val_auc_overall",
                "anomaly",
            )
        },
    }


@router.get("/revenue/trend")
def revenue_trend(request: Request, forecast_months: int = 3) -> dict:
    """Monthly net revenue from invoices plus a simple linear-trend forecast overlay."""
    with request.app.state.s.engine.connect() as c:
        df = pd.read_sql(text("SELECT invoice_date, amount FROM transactions"), c)
    df["month"] = pd.to_datetime(df["invoice_date"]).dt.to_period("M").dt.to_timestamp()
    m = df.groupby("month")["amount"].sum().sort_index()
    m = m.iloc[:-1] if len(m) > 1 else m  # last month is partial in the source data
    y = m.tail(12).to_numpy()
    x = np.arange(len(y))
    slope, icpt = np.polyfit(x, y, 1)
    fut = [float(max(slope * (len(y) + i) + icpt, 0)) for i in range(forecast_months)]
    fut_idx = pd.date_range(m.index[-1] + pd.offsets.MonthBegin(1), periods=forecast_months, freq="MS")
    return {
        "history": [{"month": d.strftime("%Y-%m"), "revenue": round(float(v), 2)} for d, v in m.items()],
        "forecast": [{"month": d.strftime("%Y-%m"), "revenue": round(v, 2)} for d, v in zip(fut_idx, fut)],
        "method": "ordinary least squares line on the last 12 complete months (overlay only, not a validated forecast)",
    }
