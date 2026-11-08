"""Single and batch scoring endpoints."""

from __future__ import annotations

from typing import Any

import io

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.deps import risk_level, score_matrix, segment_for, vector
from api.schemas.models import CustomerFeatures, PredictionOut
from src.db.models import Customer
from src.explainability.explain import explain_row
from src.features.build import FEATURE_COLUMNS
from src.utils.logging_setup import get_logger

router = APIRouter(tags=["predict"])
log = get_logger(__name__)


@router.post("/predict", response_model=PredictionOut)
def predict(request: Request, rec: CustomerFeatures) -> PredictionOut:
    """Score one customer record (unspecified optional features fall back to training medians)."""
    st = request.app.state.s
    feats = {k: v for k, v in rec.model_dump(exclude={"extra_features"}).items() if v is not None}
    feats.update(rec.extra_features)
    try:
        x = vector(st, feats)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    sc = score_matrix(st, x.reshape(1, -1)).iloc[0]
    ex = explain_row(st.churn_model, x, st.explainer)
    return PredictionOut(
        churn_probability=float(sc.churn_probability),
        churn_risk_level=risk_level(float(sc.churn_probability)),
        purchase_probability_90d=1.0 - float(sc.churn_probability),
        clv_predicted_90d=float(sc.clv_predicted_90d),
        is_anomaly=bool(sc.is_anomaly),
        segment_name=segment_for(st, x.reshape(1, -1))[0],
        explanation=ex["explanation"],
    )


def _batch_frame(st: Any, df: pd.DataFrame) -> pd.DataFrame:
    """Score a CSV: rows with customer_id use stored features; else rows must contain model feature columns."""
    if "customer_id" in df.columns and not set(["recency_days", "frequency", "monetary_total"]) <= set(df.columns):
        ids = [int(i) for i in df["customer_id"]]
        with Session(st.engine) as sess:
            found = {
                c.customer_id: c for c in sess.scalars(select(Customer).where(Customer.customer_id.in_(ids))).all()
            }
        missing = [i for i in ids if i not in found]
        if missing:
            raise HTTPException(422, f"unknown customer_id(s): {missing[:10]}")
        X = np.array([[found[i].features[f] for f in FEATURE_COLUMNS] for i in ids], dtype=float)
        out_ids = ids
    else:
        need = {"recency_days", "frequency", "monetary_total"}
        if not need <= set(df.columns):
            raise HTTPException(422, f"CSV needs either a customer_id column or at least {sorted(need)}")
        unknown = [c for c in df.columns if c not in FEATURE_COLUMNS and c != "customer_id"]
        if unknown:
            raise HTTPException(422, f"unknown columns: {unknown}")
        rows = []
        for _, r in df.iterrows():
            rows.append(vector(st, {k: float(v) for k, v in r.items() if k in FEATURE_COLUMNS and pd.notna(v)}))
        X = np.array(rows)
        out_ids = df["customer_id"].astype(int).tolist() if "customer_id" in df.columns else list(range(1, len(df) + 1))
    sc = score_matrix(st, X)
    sc.insert(0, "customer_id", out_ids)
    sc["churn_risk_level"] = sc["churn_probability"].map(risk_level)
    sc["segment_name"] = segment_for(st, X)
    return sc.round({"churn_probability": 4, "clv_predicted_90d": 2})


def _pdf(df: pd.DataFrame) -> bytes:
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Vantara batch scoring report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(
        0,
        6,
        f"Customers scored: {len(df)}   High risk: {(df.churn_risk_level == 'high').sum()}   "
        f"Mean churn probability: {df.churn_probability.mean():.1%}",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.ln(2)
    cols = ["customer_id", "churn_probability", "churn_risk_level", "clv_predicted_90d", "segment_name"]
    widths = [28, 34, 32, 40, 56]
    pdf.set_font("Helvetica", "B", 8)
    for c, w in zip(cols, widths):
        pdf.cell(w, 6, c, border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 8)
    for _, r in df.head(200).iterrows():
        for c, w in zip(cols, widths):
            v = r[c]
            if c == "churn_probability":
                cell = f"{float(v):.4f}"
            elif c == "clv_predicted_90d":
                cell = f"{float(v):,.2f}"
            else:
                cell = str(v)[:30]
            pdf.cell(w, 5, cell, border=1)
        pdf.ln()
    if len(df) > 200:
        pdf.cell(0, 6, f"(first 200 of {len(df)} rows shown; use CSV for all)")
    return bytes(pdf.output())


@router.post("/predict/batch", response_model=None)
async def predict_batch(
    request: Request, file: UploadFile = File(...), format: str = Query("json", pattern="^(json|csv|pdf)$")
) -> Response | list[dict]:
    """Batch scoring from an uploaded CSV. format=json|csv|pdf."""
    st = request.app.state.s
    raw = await file.read()
    if len(raw) > 5_000_000:
        raise HTTPException(413, "CSV larger than 5 MB")
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(422, f"could not read CSV: {exc}") from exc
    if df.empty:
        raise HTTPException(422, "CSV has no rows")
    out = _batch_frame(st, df)
    log.info("batch scored rows=%d", len(out))
    if format == "csv":
        return Response(
            out.to_csv(index=False),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=scores.csv"},
        )
    if format == "pdf":
        return Response(
            _pdf(out), media_type="application/pdf", headers={"Content-Disposition": "attachment; filename=scores.pdf"}
        )
    return out.to_dict(orient="records")
