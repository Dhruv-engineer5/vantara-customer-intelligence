"""Private workspace endpoints: upload, list and delete a signed-in user's datasets."""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from api.auth import require_user
from src.db.models import (
    UserCustomer,
    UserDataset,
    UserPrediction,
    UserSegment,
    UserTransaction,
)
from src.userdata import (
    GLOBAL_MAX_STORED_ROWS,
    MAX_DATASETS_PER_USER,
    MAX_FILE_BYTES,
    UploadError,
    process_upload,
)
from src.utils.logging_setup import get_logger

router = APIRouter(tags=["datasets"])
log = get_logger(__name__)


def _summary(d: UserDataset) -> dict:
    return {
        "dataset_id": d.dataset_id,
        "name": d.name,
        "status": d.status,
        "error": d.error,
        "n_rows": d.n_rows,
        "n_customers": d.n_customers,
        "first_date": d.first_date.date().isoformat() if d.first_date else None,
        "last_date": d.last_date.date().isoformat() if d.last_date else None,
        "created_at": d.created_at.isoformat(),
    }


def _owned(request: Request, dataset_id: str) -> str:
    """The dataset id, proven to belong to the signed-in user (404 otherwise)."""
    user = require_user(request)
    with Session(request.app.state.s.engine) as sess:
        owner = sess.scalar(select(UserDataset.user_id).where(UserDataset.dataset_id == dataset_id))
    if owner is None or owner != user:
        raise HTTPException(404, "dataset not found")
    return dataset_id


@router.get("/datasets")
def list_datasets(request: Request) -> list[dict]:
    """The signed-in user's datasets (newest first)."""
    user = require_user(request)
    with Session(request.app.state.s.engine) as sess:
        rows = sess.scalars(
            select(UserDataset).where(UserDataset.user_id == user).order_by(UserDataset.created_at.desc())
        ).all()
    return [_summary(d) for d in rows]


@router.post("/datasets", status_code=201)
async def create_dataset(request: Request, file: UploadFile = File(...), name: str = Form("My dataset")) -> dict:
    """Upload a transaction CSV (Online Retail II layout), clean it, score it and store the workspace."""
    st = request.app.state.s
    user = require_user(request)
    with Session(st.engine) as sess:
        n_existing = sess.scalar(select(func.count()).select_from(UserDataset).where(UserDataset.user_id == user))
        total_rows = sess.scalar(select(func.coalesce(func.sum(UserDataset.n_rows), 0))) or 0
    if (n_existing or 0) >= MAX_DATASETS_PER_USER:
        raise HTTPException(
            409, f"you already have {MAX_DATASETS_PER_USER} datasets; delete one before uploading another"
        )
    if total_rows >= GLOBAL_MAX_STORED_ROWS:
        raise HTTPException(503, "the demo is at its storage limit right now; please try again later")
    raw = await file.read(MAX_FILE_BYTES + 1)  # bounded read: never buffers more than the cap
    if len(raw) > MAX_FILE_BYTES:
        raise HTTPException(413, f"CSV larger than {MAX_FILE_BYTES // 1_000_000} MB")
    name = (name or "My dataset").strip()[:120] or "My dataset"
    ds_id = str(uuid.uuid4())
    with Session(st.engine) as sess:
        sess.add(
            UserDataset(
                dataset_id=ds_id, user_id=user, name=name, status="processing", created_at=datetime.utcnow()
            )
        )
        sess.commit()
    try:
        out = process_upload(st, raw, st.cfg)
    except UploadError as exc:
        with Session(st.engine) as sess:
            sess.query(UserDataset).filter(UserDataset.dataset_id == ds_id).update(
                {"status": "failed", "error": str(exc)[:500]}
            )
            sess.commit()
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("upload processing failed")
        with Session(st.engine) as sess:
            sess.query(UserDataset).filter(UserDataset.dataset_id == ds_id).update(
                {"status": "failed", "error": "processing failed on our side; try a smaller or cleaner CSV"[:500]}
            )
            sess.commit()
        raise HTTPException(500, "processing failed on our side; try a smaller or cleaner CSV") from exc
    with Session(st.engine) as sess:
        sess.bulk_insert_mappings(
            UserCustomer, [{**c, "dataset_id": ds_id} for c in out["customers"]]
        )
        sess.bulk_insert_mappings(
            UserPrediction, [{**p, "dataset_id": ds_id} for p in out["predictions"]]
        )
        sess.bulk_insert_mappings(UserSegment, [{**s, "dataset_id": ds_id} for s in out["segments"]])
        sess.bulk_insert_mappings(
            UserTransaction, [{**t, "dataset_id": ds_id} for t in out["transactions"]]
        )
        sess.query(UserDataset).filter(UserDataset.dataset_id == ds_id).update(
            {
                "status": "ready",
                "n_rows": out["n_rows"],
                "n_customers": out["n_customers"],
                "first_date": out["first_date"],
                "last_date": out["last_date"],
                "recommender": out["recommender"],
            }
        )
        sess.commit()
        ds = sess.get(UserDataset, ds_id)
        return {"dataset": _summary(ds), "report": out["report"]}


@router.delete("/datasets/{dataset_id}", status_code=204)
def delete_dataset(request: Request, dataset_id: str) -> None:
    """Delete one of the signed-in user's datasets and all its rows."""
    st = request.app.state.s
    ds_id = _owned(request, dataset_id)
    with Session(st.engine) as sess:
        for model in (UserCustomer, UserPrediction, UserSegment, UserTransaction):
            sess.execute(delete(model).where(model.dataset_id == ds_id))
        sess.execute(delete(UserDataset).where(UserDataset.dataset_id == ds_id))
        sess.commit()
    log.info("dataset deleted id=%s", ds_id)
