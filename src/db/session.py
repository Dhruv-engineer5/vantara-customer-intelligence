"""Engine/session helpers and first-start seeding from the processed data files."""

from __future__ import annotations

from typing import Any

import concurrent.futures  # noqa: F401  (kept for future async probes)
import hashlib
import json
import re
import threading
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, func, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from src.db.models import Base, Customer, Prediction, PredictionHistory, ScoreRun, Segment, Transaction
from src.features.build import FEATURE_COLUMNS
from src.utils.config import load_config, rpath
from src.utils.logging_setup import get_logger

log = get_logger(__name__)


class _ProbeTimeout(Exception):
    """A single connection probe exceeded its hard wall-clock bound."""


def _scrub(exc: BaseException) -> str:
    """First line of the error with host/user/IP details redacted (no DSN parts in logs)."""
    msg = (str(exc).splitlines() or [""])[0]
    msg = re.sub(r'"[^"]*"', '"..."', msg)
    msg = re.sub(r"user '\S+'", "user '...'", msg)
    return msg[:160]


def _probe(engine: Engine, hard_timeout: float) -> None:
    """One bounded connect+SELECT 1. Runs in a daemon thread so a wedged libpq call
    (e.g. a pooler holding the session while a compute wakes, which libpq's
    connect_timeout does not reliably bound) cannot hang startup forever."""
    err: dict[str, BaseException] = {}

    def _do() -> None:
        try:
            with engine.connect() as c:
                c.execute(text("SELECT 1"))
        except BaseException as exc:  # noqa: BLE001
            err["e"] = exc

    t = threading.Thread(target=_do, daemon=True)
    t.start()
    t.join(hard_timeout)
    if t.is_alive():
        raise _ProbeTimeout(f"probe exceeded {hard_timeout:.0f}s")
    if "e" in err:
        raise err["e"]


def make_engine(url: str, retries: int = 12, wait: float = 2.5, connect_timeout: float = 6.0,
                hard_timeout: float = 15.0) -> Engine:
    """Create an engine and wait until the database accepts connections.

    connect_timeout bounds the libpq socket phase of each attempt; hard_timeout
    bounds the whole probe (connect + SELECT 1) wall-clock, because a wedged
    connection (e.g. a pooler holding the session while a compute wakes) can
    otherwise hang the process forever so the service never binds its port.
    """
    kwargs: dict[str, Any] = {"pool_pre_ping": True, "future": True}
    if url.startswith("postgresql"):
        kwargs["connect_args"] = {"connect_timeout": int(connect_timeout)}
    engine = create_engine(url, **kwargs)
    t0 = time.monotonic()
    for attempt in range(retries):
        try:
            _probe(engine, hard_timeout)
            log.info("database ready attempt=%d after=%.1fs", attempt + 1, time.monotonic() - t0)
            return engine
        except Exception as exc:  # noqa: BLE001
            pgcode = getattr(getattr(exc, "orig", None), "pgcode", None) or getattr(exc, "pgcode", None)
            log.warning(
                "database not ready attempt=%d err=%s pgcode=%s detail=%s",
                attempt + 1, type(exc).__name__, pgcode, _scrub(exc),
            )
            time.sleep(wait)
    raise RuntimeError("database unreachable")


def build_invoices(cfg: dict) -> pd.DataFrame:
    """Invoice-level table from the interim transactions (cached as data/processed/invoices.parquet)."""
    out = rpath(cfg, "invoices")
    if out.exists():
        return pd.read_parquet(out)
    tx = pd.read_parquet(rpath(cfg, "interim_transactions"))
    inv = (
        tx.groupby(["invoice", "customer_id"])
        .agg(
            invoice_date=("invoice_date", "max"),
            amount=("line_total", "sum"),
            items=("quantity", "sum"),
            is_return=("is_return", "max"),
        )
        .reset_index()
    )
    inv.to_parquet(out, index=False)
    return inv


def artifact_hash(cfg: dict) -> str:
    """SHA-256 over the files that define customer scores (metadata, scores, segments, features)."""
    h = hashlib.sha256()
    for path in (
        Path(rpath(cfg, "artifacts")) / "metadata.json",
        rpath(cfg, "scores"),
        rpath(cfg, "segments"),
        rpath(cfg, "features"),
    ):
        h.update(Path(path).read_bytes())
    return h.hexdigest()


def current_score_run(engine: Engine) -> ScoreRun | None:
    """Latest score run recorded in the database (None if none)."""
    with Session(engine) as s:
        r = s.scalars(select(ScoreRun).order_by(ScoreRun.run_id.desc()).limit(1)).first()
        if r is not None:
            s.expunge(r)
        return r


def _score_rows(cfg: dict, meta: dict, now: datetime) -> Any:
    table = pd.read_parquet(rpath(cfg, "features"))
    scores = pd.read_parquet(rpath(cfg, "scores"))
    seg = pd.read_parquet(rpath(cfg, "segments"))
    cust = [
        {
            "customer_id": int(cid),
            "country": str(r["country"]),
            "recency_days": float(r["recency_days"]),
            "frequency": float(r["frequency"]),
            "monetary_total": float(r["monetary_total"]),
            "clv_historical": float(r["clv_historical"]),
            "engagement_score": float(r["engagement_score"]),
            "features": {c: float(r[c]) for c in FEATURE_COLUMNS},
        }
        for cid, r in table.iterrows()
    ]
    pred = [
        {
            "customer_id": int(cid),
            "churn_probability": float(r["churn_probability"]),
            "clv_predicted": float(r["clv_predicted"]),
            "anomaly_score": float(r["anomaly_score"]),
            "is_anomaly": bool(r["is_anomaly"]),
            "data_split": str(r["data_split"]),
            "model_name": meta["production_churn_model"],
            "scored_at": now,
        }
        for cid, r in scores.iterrows()
    ]
    segs = [
        {"customer_id": int(cid), "segment_id": int(r["segment_id"]), "segment_name": str(r["segment_name"])}
        for cid, r in seg.iterrows()
    ]
    return table, cust, pred, segs


def seed_if_empty(engine: Engine, cfg: dict | None = None) -> bool:
    """First start: create tables and load everything. Later starts: refresh scores if the artifacts changed.

    Returns True if anything was loaded. A refresh never deletes data: the previous prediction rows are copied to
    prediction_history under their score run, then customer features, predictions and segments are upserted.
    """
    cfg = cfg or load_config()
    log.info("schema init start")
    Base.metadata.create_all(engine)
    log.info("schema init done")
    with Session(engine) as s:
        empty = not s.scalar(select(func.count()).select_from(Customer))
    if empty:
        log.info("seed start")
        _load_all(engine, cfg)
        return True
    log.info("seed refresh start")
    return refresh_scores(engine, cfg)


def _load_all(engine: Engine, cfg: dict) -> None:
    meta = json.loads((Path(rpath(cfg, "artifacts")) / "metadata.json").read_text())
    now = datetime.utcnow()
    table, cust, pred, segs = _score_rows(cfg, meta, now)
    inv = build_invoices(cfg)
    inv = inv[inv["customer_id"].isin(table.index)]
    with Session(engine) as s:
        s.bulk_insert_mappings(Customer, cust)
        s.flush()
        s.bulk_insert_mappings(Transaction, _tx_rows(inv))
        s.bulk_insert_mappings(Prediction, pred)
        s.bulk_insert_mappings(Segment, segs)
        s.add(
            ScoreRun(
                artifact_hash=artifact_hash(cfg),
                model_name=meta["production_churn_model"],
                n_customers=len(table),
                loaded_at=now,
            )
        )
        s.commit()
    log.info("seeded database customers=%d invoices=%d", len(table), len(inv))


def _tx_rows(inv: pd.DataFrame) -> list[dict]:
    return [
        {
            "invoice": str(r.invoice),
            "customer_id": int(r.customer_id),
            "invoice_date": r.invoice_date.to_pydatetime(),
            "amount": float(r.amount),
            "items": float(r.items),
            "is_return": bool(r.is_return),
        }
        for r in inv.itertuples()
    ]


def refresh_scores(engine: Engine, cfg: dict | None = None, force: bool = False) -> bool:
    """Load new scores when the artifact hash differs from the latest score run. Non-destructive."""
    cfg = cfg or load_config()
    Base.metadata.create_all(engine)
    new_hash = artifact_hash(cfg)
    last = current_score_run(engine)
    if last is not None and last.artifact_hash == new_hash and not force:
        return False
    meta = json.loads((Path(rpath(cfg, "artifacts")) / "metadata.json").read_text())
    now = datetime.utcnow()
    table, cust, pred, segs = _score_rows(cfg, meta, now)
    with Session(engine) as s:
        old_run = last.run_id if last is not None else 0  # 0 = rows that pre-date score-run tracking
        old = s.scalars(select(Prediction)).all()
        s.bulk_insert_mappings(
            PredictionHistory,
            [
                {
                    "customer_id": p.customer_id,
                    "run_id": old_run,
                    "churn_probability": p.churn_probability,
                    "clv_predicted": p.clv_predicted,
                    "anomaly_score": p.anomaly_score,
                    "is_anomaly": p.is_anomaly,
                    "data_split": p.data_split,
                    "model_name": p.model_name,
                    "scored_at": p.scored_at,
                }
                for p in old
                if not s.get(PredictionHistory, (p.customer_id, old_run))
            ],
        )
        existing = set(s.scalars(select(Customer.customer_id)).all())
        for row in cust:
            s.merge(Customer(**row))
        s.flush()
        new_ids = [c["customer_id"] for c in cust if c["customer_id"] not in existing]
        if new_ids:
            inv = build_invoices(cfg)
            inv = inv[inv["customer_id"].isin(new_ids)]
            s.bulk_insert_mappings(Transaction, _tx_rows(inv))
        for row in pred:
            s.merge(Prediction(**row))
        for row in segs:
            s.merge(Segment(**row))
        s.add(
            ScoreRun(
                artifact_hash=new_hash, model_name=meta["production_churn_model"], n_customers=len(table), loaded_at=now
            )
        )
        s.commit()
    log.info("refreshed scores run_hash=%s customers=%d archived=%d", new_hash[:12], len(table), len(old))
    return True


if __name__ == "__main__":  # explicit refresh: python -m src.db.session --url <DATABASE_URL>
    import argparse
    import os

    ap = argparse.ArgumentParser(
        description="Refresh database scores from the current model artifacts (non-destructive)."
    )
    ap.add_argument("--url", default=os.environ.get("DATABASE_URL"))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    print("refreshed" if refresh_scores(make_engine(a.url), force=a.force) else "up to date")
