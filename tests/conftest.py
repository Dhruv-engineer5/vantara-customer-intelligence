"""Shared fixtures: synthetic transactions and a real PostgreSQL database for API tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture()
def tx() -> pd.DataFrame:
    """Small hand-checkable transaction table."""
    d = pd.Timestamp
    rows = [
        # customer 1: three orders before cutoff 2011-01-01, one after (inside window)
        ("A1", "S1", "HEART LIGHT", 2, d("2010-10-01"), 5.0, 1, "United Kingdom"),
        ("A2", "S1", "HEART LIGHT", 1, d("2010-11-01"), 5.0, 1, "United Kingdom"),
        ("A3", "S2", "TEA MUG", 4, d("2010-12-20"), 2.5, 1, "United Kingdom"),
        ("A4", "S2", "TEA MUG", 4, d("2011-01-15"), 2.5, 1, "United Kingdom"),
        # customer 2: one old order, return, nothing after
        ("B1", "S3", "PAPER BAG", 10, d("2010-06-01"), 1.0, 2, "Germany"),
        ("CB2", "S3", "PAPER BAG", -2, d("2010-06-10"), 1.0, 2, "Germany"),
        # customer 3: first purchase only after cutoff (must not appear)
        ("C1", "S1", "HEART LIGHT", 1, d("2011-02-01"), 5.0, 3, "France"),
    ]
    df = pd.DataFrame(
        rows,
        columns=["invoice", "stock_code", "description", "quantity", "invoice_date", "price", "customer_id", "country"],
    )
    df["is_return"] = df["invoice"].str.startswith("C")
    df["line_total"] = df["quantity"] * df["price"]
    return df


@pytest.fixture(scope="session")
def pg_engine():
    """Real PostgreSQL: DATABASE_URL if set (e.g. in docker compose), else a pgserver-managed instance."""
    from sqlalchemy import create_engine, text

    url = os.environ.get("DATABASE_URL")
    if not url:
        pgserver = pytest.importorskip("pgserver")
        import tempfile

        d = tempfile.mkdtemp(prefix="vantara_pg_")
        srv = pgserver.get_server(d, cleanup_mode="stop")
        srv.psql("CREATE DATABASE vantara_test;")
        url = f"postgresql+psycopg2://postgres:@/vantara_test?host={d}"
    else:
        # Never run destructive test setup against the live application database: use a separate one.
        from sqlalchemy.engine import make_url

        admin = create_engine(url, isolation_level="AUTOCOMMIT")
        with admin.connect() as c:
            if not c.execute(text("SELECT 1 FROM pg_database WHERE datname='vantara_test'")).scalar():
                c.execute(text("CREATE DATABASE vantara_test"))
        admin.dispose()
        url = make_url(url).set(database="vantara_test").render_as_string(hide_password=False)
    eng = create_engine(url)
    with eng.connect() as c:
        c.execute(text("SELECT 1"))
    return eng


@pytest.fixture(scope="session")
def client(pg_engine):
    """FastAPI TestClient wired to a freshly seeded real PostgreSQL."""
    from fastapi.testclient import TestClient

    from api.deps import load_state
    from api.main import app
    from src.db.models import Base

    Base.metadata.drop_all(pg_engine)
    app.state.s = load_state(pg_engine)
    with TestClient(app) as c:
        yield c
