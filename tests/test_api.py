"""API integration tests against a real PostgreSQL instance."""

import io
import time

import pytest


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok", "database": "ok", "model_loaded": True}


def test_metadata_has_targets(client):
    m = client.get("/model/metadata").json()
    assert m["production_churn_model"] in ("xgboost", "lightgbm")
    assert set(m["targets_met"]) == {"churn_auc", "churn_recall", "clv_r2"}


def test_customer_lookup_and_404(client):
    r = client.get("/customers/12347")
    assert r.status_code == 200 and 0 <= r.json()["churn_probability"] <= 1
    assert client.get("/customers/1").status_code == 404


def test_customer_filters(client):
    rows = client.get("/customers", params={"risk": "high", "limit": 50}).json()
    assert rows and all(r["churn_probability"] >= 0.7 for r in rows)
    assert client.get("/customers", params={"value_tier": "bogus"}).status_code == 422


def test_leaderboard_sorted(client):
    lb = client.get("/customers/leaderboard", params={"n": 10}).json()
    v = [r["value_at_risk"] for r in lb]
    assert v == sorted(v, reverse=True) and len(lb) == 10


def test_explain(client):
    r = client.get("/customers/12347/explain").json()
    assert r["top_factors"] and "Churn risk" in r["explanation"]


def test_predict_single_and_validation(client):
    ok = client.post("/predict", json={"recency_days": 400, "frequency": 1, "monetary_total": 50})
    assert ok.status_code == 200
    body = ok.json()
    assert body["churn_risk_level"] in ("low", "moderate", "high") and body["segment_name"]
    hot = client.post("/predict", json={"recency_days": 2, "frequency": 30, "monetary_total": 20000}).json()
    assert hot["churn_probability"] < body["churn_probability"]  # predictions change with inputs
    assert client.post("/predict", json={"recency_days": -5, "frequency": 1, "monetary_total": 1}).status_code == 422
    assert (
        client.post("/predict", json={"recency_days": 1, "frequency": 1, "monetary_total": 1, "bogus": 1}).status_code
        == 422
    )
    assert (
        client.post(
            "/predict", json={"recency_days": 1, "frequency": 1, "monetary_total": 1, "extra_features": {"nope": 1}}
        ).status_code
        == 422
    )


@pytest.mark.parametrize("fmt,ctype", [("json", "application/json"), ("csv", "text/csv"), ("pdf", "application/pdf")])
def test_batch_formats(client, fmt, ctype):
    csv = b"customer_id\n12347\n12348\n"
    r = client.post("/predict/batch", params={"format": fmt}, files={"file": ("x.csv", io.BytesIO(csv), "text/csv")})
    assert r.status_code == 200 and ctype in r.headers["content-type"]
    if fmt == "pdf":
        assert r.content.startswith(b"%PDF")


def test_batch_feature_csv_and_errors(client):
    csv = b"recency_days,frequency,monetary_total\n10,5,500\n300,1,40\n"
    r = client.post("/predict/batch", files={"file": ("x.csv", io.BytesIO(csv), "text/csv")})
    rows = r.json()
    assert r.status_code == 200 and len(rows) == 2 and rows[0]["churn_probability"] < rows[1]["churn_probability"]
    bad = client.post("/predict/batch", files={"file": ("x.csv", io.BytesIO(b"a,b\n1,2\n"), "text/csv")})
    assert bad.status_code == 422
    unk = client.post("/predict/batch", files={"file": ("x.csv", io.BytesIO(b"customer_id\n1\n"), "text/csv")})
    assert unk.status_code == 422


def test_segments_and_revenue(client):
    assert len(client.get("/segments").json()) >= 3
    t = client.get("/revenue/trend").json()
    assert t["history"] and len(t["forecast"]) == 3


def test_single_prediction_latency(client):
    """Single-customer prediction p95 under the 400 ms brief target (in-process, real model + DB)."""
    body = {"recency_days": 50, "frequency": 4, "monetary_total": 900}
    client.post("/predict", json=body)  # warm-up
    times = []
    for _ in range(30):
        t = time.perf_counter()
        assert client.post("/predict", json=body).status_code == 200
        times.append((time.perf_counter() - t) * 1000)
    times.sort()
    assert times[int(0.95 * len(times)) - 1] < 400
