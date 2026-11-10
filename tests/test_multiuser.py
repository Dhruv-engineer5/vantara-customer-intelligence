"""Multi-user workspace tests: auth, tenant isolation, upload validation.

Auth uses the documented test path: VANTARA_AUTH_TEST_SECRET accepts locally minted
HS256 tokens, so no Supabase project is needed. The database is the shared real-Postgres
fixture; these tests clean up the datasets they create.
"""

from __future__ import annotations

import time

import jwt
import pandas as pd
import pytest

SECRET = "vantara-test-secret"  # noqa: S105 - test-only key, never a real credential

USER_A = "user-aaaa"
USER_B = "user-bbbb"


@pytest.fixture(autouse=True)
def _test_auth(monkeypatch):
    monkeypatch.setenv("VANTARA_AUTH_TEST_SECRET", SECRET)


def tok(user: str) -> dict[str, str]:
    token = jwt.encode(
        {"sub": user, "aud": "authenticated", "exp": int(time.time()) + 3600}, SECRET, algorithm="HS256"
    )
    return {"Authorization": f"Bearer {token}"}


def make_csv(n_customers: int = 30, rows_per: int = 12) -> bytes:
    """A synthetic but schema-compatible transaction CSV (string customer ids on purpose)."""
    rng = pd.date_range("2026-05-01", periods=120, freq="D")
    rows = []
    inv = 1000
    for c in range(n_customers):
        for i in range(rows_per):
            day = rng[(c * 7 + i * 9) % len(rng)]
            rows.append(
                {
                    "Invoice": str(inv := inv + 1),
                    "StockCode": f"S{(c + i) % 15}",
                    "Description": f"TEST PRODUCT {(c + i) % 15}",
                    "Quantity": 1 + (i % 5),
                    "InvoiceDate": day.strftime("%Y-%m-%d %H:%M:%S"),
                    "Price": round(2.5 + (i % 7) * 1.5, 2),
                    "Customer ID": f"U{c + 1000}",
                    "Country": "India",
                }
            )
    return pd.DataFrame(rows).to_csv(index=False).encode()


@pytest.fixture()
def uploaded(client):
    """A ready dataset owned by USER_A; deleted after the test."""
    r = client.post(
        "/datasets", headers=tok(USER_A), files={"file": ("t.csv", make_csv(), "text/csv")}, data={"name": "Test data"}
    )
    assert r.status_code == 201, r.text
    ds = r.json()["dataset"]
    yield ds
    client.delete(f"/datasets/{ds['dataset_id']}", headers=tok(USER_A))


def test_sample_stays_public(client):
    r = client.get("/customers", params={"limit": 5})
    assert r.status_code == 200
    assert len(r.json()) == 5
    r = client.get("/segments")
    assert r.status_code == 200


def test_private_reads_require_signin(client, uploaded):
    ds = uploaded["dataset_id"]
    for path in ("/customers", "/customers/leaderboard", "/segments", "/revenue/trend"):
        assert client.get(path, params={"dataset": ds}).status_code == 401, path
    assert client.get("/datasets").status_code == 401


def test_upload_list_and_read_own_dataset(client, uploaded):
    ds = uploaded
    assert ds["status"] == "ready"
    assert ds["n_customers"] == 30
    mine = client.get("/datasets", headers=tok(USER_A)).json()
    assert [d["dataset_id"] for d in mine] == [ds["dataset_id"]]
    custs = client.get("/customers", headers=tok(USER_A), params={"dataset": ds["dataset_id"], "limit": 100}).json()
    assert len(custs) == 30
    assert custs[0]["data_split"] == "user"
    cid = custs[0]["customer_id"]
    one = client.get(f"/customers/{cid}", headers=tok(USER_A), params={"dataset": ds["dataset_id"]})
    assert one.status_code == 200
    assert one.json()["segment_name"]
    ex = client.get(f"/customers/{cid}/explain", headers=tok(USER_A), params={"dataset": ds["dataset_id"]})
    assert ex.status_code == 200
    assert ex.json()["top_factors"]
    segs = client.get("/segments", headers=tok(USER_A), params={"dataset": ds["dataset_id"]}).json()
    assert sum(s["customers"] for s in segs) == 30
    rev = client.get("/revenue/trend", headers=tok(USER_A), params={"dataset": ds["dataset_id"]}).json()
    assert rev["history"]
    rec = client.get(f"/customers/{cid}/recommendations", headers=tok(USER_A), params={"dataset": ds["dataset_id"]})
    assert rec.status_code == 200
    lb = client.get("/customers/leaderboard", headers=tok(USER_A), params={"dataset": ds["dataset_id"]}).json()
    assert lb and "value_at_risk" in lb[0]


def test_tenant_isolation(client, uploaded):
    ds = uploaded["dataset_id"]
    # USER_B cannot see, read, or delete USER_A's dataset (404, not 403: ids are not enumerable).
    assert client.get("/customers/U1000", headers=tok(USER_B), params={"dataset": ds}).status_code == 404
    assert client.get("/customers", headers=tok(USER_B), params={"dataset": ds}).status_code == 404
    assert client.get("/segments", headers=tok(USER_B), params={"dataset": ds}).status_code == 404
    assert client.delete(f"/datasets/{ds}", headers=tok(USER_B)).status_code == 404
    assert client.get("/datasets", headers=tok(USER_B)).json() == []
    # A forged/expired token gets a 401.
    bad = {"Authorization": "Bearer not-a-real-token"}
    assert client.get("/datasets", headers=bad).status_code == 401


def test_delete_removes_everything(client, uploaded):
    ds = uploaded["dataset_id"]
    assert client.delete(f"/datasets/{ds}", headers=tok(USER_A)).status_code == 204
    assert client.get("/customers", headers=tok(USER_A), params={"dataset": ds}).status_code == 404
    assert client.get("/datasets", headers=tok(USER_A)).json() == []


def test_bad_csv_rejected(client):
    bad = pd.DataFrame({"a": [1, 2]}).to_csv(index=False).encode()
    r = client.post("/datasets", headers=tok(USER_A), files={"file": ("x.csv", bad, "text/csv")})
    assert r.status_code == 422
    assert "missing columns" in r.json()["detail"]
    # A failed upload leaves a failed dataset row, which the owner can see and delete.
    mine = client.get("/datasets", headers=tok(USER_A)).json()
    failed = [d for d in mine if d["status"] == "failed"]
    for d in failed:
        client.delete(f"/datasets/{d['dataset_id']}", headers=tok(USER_A))


def test_oversize_csv_rejected(client, monkeypatch):
    monkeypatch.setattr("api.routers.datasets.MAX_FILE_BYTES", 100)
    r = client.post(
        "/datasets", headers=tok(USER_A), files={"file": ("x.csv", b"a,b,c\n" * 50, "text/csv")}
    )
    assert r.status_code == 413


def test_dataset_cap(client, uploaded, monkeypatch):
    monkeypatch.setattr("api.routers.datasets.MAX_DATASETS_PER_USER", 1)
    r = client.post(
        "/datasets", headers=tok(USER_A), files={"file": ("t.csv", make_csv(), "text/csv")}
    )
    assert r.status_code == 409


def test_expired_token_rejected(client):
    expired = jwt.encode(
        {"sub": USER_A, "aud": "authenticated", "exp": int(time.time()) - 10}, SECRET, algorithm="HS256"
    )
    r = client.get("/datasets", headers={"Authorization": f"Bearer {expired}"})
    assert r.status_code == 401


def test_wrong_audience_rejected(client):
    bad = jwt.encode(
        {"sub": USER_A, "aud": "someone-else", "exp": int(time.time()) + 3600}, SECRET, algorithm="HS256"
    )
    r = client.get("/datasets", headers={"Authorization": f"Bearer {bad}"})
    assert r.status_code == 401


def test_test_secret_never_active_with_supabase_url(client, monkeypatch):
    """If SUPABASE_URL is set (any real deployment), the test secret cannot validate tokens."""
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    r = client.get("/datasets", headers=tok(USER_A))
    # With a JWKS configured the test token is rejected (401 against the unreachable/invalid
    # project keys, or 503 while the key endpoint cannot be fetched) - but never authorized.
    assert r.status_code in (401, 503)
    assert r.status_code != 200


def _delete_failed(client):
    """Failed uploads leave 'failed' dataset rows; remove them so caps stay predictable."""
    for d in client.get("/datasets", headers=tok(USER_A)).json():
        if d["status"] == "failed":
            client.delete(f"/datasets/{d['dataset_id']}", headers=tok(USER_A))


def test_tenant_isolation_all_read_endpoints(client, uploaded):
    """USER_B gets 404 (not 403, ids are not enumerable) on every dataset-scoped read."""
    ds = uploaded["dataset_id"]
    cid = "U1000"  # a real customer id inside USER_A's uploaded dataset
    for path in (
        "/customers",
        "/customers/leaderboard",
        "/segments",
        "/revenue/trend",
        f"/customers/{cid}",
        f"/customers/{cid}/explain",
        f"/customers/{cid}/recommendations",
    ):
        r = client.get(path, headers=tok(USER_B), params={"dataset": ds})
        assert r.status_code == 404, f"{path} returned {r.status_code} for a foreign dataset"
    # Batch scoring takes an uploaded CSV, not a dataset id, so it has no cross-user case:
    # it never reads private workspace rows (verified by code inspection of api/routers/predict.py).


def test_upload_too_few_customers_rejected(client):
    """Upload minimum caps: under 20 customers is a 422 with a clear reason."""
    r = client.post(
        "/datasets",
        headers=tok(USER_A),
        files={"file": ("t.csv", make_csv(n_customers=5, rows_per=45), "text/csv")},
    )
    assert r.status_code == 422
    assert "customers" in r.json()["detail"]
    _delete_failed(client)


def test_upload_too_short_history_rejected(client):
    """Upload minimum caps: under 30 days of history is a 422 with a clear reason."""
    rows = []
    inv = 5000
    for c in range(25):  # 25 customers passes the customer minimum
        for i in range(10):  # 250 rows passes the row minimum
            day = pd.Timestamp("2026-09-01") + pd.Timedelta(days=(c + i) % 10)  # 10-day span only
            rows.append(
                {
                    "Invoice": str(inv := inv + 1),
                    "StockCode": f"S{(c + i) % 15}",
                    "Description": f"TEST PRODUCT {(c + i) % 15}",
                    "Quantity": 1 + (i % 5),
                    "InvoiceDate": day.strftime("%Y-%m-%d %H:%M:%S"),
                    "Price": round(2.5 + (i % 7) * 1.5, 2),
                    "Customer ID": f"H{c + 2000}",
                    "Country": "India",
                }
            )
    csv_bytes = pd.DataFrame(rows).to_csv(index=False).encode()
    r = client.post("/datasets", headers=tok(USER_A), files={"file": ("t.csv", csv_bytes, "text/csv")})
    assert r.status_code == 422
    assert "days" in r.json()["detail"]
    _delete_failed(client)
