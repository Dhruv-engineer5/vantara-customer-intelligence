"""Dashboard HTTP client cache policy: private responses never share the cache.

Regression tests for the cross-user leak where the cached dashboard getter took
underscore-prefixed parameters, which Streamlit excludes from the cache key, so
one signed-in user's API responses were served to another user. These tests
exercise the REAL Streamlit cache in-process (two "sessions" = sequential calls
in this process, which is exactly how the shared server-side cache behaves).
"""

from __future__ import annotations

import pytest
import streamlit as st

from frontend import api_client


class _Resp:
    def __init__(self, payload):
        self.status_code = 200
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


@pytest.fixture()
def http(monkeypatch):
    """Fake requests.get that answers with the caller's own Authorization header."""
    calls = []

    def fake_get(url, params=None, headers=None, timeout=None):
        auth = (headers or {}).get("Authorization", "anon")
        calls.append({"url": url, "params": dict(params or {}), "auth": auth})
        return _Resp({"served_for": auth, "dataset": (params or {}).get("dataset")})

    monkeypatch.setattr(api_client.requests, "get", fake_get)
    st.cache_data.clear()
    yield calls
    st.cache_data.clear()


def test_two_users_same_endpoint_never_share(http):
    a = api_client.get("/datasets", token="token-A")
    b = api_client.get("/datasets", token="token-B")
    assert a["served_for"] == "Bearer token-A"
    assert b["served_for"] == "Bearer token-B"
    assert [c["auth"] for c in http] == ["Bearer token-A", "Bearer token-B"]


def test_authed_calls_always_hit_api_no_cache_hits(http):
    api_client.get("/customers", token="token-A", dataset="ds-1")
    api_client.get("/customers", token="token-A", dataset="ds-1")
    assert len(http) == 2, "authed responses must never be served from cache"


def test_same_endpoint_different_dataset_both_reach_api(http):
    api_client.get("/customers", token="token-A", dataset="ds-1", limit=5)
    api_client.get("/customers", token="token-B", dataset="ds-2", limit=5)
    assert [c["params"]["dataset"] for c in http] == ["ds-1", "ds-2"]
    assert [c["auth"] for c in http] == ["Bearer token-A", "Bearer token-B"]


def test_public_sample_is_cached_but_never_served_to_authed(http):
    api_client.get("/customers", limit=5)
    api_client.get("/customers", limit=5)
    assert len(http) == 1, "public sample responses should share the 5-minute cache"
    # An authed call for the same path+params must not receive the public entry.
    r = api_client.get("/customers", token="token-A", dataset="sample", limit=5)
    assert r["served_for"] == "Bearer token-A"
    assert len(http) == 2
    # And the public cache must not absorb the authed response either.
    r2 = api_client.get("/customers", limit=5)
    assert r2["served_for"] == "anon"
    assert len(http) == 2, "public entry must still be the original anon one"


def test_public_cache_clear_forces_refetch(http):
    api_client.get("/customers", limit=5)
    st.cache_data.clear()
    api_client.get("/customers", limit=5)
    assert len(http) == 2
