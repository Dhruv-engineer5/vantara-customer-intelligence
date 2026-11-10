"""HTTP client for the Vantara API used by the Streamlit dashboard.

Cache policy (regression-tested in tests/test_dashboard_cache.py):

- Public sample-workspace responses may share the process-wide 5-minute cache.
- Token-authenticated (private workspace) responses are NEVER cached and never
  served from the shared cache, and the access token is never part of a cache
  key. A shared cache keyed by anything weaker than the full credential can
  serve one user's data to another. That bug happened here once: the cached
  getter took ``_token``/``_dataset`` parameters, and Streamlit silently
  excludes underscore-prefixed parameters from the cache key, so every user
  shared one cache entry per endpoint. Do not reintroduce a shared cache for
  authenticated responses; if one is ever needed, it must key on the full
  credential, respect logout/expiry, and be covered by multi-user tests.
"""

from __future__ import annotations

import os
import time
from typing import Any

import requests
import streamlit as st

API = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")


def _request_json(path: str, headers: dict[str, str], params: dict[str, Any]) -> Any:
    """GET JSON with retries: the free API caps concurrent requests and answers
    the excess with 503 instead of queueing them."""
    last: requests.RequestException | None = None
    for attempt in range(4):
        try:
            r = requests.get(f"{API}{path}", params=params, headers=headers, timeout=30)
            if r.status_code == 503:
                raise requests.HTTPError("API busy (503)", response=r)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:  # noqa: PERF203
            last = exc
            time.sleep(1.5 * (attempt + 1))
    assert last is not None
    raise last


@st.cache_data(ttl=300, show_spinner=False)
def _get_public(path: str, **params: Any) -> Any:
    """Cached GET for the PUBLIC sample workspace only. Never receives a token."""
    return _request_json(path, {}, {**params, "dataset": "sample"})


def get(path: str, token: str | None = None, dataset: str = "sample", **params: Any) -> Any:
    """GET JSON from the API.

    token authorizes access to a private workspace; dataset selects it. When a
    token is present the request always goes to the API (never the shared
    cache), so one signed-in user can never receive another user's cached
    response. Only token-free sample-workspace calls use the 5-minute cache.
    """
    if token:
        return _request_json(path, {"Authorization": f"Bearer {token}"}, {**params, "dataset": dataset})
    return _get_public(path, **params)
