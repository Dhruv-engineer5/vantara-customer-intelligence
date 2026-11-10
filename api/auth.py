"""Supabase Auth (GoTrue) access-token verification.

The dashboard signs users in against the Supabase project's Auth REST endpoint and sends the
resulting access token as `Authorization: Bearer <token>`. The API verifies it against the
project's public JWKS endpoint - verification needs no shared secret, only the project URL.
Issuer, audience, expiry and algorithm are all enforced.

Configuration: SUPABASE_JWKS_URL, or SUPABASE_URL (the JWKS path and issuer are derived).

Test override: when (and only when) SUPABASE_URL / SUPABASE_JWKS_URL are NOT set, the local
test suite may set VANTARA_AUTH_TEST_SECRET to accept locally minted HS256 tokens. On a real
deployment SUPABASE_URL is always set, which disables this path no matter what environment
or request input says - it cannot be activated by a client or by misconfiguration.
"""

from __future__ import annotations

import functools
import os

import jwt
from fastapi import HTTPException, Request
from jwt import PyJWKClient

# Production accepts only the asymmetric algorithms Supabase signing keys use. HS256 is
# deliberately absent: it only ever appears via the test override below.
PROD_ALGORITHMS = ["ES256", "RS256"]
AUDIENCE = "authenticated"


@functools.lru_cache(maxsize=4)
def _jwks_client(url: str) -> PyJWKClient:
    return PyJWKClient(url, cache_keys=True, lifespan=600)


def _jwks_url() -> str | None:
    url = os.environ.get("SUPABASE_JWKS_URL")
    if url:
        return url
    base = os.environ.get("SUPABASE_URL", "").rstrip("/")
    return f"{base}/auth/v1/.well-known/jwks.json" if base else None


def _issuer() -> str | None:
    base = os.environ.get("SUPABASE_URL", "").rstrip("/")
    if base:
        return f"{base}/auth/v1"
    jwks = os.environ.get("SUPABASE_JWKS_URL", "")
    suffix = "/.well-known/jwks.json"
    return jwks[: -len(suffix)] if jwks.endswith(suffix) else None


def _decode(token: str) -> dict:
    """Verify a Supabase access token and return its claims (401 on any failure)."""
    url = _jwks_url()
    if url and _issuer() is None:
        raise HTTPException(503, "sign-in is misconfigured on this deployment")
    if not url:
        # Test-only path: no Supabase project configured at all (local dev / CI).
        test_secret = os.environ.get("VANTARA_AUTH_TEST_SECRET")
        if not test_secret:
            raise HTTPException(503, "sign-in is not configured on this deployment")
        try:
            return jwt.decode(token, test_secret, algorithms=["HS256"], audience=AUDIENCE)
        except jwt.PyJWTError as exc:
            raise HTTPException(401, "invalid or expired sign-in token") from exc
    try:
        key = _jwks_client(url).get_signing_key_from_jwt(token).key
        return jwt.decode(
            token, key, algorithms=PROD_ALGORITHMS, audience=AUDIENCE, issuer=_issuer(), leeway=30
        )
    except jwt.PyJWTError as exc:
        raise HTTPException(401, f"invalid or expired sign-in token: {exc.__class__.__name__}") from exc


def current_user(request: Request) -> str | None:
    """The signed-in user's id, or None when no Bearer token is present."""
    header = request.headers.get("authorization", "")
    if not header.lower().startswith("bearer "):
        return None
    return str(_decode(header[7:].strip()).get("sub") or "") or None


def require_user(request: Request) -> str:
    """The signed-in user's id; 401 when the request is not authenticated."""
    user = current_user(request)
    if not user:
        raise HTTPException(401, "sign in to use your own datasets")
    return user
