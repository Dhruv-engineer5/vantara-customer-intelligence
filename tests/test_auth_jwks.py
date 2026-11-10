"""Deployed auth path: Supabase-style ES256 JWTs verified through the JWKS branch.

Suite-based, not live: this exercises the real production code path in
api.auth._decode (JWKS key resolution, ES256/RS256 algorithm whitelist, audience
and issuer validation) with an in-test ES256 keypair standing in for the
Supabase project's signing key. Complements the live staging checks, which used
real Supabase-issued tokens via the actual hosted JWKS endpoint.
"""

from __future__ import annotations

import time

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

SUPABASE_BASE = "https://example.supabase.co"
ISSUER = f"{SUPABASE_BASE}/auth/v1"


class _StubJWKClient:
    """Stands in for jwt.PyJWKClient: returns the in-test public key for any token."""

    def __init__(self, public_pem: bytes):
        self._public_pem = public_pem

    def get_signing_key_from_jwt(self, token: str):
        return type("SigningKey", (), {"key": self._public_pem})()


@pytest.fixture()
def jwks_path(monkeypatch):
    """Configure the deployed JWKS path with an in-test ES256 keypair.

    VANTARA_AUTH_TEST_SECRET is explicitly removed: with SUPABASE_URL set the code
    must authenticate via JWKS alone, proving the deployed path stands on its own.
    """
    priv = ec.generate_private_key(ec.SECP256R1())
    pub_pem = priv.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    monkeypatch.setenv("SUPABASE_URL", SUPABASE_BASE)
    monkeypatch.delenv("VANTARA_AUTH_TEST_SECRET", raising=False)
    monkeypatch.setattr("api.auth._jwks_client", lambda url: _StubJWKClient(pub_pem))
    return priv


def _mint(priv, **overrides) -> str:
    claims = {"sub": "user-es256", "aud": "authenticated", "iss": ISSUER, "exp": int(time.time()) + 3600}
    claims.update(overrides)
    return jwt.encode(claims, priv, algorithm="ES256")


def test_es256_jwks_token_accepted(client, jwks_path):
    """A valid ES256 token with correct audience and issuer authenticates on the deployed path."""
    r = client.get("/datasets", headers={"Authorization": f"Bearer {_mint(jwks_path)}"})
    assert r.status_code == 200
    assert r.json() == []  # a brand-new user owns no datasets


def test_hs256_token_rejected_on_jwks_path(client, jwks_path):
    """HS256 (including the test-secret shape) is outside the production algorithm whitelist."""
    bad = jwt.encode(
        {"sub": "user-es256", "aud": "authenticated", "iss": ISSUER, "exp": int(time.time()) + 3600},
        "vantara-test-secret",  # noqa: S106 - proving this exact secret cannot authenticate here
        algorithm="HS256",
    )
    r = client.get("/datasets", headers={"Authorization": f"Bearer {bad}"})
    assert r.status_code == 401


def test_wrong_issuer_rejected(client, jwks_path):
    r = client.get("/datasets", headers={"Authorization": f"Bearer {_mint(jwks_path, iss='https://evil.example')}"})
    assert r.status_code == 401


def test_expired_es256_rejected(client, jwks_path):
    r = client.get("/datasets", headers={"Authorization": f"Bearer {_mint(jwks_path, exp=int(time.time()) - 120)}"})
    assert r.status_code == 401
