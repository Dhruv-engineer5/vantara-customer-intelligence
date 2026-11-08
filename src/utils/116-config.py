"""Config loader and shared helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load config/config.yaml (or a given path) into a dict."""
    p = Path(path) if path else ROOT / "config" / "config.yaml"
    with open(p, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def rpath(cfg: dict[str, Any], key: str) -> Path:
    """Resolve a configured relative path against the project root.

    MODEL_ARTIFACTS_DIR (environment) moves the model-artifact folder, and every configured path inside it, elsewhere.
    """
    rel = cfg["paths"][key]
    override = os.environ.get("MODEL_ARTIFACTS_DIR")
    art = cfg["paths"]["artifacts"]
    if override and (rel == art or rel.startswith(art + "/")):
        return (ROOT / override) / rel[len(art) :].lstrip("/")
    return ROOT / rel


def load_dotenv(path: Path | None = None) -> None:
    """Load KEY=VALUE lines from the git-ignored .env file into os.environ (existing variables win)."""
    p = path or ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def database_url(cfg: dict[str, Any]) -> str:
    """Database URL from DATABASE_URL, else built from POSTGRES_* values in .env; never from committed credentials."""
    load_dotenv()
    url = os.environ.get(cfg["database"]["url_env"]) or cfg["database"].get("default_url", "")
    if url:
        return url
    pw = os.environ.get("POSTGRES_PASSWORD")
    if not pw or pw == "change-me-locally":
        raise RuntimeError(
            "No database credentials configured. Set DATABASE_URL, or run `python scripts/setup_env.py` "
            "to create a .env file with a random local password."
        )
    user, db = os.environ.get("POSTGRES_USER", "vantara"), os.environ.get("POSTGRES_DB", "vantara")
    host, port = os.environ.get("POSTGRES_HOST", "localhost"), os.environ.get("POSTGRES_PORT", "5432")
    return f"postgresql+psycopg2://{user}:{pw}@{host}:{port}/{db}"
