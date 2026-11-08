"""Create a local .env with a random database password (never committed, never typed into chat).

Run once before `docker compose up --build`:  python scripts/setup_env.py
Existing .env files are left untouched.
"""

from __future__ import annotations

import re
import secrets
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    """Write .env from .env.example with a generated POSTGRES_PASSWORD, unless .env already exists."""
    target = ROOT / ".env"
    if target.exists():
        print(".env already exists; leaving it unchanged.")
        return 0
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    text = re.sub(r"(?m)^POSTGRES_PASSWORD=.*$", "POSTGRES_PASSWORD=" + secrets.token_urlsafe(24), text)
    target.write_text(text, encoding="utf-8")
    print("Created .env with a random local database password.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
