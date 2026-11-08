"""Measure single-customer prediction latency against a RUNNING API and write reports/runtime_checks.json.

Usage: python scripts/benchmark_api.py [API_URL]
"""

import json
import statistics
import sys
import time
from pathlib import Path

import requests

api = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")
body = {"recency_days": 50, "frequency": 4, "monetary_total": 900}
requests.post(f"{api}/predict", json=body, timeout=30)  # warm-up
ms = []
for _ in range(200):
    t = time.perf_counter()
    r = requests.post(f"{api}/predict", json=body, timeout=30)
    r.raise_for_status()
    ms.append((time.perf_counter() - t) * 1000)
ms.sort()
res = {
    "api_url": api,
    "requests": len(ms),
    "p50_ms": round(statistics.median(ms), 1),
    "p95_ms": round(ms[int(0.95 * len(ms)) - 1], 1),
    "max_ms": round(ms[-1], 1),
    "note": "client-observed latency including HTTP, SHAP explanation, model scoring and segment assignment",
}
out = Path(__file__).resolve().parents[1] / "reports" / "runtime_checks.json"
prev = json.loads(out.read_text()) if out.exists() else {}
prev["api_single_prediction"] = res
out.write_text(json.dumps(prev, indent=2))
print(json.dumps(res, indent=2))
