"""Architecture, ER and workflow diagrams drawn with matplotlib (PNG, saved to docs/)."""

from __future__ import annotations

from typing import Any

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from src.utils.config import ROOT  # noqa: E402


def _box(ax: Any, x: float, y: float, w: float, h: float, text: str, color: str = "#e8f0fe", fs: int = 9) -> None:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=color, ec="#335", lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def _arrow(ax: Any, x1: float, y1: float, x2: float, y2: float) -> None:
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#335"})


def architecture(path: Path) -> None:
    """One-page data flow: raw CSV/xlsx -> pipeline -> models -> API -> dashboard, with PostgreSQL."""
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.5)
    ax.axis("off")
    ax.set_title("Vantara Customer Behavior Prediction Platform - architecture", fontsize=13, weight="bold")
    layers = [
        ("DATA LAYER", 0.2, "#fff4d6"),
        ("PROCESSING LAYER", 3.4, "#e6f4ea"),
        ("MODELING LAYER", 6.6, "#fde7e9"),
        ("SERVING LAYER", 9.8, "#e8f0fe"),
    ]
    for name, x, c in layers:
        ax.add_patch(FancyBboxPatch((x, 0.4), 3.0, 5.5, boxstyle="round,pad=0.03", fc=c, ec="#999", lw=1, alpha=0.5))
        ax.text(x + 1.5, 5.65, name, ha="center", fontsize=10, weight="bold")
    _box(ax, 0.4, 4.0, 2.6, 1.0, "UCI Online Retail II\n(xlsx, 1.07M lines)\ndata/raw")
    _box(ax, 0.4, 2.3, 2.6, 1.0, "Cleaned transactions\n(793,850 lines)\ndata/interim")
    _box(ax, 0.4, 0.7, 2.6, 1.2, "PostgreSQL\ncustomers, transactions,\npredictions, segments", "#ffe0a3")
    _box(ax, 3.6, 4.0, 2.6, 1.0, "Validate + clean\n(src/data)")
    _box(ax, 3.6, 2.3, 2.6, 1.0, "Customer features\n(cutoff 2011-09-09)\nsrc/features")
    _box(ax, 3.6, 0.7, 2.6, 1.2, "Feature table +\npurchase sequences\ndata/processed")
    _box(ax, 6.8, 4.0, 2.6, 1.0, "6 ML + ANN + LSTM\nchurn models")
    _box(ax, 6.8, 2.8, 2.6, 0.9, "CLV regressors,\nautoencoder anomalies")
    _box(ax, 6.8, 1.6, 2.6, 0.9, "K-Means / GMM /\nDBSCAN segments")
    _box(ax, 6.8, 0.6, 2.6, 0.8, "SHAP / LIME / PDP\nmodels_artifacts")
    _box(ax, 10.0, 3.9, 2.6, 1.2, "FastAPI service\n/predict, /predict/batch,\n/model/metadata, /health")
    _box(ax, 10.0, 2.0, 2.6, 1.4, "Streamlit dashboard\nsegments, leaderboard,\nrevenue, SHAP panel, CSV", "#cfe0ff")
    _box(ax, 10.0, 0.6, 2.6, 0.9, "docker compose up\n(db + api + dashboard)", "#d9d9d9")
    for a in [
        (1.7, 4.0, 1.7, 3.3),
        (3.0, 4.5, 3.6, 4.5),
        (4.9, 4.0, 4.9, 3.3),
        (4.9, 2.3, 4.9, 1.9),
        (6.2, 1.3, 6.8, 4.3),
        (9.4, 4.4, 10.0, 4.5),
        (11.3, 3.9, 11.3, 3.4),
    ]:
        _arrow(ax, *a)
    ax.text(
        6.5,
        0.1,
        "The API reads and writes PostgreSQL (predictions, segments); the dashboard only talks to the API.",
        ha="center",
        fontsize=8.5,
    )
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def er(path: Path) -> None:
    """ER diagram of the four PostgreSQL tables."""
    fig, ax = plt.subplots(figsize=(11, 6.6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.6)
    ax.axis("off")
    ax.set_title("PostgreSQL schema (ER diagram)", fontsize=13, weight="bold", y=1.0)
    tables = {
        "customers": (
            0.4,
            2.2,
            [
                "PK customer_id",
                "country",
                "recency_days",
                "frequency",
                "monetary_total",
                "clv_historical",
                "engagement_score",
                "features (JSON)",
            ],
        ),
        "transactions": (
            4.2,
            0.4,
            ["PK id", "invoice", "FK customer_id", "invoice_date", "amount", "items", "is_return"],
        ),
        "predictions": (
            8.0,
            3.3,
            [
                "PK/FK customer_id",
                "churn_probability",
                "clv_predicted",
                "anomaly_score",
                "is_anomaly",
                "data_split",
                "model_name",
                "scored_at",
            ],
        ),
        "segments": (8.0, 0.3, ["PK/FK customer_id", "segment_id", "segment_name"]),
    }
    pos = {}
    for name, (x, y, cols) in tables.items():
        h = 0.34 * (len(cols) + 1)
        ax.add_patch(FancyBboxPatch((x, y), 2.7, h, boxstyle="square,pad=0", fc="#fff", ec="#335", lw=1.4))
        ax.add_patch(
            FancyBboxPatch((x, y + h - 0.34), 2.7, 0.34, boxstyle="square,pad=0", fc="#cfe0ff", ec="#335", lw=1.4)
        )
        ax.text(x + 1.35, y + h - 0.17, name, ha="center", va="center", weight="bold", fontsize=10)
        for i, c in enumerate(cols):
            ax.text(x + 0.1, y + h - 0.34 * (i + 1) - 0.17, c, va="center", fontsize=8.5)
        pos[name] = (x, y, h)
    cx, cy, ch = pos["customers"]
    for tgt, label in [("transactions", "1 : N"), ("predictions", "1 : 1"), ("segments", "1 : 1")]:
        tx, ty, th = pos[tgt]
        _arrow(ax, cx + 2.7, cy + ch / 2, tx, ty + th / 2)
        ax.text((cx + 2.7 + tx) / 2, (cy + ch / 2 + ty + th / 2) / 2 + 0.12, label, fontsize=8, color="#a33")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def workflow(path: Path) -> None:
    """Workflow diagram of the training pipeline and scoring flow."""
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    ax.set_title("End-to-end workflow: python -m src.pipeline, then docker compose up", fontsize=12, weight="bold")
    steps = [
        "Load 2 sheets\n+ validate",
        "Clean\n(IDs, returns,\nprice, dupes,\nIQR)",
        "Features\n(< cutoff only)\n+ leakage tests",
        "Split 70/15/15\n(stratified)",
        "Train + tune\n(CV on train)",
        "Select on val;\nreport test once",
        "Segments\n+ SHAP/LIME",
        "Seed\nPostgreSQL",
        "API +\ndashboard",
    ]
    w = 1.3
    for i, s in enumerate(steps):
        x = 0.2 + i * 1.42
        _box(ax, x, 1.6, w, 1.5, s, "#e6f4ea" if i < 7 else "#e8f0fe", fs=8)
        if i:
            _arrow(ax, x - 0.12, 2.35, x, 2.35)
    ax.text(
        6.5,
        0.8,
        "Every stage is scripted (no notebook-only steps) and logs to the console; config in config/config.yaml, seed 42.",  # noqa: E501
        ha="center",
        fontsize=9,
    )
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def run() -> None:
    """Draw all three diagrams into docs/."""
    d = ROOT / "docs"
    d.mkdir(exist_ok=True)
    architecture(d / "architecture_diagram.png")
    er(d / "er_diagram.png")
    workflow(d / "workflow_diagram.png")


if __name__ == "__main__":
    run()
