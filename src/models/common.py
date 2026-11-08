"""Shared data splitting, metrics and experiment logging."""

from __future__ import annotations

from typing import Any

import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    mean_squared_error,
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.features.build import FEATURE_COLUMNS
from src.utils.config import rpath


def load_table(cfg: dict) -> pd.DataFrame:
    """Load the processed customer-level table."""
    return pd.read_parquet(rpath(cfg, "features"))


def make_split(table: pd.DataFrame, cfg: dict) -> dict[str, np.ndarray]:
    """Positional index arrays for the split stored in the table (made before any statistic was fitted).

    Tables without a stored assignment (unit tests) fall back to the same stratified 70/15/15 logic.
    """
    if "data_split" in table.columns:
        return {k: np.where(table["data_split"].to_numpy() == k)[0] for k in ("train", "val", "test")}
    seed = cfg["seed"]
    idx = np.arange(len(table))
    s = cfg["split"]
    tr, rest = train_test_split(idx, train_size=s["train"], stratify=table["churn"], random_state=seed)
    rel = s["val"] / (s["val"] + s["test"])
    va, te = train_test_split(rest, train_size=rel, stratify=table["churn"].iloc[rest], random_state=seed)
    return {"train": tr, "val": va, "test": te}


def xy(table: pd.DataFrame, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Feature matrix and churn label for the given rows."""
    return table[FEATURE_COLUMNS].iloc[idx].to_numpy(dtype=float), table["churn"].iloc[idx].to_numpy()


def clf_metrics(y: Any, proba: Any, thr: float = 0.5) -> dict[str, float]:
    """ROC-AUC, recall, precision, F1 and accuracy for the churned (positive) class."""
    pred = (proba >= thr).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, proba)),
        "recall": float(recall_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred)),
        "accuracy": float(accuracy_score(y, pred)),
        **dict(zip(("tn", "fp", "fn", "tp"), (int(v) for v in confusion_matrix(y, pred, labels=[0, 1]).ravel()))),
    }


def reg_metrics(y: Any, pred: Any) -> dict[str, float]:
    """R2, MAE and RMSE on the raw scale."""
    return {
        "r2": float(r2_score(y, pred)),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(mean_squared_error(y, pred) ** 0.5),
    }


class RunLog:
    """CSV experiment log, rewritten per training run, so comparison tables regenerate from it."""

    FIELDS = [
        "task",
        "model",
        "split",
        "roc_auc",
        "recall",
        "precision",
        "f1",
        "accuracy",
        "r2",
        "mae",
        "rmse",
        "tn",
        "fp",
        "fn",
        "tp",
        "train_seconds",
        "params",
    ]

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as fh:
            csv.DictWriter(fh, self.FIELDS).writeheader()

    def add(self, task: str, model: str, split: str, metrics: dict, seconds: float, params: dict) -> None:
        """Write one run row."""
        row = {k: "" for k in self.FIELDS}
        row.update(
            {"task": task, "model": model, "split": split, "train_seconds": round(seconds, 2), "params": str(params)}
        )
        row.update({k: round(v, 4) for k, v in metrics.items() if k in row})
        with open(self.path, "a", newline="") as fh:
            csv.DictWriter(fh, self.FIELDS).writerow(row)


class Timer:
    """Tiny context timer."""

    def __enter__(self) -> "Timer":
        self.t = time.time()
        return self

    def __exit__(self, *a: Any) -> None:
        self.seconds = time.time() - self.t
