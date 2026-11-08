"""Training-code tests on tiny settings, run against temp output dirs so shipped artifacts stay untouched."""

import copy
import shutil

import numpy as np
import pandas as pd
import pytest

from src.models import classical, deep
from src.utils.config import ROOT, load_config


@pytest.fixture()
def tiny_cfg(tmp_path):
    cfg = copy.deepcopy(load_config())
    for k in ("artifacts", "reports"):
        (tmp_path / k).mkdir()
        cfg["paths"][k] = str(tmp_path / k)
    cfg["paths"]["scores"] = str(tmp_path / "scores.parquet")
    cfg["paths"]["segments"] = str(tmp_path / "segments.parquet")
    cfg["models"].update({"n_iter_search": 1, "cv_folds": 2})
    cfg["models"]["deep"].update({"epochs": 2, "patience": 1})
    return cfg


def test_classical_tune_runs(tiny_cfg):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    y = (X[:, 0] + rng.normal(scale=0.5, size=200) > 0).astype(int)
    for name, (est, grid) in classical.churn_candidates(tiny_cfg).items():
        best, params, secs = classical.tune(name, est, grid, X, y, tiny_cfg)
        assert best.predict_proba(X).shape == (200, 2)
    for name, (est, grid) in classical.clv_candidates(tiny_cfg).items():
        best, _, _ = classical.tune(name, est, grid, X, X[:, 1] ** 2, tiny_cfg, scoring="r2")
        assert best.predict(X).shape == (200,)


def test_deep_models_learn_a_signal(tiny_cfg):
    tiny_cfg["models"]["deep"].update({"epochs": 60, "patience": 10})
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 6)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    deep.set_seed(1)
    m, hist, _ = deep.fit_binary(deep.ANN(6, [16], 0.1), X[:300], y[:300], X[300:], y[300:], tiny_cfg, 1.0)
    p = deep.predict_proba(m, X[300:])
    assert ((p > 0.5) == y[300:]).mean() > 0.85
    assert hist["val_loss"][-1] < hist["val_loss"][0]
    S = rng.normal(size=(300, 5, 3)).astype(np.float32)
    ys = (S[:, -1, 0] > 0).astype(int)
    lm, _, _ = deep.fit_binary(deep.LSTMClassifier(3, 8, 0.1), S[:200], ys[:200], S[200:], ys[200:], tiny_cfg, 1.0)
    assert deep.predict_proba(lm, S[200:]).shape == (100,)
    ae, thr, _, _ = deep.fit_autoencoder(X[:300], X[300:], tiny_cfg)
    assert thr > 0 and deep.recon_error(ae, X).shape == (400,)


def test_segmentation_and_explainability_end_to_end(tiny_cfg):
    from src.explainability import explain
    from src.segmentation import cluster

    seg = cluster.run(tiny_cfg)
    assert seg["segment_id"].nunique() >= 3
    shutil.copy(ROOT / "models_artifacts" / "churn_model.joblib", tiny_cfg["paths"]["artifacts"])
    shutil.copy(ROOT / "data/processed/customer_scores.parquet", tiny_cfg["paths"]["scores"])
    ex = explain.run(tiny_cfg)
    assert {"low_risk", "high_risk", "borderline", "lime_vs_shap"} <= set(ex)
    assert ex["low_risk"]["probability"] < ex["high_risk"]["probability"]


def test_full_train_run_on_tiny_settings(tiny_cfg):
    from src.models import train

    meta = train.run(tiny_cfg)
    assert meta["production_churn_model"] in ("xgboost", "lightgbm")
    assert len(meta["churn_results"]) == 8  # 6 classical + ANN + LSTM
    sc = pd.read_parquet(tiny_cfg["paths"]["scores"])
    assert sc["churn_probability"].between(0, 1).all()
