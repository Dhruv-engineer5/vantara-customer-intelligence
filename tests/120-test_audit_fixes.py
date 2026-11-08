"""Regression tests for brief-compliance fixes: train-only inference stats, clustering metrics, LIME, timing targets."""

import json

import numpy as np
import pandas as pd

from src.features.build import CATEGORIES, FEATURE_COLUMNS, build_features
from src.utils.config import load_config, rpath


def test_api_state_uses_train_only_medians_and_cap(client):
    st = client.app.state.s
    cfg = load_config()
    table = pd.read_parquet(rpath(cfg, "features"))
    tr = table[table["data_split"] == "train"]
    assert np.allclose(st.medians.to_numpy(), tr[FEATURE_COLUMNS].median().to_numpy())
    assert st.clv_cap == float(np.log1p(tr["future_spend"].max()))


def test_segmentation_scaler_fitted_on_train_customers_only():
    import joblib

    cfg = load_config()
    table = pd.read_parquet(rpath(cfg, "features"))
    from src.segmentation.cluster import seg_matrix

    seg = joblib.load(rpath(cfg, "artifacts") / "segmentation.joblib")
    train = seg_matrix(table[table["data_split"] == "train"])
    assert np.allclose(seg["scaler"].mean_, train.mean(axis=0))
    assert not np.allclose(seg["scaler"].mean_, seg_matrix(table).mean(axis=0))


def test_clustering_metrics_elbow_and_davies_bouldin_reported():
    cfg = load_config()
    c = json.loads((rpath(cfg, "reports") / "segmentation_comparison.json").read_text())
    km = c["kmeans"]
    assert km["davies_bouldin"] > 0 and km["davies_bouldin_test"] > 0
    assert set(km["inertia_by_k"]) == set(km["silhouette_by_k"]) == set(km["davies_bouldin_by_k"])
    inertia = [km["inertia_by_k"][k] for k in sorted(km["inertia_by_k"], key=int)]
    assert inertia == sorted(inertia, reverse=True)  # inertia falls as k rises
    assert (rpath(cfg, "reports") / "kmeans_k_selection.png").exists()


def test_lime_endpoint_and_clv_total(client):
    r = client.get("/customers/12347/explain/lime").json()
    assert len(r["lime_factors"]) == 8 and r["shap_top5"]
    assert client.get("/customers/1/explain/lime").status_code == 404
    c = client.get("/customers/12347").json()
    assert abs(c["clv_total_projected"] - (c["clv_historical"] + c["clv_predicted_90d"])) < 0.02


def test_all_candidate_models_serialized():
    cfg = load_config()
    names = {p.name for p in (rpath(cfg, "artifacts") / "candidates").glob("*.joblib")}
    for m in ("logistic_regression", "decision_tree", "random_forest", "xgboost", "lightgbm", "knn"):
        assert f"churn_{m}.joblib" in names
    assert any(n.startswith("clv_") for n in names)


def test_next_purchase_timing_and_category_targets_use_only_post_cutoff_first_purchase(tx):
    cutoff = pd.Timestamp("2011-01-01")
    feats, tg = build_features(tx, cutoff, 90)
    # customer 1 buys TEA MUG 14 days after cutoff; customer 2 never buys again (censored, category -1)
    assert tg.loc[1, "days_to_next"] == 14 and tg.loc[2, "days_to_next"] == 90
    assert tg.loc[1, "next_category"] in range(len(CATEGORIES)) and tg.loc[2, "next_category"] == -1


def test_deep_models_have_five_fold_cv_in_log():
    cfg = load_config()
    log = pd.read_csv(rpath(cfg, "reports") / "experiment_log.csv")
    for m in ("ann", "lstm"):
        row = log[(log["model"] == m) & (log["split"] == "val") & (log["task"] == "churn")].iloc[0]
        assert "'cv_folds': 5" in row["params"] and "cv_mean" in row["params"]
