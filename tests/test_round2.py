"""Regression tests for the second compliance pass: null-rate checks, StockCode encoding, recommender, secrets."""

import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.data.validate import ValidationError, check_null_rates
from src.features.build import FEATURE_COLUMNS, build_features
from src.recommend.recommender import fit, recommend_for
from src.utils.config import ROOT, load_config, rpath


def test_null_rate_threshold_enforced():
    df = pd.DataFrame({"a": [1, None, 3, 4], "b": [1, 2, 3, 4]})
    assert check_null_rates(df, {"a": 0.5, "b": 0.0}) == {"a": 0.25, "b": 0.0}
    with pytest.raises(ValidationError, match="a"):
        check_null_rates(df, {"a": 0.1})


def test_config_thresholds_cover_core_columns():
    lim = load_config()["data"]["max_null_rate_raw"]
    assert lim["quantity"] == 0.0 and lim["price"] == 0.0 and 0 < lim["customer_id"] < 1


def test_stockcode_frequency_encoding_uses_reference_and_unseen_is_zero(tx):
    cutoff = "2011-01-01"
    ref = {"stock_freq": {"S1": 3.0}}
    feats, _ = build_features(tx, cutoff, 90, ref)
    assert {"product_popularity_mean", "niche_product_share"} <= set(feats.columns)
    sales = tx[(tx["invoice_date"] < cutoff) & ~tx["is_return"]]
    expect = sales.assign(p=sales["stock_code"].map(ref["stock_freq"]).fillna(0.0)).groupby("customer_id")["p"].mean()
    assert np.allclose(feats["product_popularity_mean"].reindex(expect.index), expect)


def test_reference_artifact_stores_train_stockcode_frequencies():
    ref = joblib.load(rpath(load_config(), "reference_stats"))
    assert len(ref["stock_freq"]) > 1000 and min(ref["stock_freq"].values()) > 0


def test_recommender_scores_unseen_products_and_excludes_bought(monkeypatch):
    rows = [
        (c, p) for c, ps in {1: "ABC", 2: "ABC", 3: "AB", 4: "ABD", 5: "ABD", 6: "ABC", 7: "AB"}.items() for p in ps
    ]
    sales = pd.DataFrame(rows, columns=["customer_id", "stock_code"])
    import src.recommend.recommender as r

    monkeypatch.setattr(r, "MIN_CUSTOMERS_PER_PRODUCT", 2)
    model = fit(sales)
    rec = recommend_for(model, ["A", "B"], 2)
    assert [c for c, _ in rec][0] in {"C", "D"} and all(c not in ("A", "B") for c, _ in rec)
    cold = recommend_for(model, [], 2)
    assert len(cold) == 2 and cold[0][1] == 0.0


def test_recommendation_endpoint(client):
    r = client.get("/customers/12347/recommendations", params={"n": 5})
    assert r.status_code == 200
    items = r.json()["recommendations"]
    assert len(items) == 5 and all(i["stock_code"] and "score" in i for i in items)
    assert client.get("/customers/1/recommendations").status_code == 404


def test_purchase_probability_is_complement_of_churn(client):
    c = client.get("/customers/12347").json()
    assert abs(c["purchase_probability_90d"] - (1 - c["churn_probability"])) < 1e-3
    p = client.post("/predict", json={"recency_days": 20, "frequency": 5, "monetary_total": 400}).json()
    assert abs(p["purchase_probability_90d"] - (1 - p["churn_probability"])) < 1e-6


def test_recommender_eval_is_recorded_and_beats_nothing_silently():
    ev = json.loads((rpath(load_config(), "artifacts") / "recommender_eval.json").read_text())
    assert ev["test_buyers_evaluated"] > 100 and 0 <= ev["hit_rate_at_10"] <= 1
    assert "A/B" in ev["note"]


def test_gmm_and_dbscan_serialized():
    seg = joblib.load(rpath(load_config(), "artifacts") / "segmentation.joblib")
    assert {"kmeans", "gmm", "dbscan", "scaler"} <= set(seg)


def test_feature_dictionary_documents_every_model_column():
    text = (ROOT / "docs" / "FEATURES.md").read_text()
    assert [c for c in FEATURE_COLUMNS if f"`{c}`" not in text] == []


def test_no_committed_default_password():
    for name in ("docker-compose.yml", "config/config.yaml", ".env.example"):
        text = (ROOT / name).read_text()
        assert "vantara:vantara" not in text and "PASSWORD:-" not in text
    assert "?run python scripts/setup_env.py" in (ROOT / "docker-compose.yml").read_text()


def test_setup_env_generates_random_nondefault_password(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("setup_env", ROOT / "scripts" / "setup_env.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    (tmp_path / ".env.example").write_text((ROOT / ".env.example").read_text())
    monkeypatch.setattr(mod, "ROOT", Path(tmp_path))
    assert mod.main() == 0
    pw = dict(line.split("=", 1) for line in (tmp_path / ".env").read_text().splitlines() if "=" in line)[
        "POSTGRES_PASSWORD"
    ]
    assert len(pw) >= 24 and "change-me" not in pw
    before = (tmp_path / ".env").read_text()
    mod.main()
    assert (tmp_path / ".env").read_text() == before


def test_model_artifacts_dir_env_moves_artifact_paths(monkeypatch):
    from src.utils.config import load_config, rpath

    cfg = load_config()
    monkeypatch.delenv("MODEL_ARTIFACTS_DIR", raising=False)
    assert rpath(cfg, "artifacts").name == "models_artifacts"
    monkeypatch.setenv("MODEL_ARTIFACTS_DIR", "elsewhere/arts")
    assert rpath(cfg, "artifacts").as_posix().endswith("elsewhere/arts")
    assert rpath(cfg, "reference_stats").as_posix().endswith("elsewhere/arts/reference_stats.joblib")
    assert rpath(cfg, "reports").name == "reports"


def test_thresholds_come_from_config():
    from api.deps import risk_level
    from src.features.build import COUNTRY_GROUPS, RECENT_WINDOWS
    from src.utils.config import load_config

    cfg = load_config()
    assert list(RECENT_WINDOWS) == cfg["features"]["recent_windows_days"]
    assert COUNTRY_GROUPS == cfg["features"]["country_groups"]
    assert risk_level(cfg["risk"]["high_churn"]) == "high"
    assert risk_level(cfg["risk"]["moderate_churn"]) == "moderate"
    assert risk_level(0.0) == "low"
