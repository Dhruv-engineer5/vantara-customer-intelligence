"""Score refresh: new artifacts change customer scores, keep old prediction rows, detect stale scores."""

import copy
import json
import shutil

import pandas as pd
import pytest
from sqlalchemy import create_engine, func, select, text
from sqlalchemy.orm import Session

from src.db.models import Base, Customer, Prediction, PredictionHistory, ScoreRun
from src.db.session import artifact_hash, current_score_run, refresh_scores, seed_if_empty
from src.utils.config import load_config, rpath


@pytest.fixture()
def iso_engine(pg_engine):
    """Engine pinned to its own schema so the shared API test tables are untouched."""
    with pg_engine.begin() as c:
        c.execute(text("DROP SCHEMA IF EXISTS refresh_test CASCADE"))
        c.execute(text("CREATE SCHEMA refresh_test"))
    eng = create_engine(pg_engine.url, connect_args={"options": "-csearch_path=refresh_test"})
    yield eng
    Base.metadata.drop_all(eng)
    eng.dispose()
    with pg_engine.begin() as c:
        c.execute(text("DROP SCHEMA IF EXISTS refresh_test CASCADE"))


def _cfg_copy(tmp_path):
    cfg = copy.deepcopy(load_config())
    art = tmp_path / "art"
    art.mkdir()
    shutil.copy(rpath(cfg, "artifacts") / "metadata.json", art / "metadata.json")
    cfg["paths"]["artifacts"] = str(art)
    for key in ("scores", "segments", "features"):
        dst = tmp_path / f"{key}.parquet"
        shutil.copy(rpath(cfg, key), dst)
        cfg["paths"][key] = str(dst)
    return cfg


def test_refresh_changes_scores_and_preserves_history(iso_engine, tmp_path):
    cfg = _cfg_copy(tmp_path)
    assert seed_if_empty(iso_engine, cfg) is True
    assert seed_if_empty(iso_engine, cfg) is False  # unchanged artifacts: nothing happens
    first = current_score_run(iso_engine)
    assert first.artifact_hash == artifact_hash(cfg)
    with Session(iso_engine) as s:
        n_pred = s.scalar(select(func.count()).select_from(Prediction))
        n_cust = s.scalar(select(func.count()).select_from(Customer))
        old_p = s.get(Prediction, 12347).churn_probability

    # simulate retraining: different scores in the artifacts
    sc = pd.read_parquet(cfg["paths"]["scores"])
    sc["churn_probability"] = (sc["churn_probability"] * 0.5).clip(0, 1)
    sc.to_parquet(cfg["paths"]["scores"])
    assert artifact_hash(cfg) != first.artifact_hash  # stale scores are detectable

    assert seed_if_empty(iso_engine, cfg) is True
    second = current_score_run(iso_engine)
    assert second.run_id != first.run_id and second.artifact_hash == artifact_hash(cfg)
    with Session(iso_engine) as s:
        assert s.scalar(select(func.count()).select_from(Customer)) == n_cust  # no customers lost
        assert s.scalar(select(func.count()).select_from(Prediction)) == n_pred
        assert s.get(Prediction, 12347).churn_probability == pytest.approx(old_p * 0.5, rel=1e-6)
        hist = s.get(PredictionHistory, (12347, first.run_id))
        assert hist is not None and hist.churn_probability == pytest.approx(old_p)  # old score preserved
        assert s.scalar(select(func.count()).select_from(PredictionHistory)) == n_pred
        assert s.scalar(select(func.count()).select_from(ScoreRun)) == 2


def test_database_without_score_run_is_archived_not_dropped(iso_engine, tmp_path):
    cfg = _cfg_copy(tmp_path)
    seed_if_empty(iso_engine, cfg)
    with iso_engine.begin() as c:
        c.execute(text("DELETE FROM score_runs"))  # a database created before score runs existed
    assert refresh_scores(iso_engine, cfg) is True
    with Session(iso_engine) as s:
        assert s.scalar(select(func.count()).select_from(PredictionHistory)) > 0
        assert json.loads(open(cfg["paths"]["artifacts"] + "/metadata.json").read())["n_customers"] > 0
