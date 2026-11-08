"""Classical ML churn models and CLV regressors with CV tuning on the training set only."""

from __future__ import annotations

from typing import Any

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, XGBRegressor

from src.models.common import Timer


def churn_candidates(cfg: dict) -> dict[str, tuple]:
    """name -> (estimator, param_grid). Scale-sensitive models are wrapped with StandardScaler."""
    s, m = cfg["seed"], cfg["models"]
    pre = lambda est: Pipeline([("scale", StandardScaler()), ("clf", est)])  # noqa: E731
    return {
        "logistic_regression": (
            pre(LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=s)),
            {"clf__C": m["logreg"]["C"], "clf__penalty": m["logreg"]["penalty"]},
        ),
        "decision_tree": (
            DecisionTreeClassifier(class_weight="balanced", random_state=s),
            {"max_depth": m["decision_tree"]["max_depth"], "min_samples_leaf": m["decision_tree"]["min_samples_leaf"]},
        ),
        "random_forest": (
            RandomForestClassifier(class_weight="balanced", random_state=s, n_jobs=2),
            {k: m["random_forest"][k] for k in ("n_estimators", "max_depth", "min_samples_leaf")},
        ),
        "xgboost": (
            XGBClassifier(
                n_estimators=300, base_score=0.5, eval_metric="auc", random_state=s, n_jobs=2, tree_method="hist"
            ),
            dict(m["xgboost"]),
        ),
        "lightgbm": (
            LGBMClassifier(
                n_estimators=300, class_weight="balanced", random_state=s, n_jobs=2, verbose=-1, subsample_freq=1
            ),
            dict(m["lightgbm"]),
        ),
        "knn": (pre(KNeighborsClassifier()), {"clf__n_neighbors": m["knn"]["n_neighbors"]}),
    }


def tune(name: str, est: Any, grid: dict, X: Any, y: Any, cfg: dict, scoring: str = "roc_auc") -> tuple[Any, dict]:
    """Randomised CV search on the training split; returns (best_estimator, params, seconds)."""
    folds, seed = cfg["models"]["cv_folds"], cfg["seed"]
    if scoring == "roc_auc":
        cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    else:  # regression: stratify folds on spend bins (zero spend is its own bin) so every fold sees the whole range
        yy = np.asarray(y, dtype=float)
        bins = np.zeros(len(yy), dtype=int)
        pos = yy > 0
        if pos.sum() >= folds * 4:
            edges = np.quantile(yy[pos], [0.25, 0.5, 0.75])
            bins[pos] = 1 + np.searchsorted(edges, yy[pos])
        cv = list(StratifiedKFold(folds, shuffle=True, random_state=seed).split(np.zeros(len(yy)), bins))
    n_iter = min(cfg["models"]["n_iter_search"], int(np.prod([len(v) for v in grid.values()])))
    with Timer() as t:
        if name == "xgboost":  # class imbalance handled with scale_pos_weight
            est.set_params(scale_pos_weight=float((y == 0).sum() / max((y == 1).sum(), 1)))
        search = RandomizedSearchCV(
            est, grid, n_iter=n_iter, scoring=scoring, cv=cv, random_state=cfg["seed"], n_jobs=1
        )
        search.fit(X, y)
        best = search.best_estimator_
        if name == "xgboost":  # early stopping on a validation fold carved out of the TRAINING split only
            from sklearn.model_selection import train_test_split

            Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.15, stratify=y, random_state=cfg["seed"])
            probe = clone(best).set_params(n_estimators=600, early_stopping_rounds=30)
            probe.fit(Xa, ya, eval_set=[(Xb, yb)], verbose=False)
            best = clone(best).set_params(n_estimators=int(probe.best_iteration) + 1)
            best.fit(X, y)
            search.best_params_["n_estimators_early_stopped"] = int(probe.best_iteration) + 1
    i = int(search.best_index_)
    search.best_params_["cv_mean"] = round(float(search.cv_results_["mean_test_score"][i]), 4)
    search.best_params_["cv_std"] = round(float(search.cv_results_["std_test_score"][i]), 4)
    search.best_params_["cv_folds"] = folds
    return best, search.best_params_, t.seconds


def clv_candidates(cfg: dict) -> dict[str, tuple]:
    """CLV regression candidates (target is log1p of next-window spend)."""
    s = cfg["seed"]
    return {
        "ridge": (Pipeline([("scale", StandardScaler()), ("reg", Ridge())]), {"reg__alpha": [0.1, 1.0, 10.0, 100.0]}),
        "random_forest_reg": (
            RandomForestRegressor(random_state=s, n_jobs=2),
            {"n_estimators": [200, 400], "max_depth": [6, 10], "min_samples_leaf": [3, 8]},
        ),
        "xgboost_reg": (
            XGBRegressor(n_estimators=300, base_score=0.5, random_state=s, n_jobs=2, tree_method="hist"),
            {
                "learning_rate": [0.03, 0.05, 0.1],
                "max_depth": [3, 4, 6],
                "subsample": [0.7, 0.9],
                "colsample_bytree": [0.7, 0.9],
            },
        ),
    }
