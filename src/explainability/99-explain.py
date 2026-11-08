"""SHAP, LIME and partial dependence wrappers plus the plain-language explanation template."""

from __future__ import annotations

from typing import Any

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from lime.lime_tabular import LimeTabularExplainer  # noqa: E402
from sklearn.inspection import PartialDependenceDisplay  # noqa: E402

from src.features.build import FEATURE_COLUMNS  # noqa: E402
from src.utils.config import load_config, rpath  # noqa: E402
from src.utils.logging_setup import get_logger  # noqa: E402

log = get_logger(__name__)

PLAIN = {
    "recency_days": "has not bought for {v:.0f} days",
    "frequency": "has placed {v:.0f} orders",
    "monetary_total": "has spent {v:,.0f} in total",
    "orders_last_90d": "placed {v:.0f} orders in the last 90 days",
    "orders_last_30d": "placed {v:.0f} orders in the last 30 days",
    "orders_last_180d": "placed {v:.0f} orders in the last 180 days",
    "spend_last_90d": "spent {v:,.0f} in the last 90 days",
    "spend_last_30d": "spent {v:,.0f} in the last 30 days",
    "spend_last_180d": "spent {v:,.0f} in the last 180 days",
    "gap_mean_days": "usually waits {v:.0f} days between orders",
    "gap_var_days": "has irregular timing between orders",
    "freq_trend": "has an order-frequency trend of {v:+.2f}/month",
    "return_rate": "returns {v:.0%} of items",
    "tenure_days": "has been buying for {v:.0f} days",
    "engagement_score": "has an engagement score of {v:.0f}/100",
    "recent_share_90d": "made {v:.0%} of their orders in the last 90 days",
    "days_since_first_purchase": "first bought {v:.0f} days ago",
    "avg_basket_size": "buys {v:.0f} items per order",
    "monetary_avg": "spends {v:,.0f} per order",
    "n_unique_products": "has bought {v:.0f} different products",
}


def _sv(explainer: Any, X: np.ndarray) -> np.ndarray:
    """SHAP values for the positive class as a (n, features) array across SHAP/LightGBM output shapes."""
    v = explainer.shap_values(X)
    if isinstance(v, list):
        v = v[1]
    v = np.asarray(v)
    if v.ndim == 3:
        v = v[:, :, 1]
    return v


class XGBContribExplainer:
    """Exact tree contributions from XGBoost itself (pred_contribs), used because shap cannot parse the
    vector-valued base_score that xgboost 3 writes. Values are in log-odds (margin) units, like TreeExplainer."""

    def __init__(self, model: Any) -> None:
        self.booster = model.get_booster()
        self.expected_value = 0.0

    def shap_values(self, X: np.ndarray) -> np.ndarray:
        """(n, features) contributions; the bias term is stored in expected_value."""
        import xgboost as xgb

        c = self.booster.predict(xgb.DMatrix(np.asarray(X, dtype=float), feature_names=None), pred_contribs=True)
        self.expected_value = float(c[0, -1])
        return c[:, :-1]


def make_explainer(model: Any) -> Any:
    """Explainer for the production gradient-boosted model (log-odds contributions)."""
    if hasattr(model, "get_booster"):
        return XGBContribExplainer(model)
    return shap.TreeExplainer(model)


def plain_language(shap_row: np.ndarray, x_row: pd.Series, proba: float, top: int = 3) -> str:
    """Translate SHAP values into a sentence a marketing analyst can read."""
    order = np.argsort(-np.abs(shap_row))[:top]
    parts = []
    for i in order:
        name = FEATURE_COLUMNS[i]
        if name.startswith("aff_"):
            default = "buys {v:.0%} of items from the " + name[4:].replace("_", " ") + " category"
        elif name.startswith("country_"):
            default = (
                "is in the " + name[8:].replace("_", " ") + " market"
                if float(x_row.iloc[i] if hasattr(x_row, "iloc") else x_row[i])
                else "is outside the " + name[8:].replace("_", " ") + " market"
            )
        else:
            default = name.replace("_", " ") + " = {v:.2f}"
        phrase = PLAIN.get(name, default).format(v=float(x_row.iloc[i]) if hasattr(x_row, "iloc") else float(x_row[i]))
        parts.append(f"{phrase} ({'raises' if shap_row[i] > 0 else 'lowers'} risk)")
    level = "high" if proba >= 0.7 else "moderate" if proba >= 0.4 else "low"
    return f"Churn risk is {level} ({proba:.0%}) mainly because this customer " + "; ".join(parts) + "."


def explain_row(model: Any, x: np.ndarray, explainer: Any = None) -> dict:
    """Return per-feature SHAP contributions and a plain-language sentence for one feature row (1D array)."""
    explainer = explainer or make_explainer(model)
    X = np.asarray(x, dtype=float).reshape(1, -1)
    sv = _sv(explainer, X)[0]
    proba = float(model.predict_proba(X)[0, 1])
    top = sorted(zip(FEATURE_COLUMNS, sv.tolist(), X[0].tolist()), key=lambda t: -abs(t[1]))[:8]
    return {
        "churn_probability": proba,
        "top_factors": [{"feature": f, "shap": s, "value": v} for f, s, v in top],
        "explanation": plain_language(sv, pd.Series(X[0]), proba),
    }


def _force_figure(path: Path, base: float, sv_row: Any, x_row: Any, top: int = 6) -> None:
    """True SHAP force plot of the top contributions; the remaining features are summed into one 'others' term.

    Plotting all 37 features makes the library's labels overlap, so they are grouped (the sum is unchanged).
    """
    order = np.argsort(-np.abs(sv_row))
    keep, rest = order[:top], order[top:]
    vals = np.append(sv_row[keep], sv_row[rest].sum())
    names = [f"{FEATURE_COLUMNS[j]} = {x_row.iloc[j]:.2f}" for j in keep] + [f"other {len(rest)} features"]
    shap.force_plot(base, vals, feature_names=names, matplotlib=True, show=False, figsize=(15, 3.6), text_rotation=20)
    ax = plt.gca()
    base_t = next((t for t in ax.texts if t.get_text() == "base value"), None)
    fx_t = next((t for t in ax.texts if t.get_text() == "f(x)"), None)
    lo, hi = ax.get_xlim()
    if (
        base_t is not None
        and fx_t is not None
        and abs(base_t.get_position()[0] - fx_t.get_position()[0]) < 0.12 * (hi - lo)
    ):
        base_t.set_text("base value = f(x)")  # captions would overlap when the prediction equals the base value
        fx_t.set_visible(False)
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close()


def _contribution_figure(
    path: Path, label: str, proba: float, base: float, sv_row: Any, x_row: Any, top: int = 8
) -> None:
    """Readable per-customer contribution chart (replaces the library force plot, whose labels overlap)."""
    order = np.argsort(-np.abs(sv_row))[:top][::-1]
    names = [f"{FEATURE_COLUMNS[j]} = {x_row.iloc[j]:.2f}" for j in order]
    vals = sv_row[order]
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    ax.barh(names, vals, color=["#d62728" if v > 0 else "#1f77b4" for v in vals])
    ax.axvline(0, color="#444", lw=0.8)
    ax.set_xlabel("Contribution to churn risk, log-odds units (right = raises risk)")
    ax.set_title(
        f"{label.replace('_', ' ')} customer: churn probability {proba:.1%} (base log-odds {base:.2f}, "
        f"top {top} of {len(sv_row)} features)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def make_lime(X_train: np.ndarray, seed: int = 42) -> LimeTabularExplainer:
    """LIME explainer whose discretisation statistics come from TRAINING customers only."""
    return LimeTabularExplainer(
        X_train,
        feature_names=FEATURE_COLUMNS,
        class_names=["retained", "churned"],
        discretize_continuous=True,
        random_state=seed,
    )


def lime_row(model: Any, lime: LimeTabularExplainer, x: np.ndarray, n: int = 8, samples: int = 1000) -> dict:
    """Local LIME explanation for one feature row, with the overlap against SHAP's top features."""
    x = np.asarray(x, dtype=float).ravel()
    exp = lime.explain_instance(x, model.predict_proba, num_features=n, num_samples=samples)
    items = [{"rule": r, "weight": float(w)} for r, w in exp.as_list()]
    return {"lime_factors": items, "local_prediction": float(exp.local_pred[0])}


def run(cfg: dict | None = None) -> dict:
    """Produce SHAP global + 3 individual plots, a LIME example vs SHAP, PDPs and examples JSON."""
    cfg = cfg or load_config()
    rep = rpath(cfg, "reports")
    model = joblib.load(rpath(cfg, "artifacts") / "churn_model.joblib")
    table = pd.read_parquet(rpath(cfg, "features"))
    scores = pd.read_parquet(rpath(cfg, "scores"))
    X = table[FEATURE_COLUMNS].to_numpy(dtype=float)
    expl = make_explainer(model)
    sv = _sv(expl, X)
    base = expl.expected_value
    base = float(base[1] if np.ndim(base) else base)
    shap.summary_plot(sv, table[FEATURE_COLUMNS], show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(rep / "shap_global_summary.png", dpi=130)
    plt.close()
    p = scores["churn_probability"].to_numpy()
    picks = {
        "low_risk": int(np.argmin(p)),
        "high_risk": int(np.argmax(p)),
        "borderline": int(np.argmin(np.abs(p - 0.5))),
    }
    examples = {}
    for label, i in picks.items():
        _force_figure(rep / f"shap_force_{label}.png", base, sv[i], table[FEATURE_COLUMNS].iloc[i])
        _contribution_figure(
            rep / f"shap_bars_{label}.png",
            label,
            float(p[i]),
            base,
            sv[i],
            table[FEATURE_COLUMNS].iloc[i],
        )
        examples[label] = {
            "customer_id": int(table.index[i]),
            "probability": float(p[i]),
            "explanation": plain_language(sv[i], table[FEATURE_COLUMNS].iloc[i], float(p[i])),
        }
    # LIME on the borderline customer, compared with SHAP's top features for the same customer
    lime = make_lime(X[(table["data_split"] == "train").to_numpy()], cfg["seed"])
    i = picks["borderline"]
    exp = lime.explain_instance(X[i], model.predict_proba, num_features=8, num_samples=2000)
    fig = exp.as_pyplot_figure()
    fig.tight_layout()
    fig.savefig(rep / "lime_borderline.png", dpi=120)
    plt.close(fig)
    lime_top = [f for f, _ in exp.as_list()][:5]
    shap_top = [FEATURE_COLUMNS[j] for j in np.argsort(-np.abs(sv[i]))[:5]]
    overlap = [f for f in shap_top if any(f in item for item in lime_top)]
    examples["lime_vs_shap"] = {
        "customer_id": int(table.index[i]),
        "lime_top": lime_top,
        "shap_top": shap_top,
        "features_in_both": overlap,
    }
    # Partial dependence for the three most influential features
    imp = np.abs(sv).mean(axis=0)
    top3 = [FEATURE_COLUMNS[j] for j in np.argsort(-imp)[:3]]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    PartialDependenceDisplay.from_estimator(model, table[FEATURE_COLUMNS], top3, ax=ax)
    fig.tight_layout()
    fig.savefig(rep / "partial_dependence.png", dpi=120)
    plt.close(fig)
    examples["top3_features"] = top3
    (rep / "explainability_examples.json").write_text(json.dumps(examples, indent=2))
    log.info("explainability done top3=%s", top3)
    return examples


if __name__ == "__main__":
    run()
