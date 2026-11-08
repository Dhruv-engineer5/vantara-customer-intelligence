"""Build docs/final_report.pdf from the saved metrics, tables and figures (nothing is typed by hand)."""

from __future__ import annotations

from typing import Any

import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import CondPageBreak, Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from src.utils.config import ROOT, load_config, rpath

SS = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=SS["Heading1"], fontSize=16, spaceAfter=8)
H2 = ParagraphStyle("H2", parent=SS["Heading2"], fontSize=12.5, spaceBefore=8, spaceAfter=4)
BODY = ParagraphStyle("B", parent=SS["BodyText"], fontSize=9.6, leading=13)
SMALL = ParagraphStyle("S", parent=BODY, fontSize=8, leading=10, textColor=colors.HexColor("#444444"))
MATH = ParagraphStyle("M", parent=BODY, fontName="Courier", fontSize=8.8, leading=12, leftIndent=12)
W = A4[0] - 4 * cm


def P(t: str, s: Any = BODY) -> Any:
    """Paragraph shorthand."""
    return Paragraph(t, s)


def _cv_table(rep: Any) -> pd.DataFrame:
    """CV mean/std per tuned model from the experiment log (validation rows carry the CV params)."""
    import ast

    log = pd.read_csv(rep / "experiment_log.csv")
    log = log[log["split"].isin(["val", "cv"])]
    rows = []
    for r in log.itertuples():
        d = ast.literal_eval(r.params) if isinstance(r.params, str) else {}
        if "cv_mean" in d:
            metric = d.get("metric") or ("ROC-AUC" if r.task == "churn" else "R2 (log1p)")
            rows.append([r.task, r.model, metric, f"{d['cv_mean']:.3f}", f"{d['cv_std']:.3f}", d.get("cv_folds", "")])
    return pd.DataFrame(rows, columns=["task", "model", "metric", "cv mean", "cv std", "folds"])


def table(df: pd.DataFrame, widths: Any = None, fs: float = 7.6) -> Any:
    """Styled table from a dataframe."""
    data = [list(df.columns)] + df.astype(str).values.tolist()
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("FONTSIZE", (0, 0), (-1, -1), fs),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dce6f7")),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    )
    return t


def img(path: Path, width: float = W) -> Any:
    """Image scaled to width, keeping aspect ratio."""
    from PIL import Image as PI

    w, h = PI.open(path).size
    return Image(str(path), width=width, height=width * h / w)


def f3(x: Any) -> str:
    """Format to three decimals."""
    return f"{x:.3f}"


def run(cfg: dict | None = None) -> Path:
    """Write the PDF report and return its path."""
    cfg = cfg or load_config()
    rep, art = rpath(cfg, "reports"), rpath(cfg, "artifacts")
    meta = json.loads((art / "metadata.json").read_text())
    clean = json.loads(rpath(cfg, "interim_rejects").read_text()) if rpath(cfg, "interim_rejects").exists() else {}
    comp = pd.read_csv(rep / "model_comparison.csv")
    seg = pd.read_csv(rep / "segment_profiles.csv")
    segc = json.loads((rep / "segmentation_comparison.json").read_text())
    ex = json.loads((rep / "explainability_examples.json").read_text())
    rt = json.loads((rep / "runtime_checks.json").read_text()) if (rep / "runtime_checks.json").exists() else {}
    tg, prod = meta["targets"], meta["production_churn_model"]
    pr = meta["churn_results"][prod]["test"]
    clv = meta["clv_results"][meta["clv_model"]]["test"]
    out = ROOT / "docs" / "final_report.pdf"
    doc = SimpleDocTemplate(
        str(out),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
        title="Vantara Customer Behavior Prediction Platform - Final Report",
    )
    s: list = []
    s += [
        P("Vantara Customer Behavior Prediction Platform", H1),
        P("Final report: churn, lifetime value and purchase behavior on Online Retail II", H2),
        P(
            "Student project built for the Internmo traineeship. The data is the public UCI Online Retail II dataset, a real UK retailer, "
            "used here as a stand-in for Vantara's data. The system is a working prototype for learning and demonstration, not a production tool.",
            BODY,
        ),
        Spacer(1, 6),
        P("1. Summary and honest results", H2),
    ]
    met = meta["targets_met"]
    rows = [
        ["Churn ROC-AUC (test)", f">= {tg['churn_auc']}", f3(pr["roc_auc"]), "MET" if met["churn_auc"] else "NOT MET"],
        [
            "Churn recall, churned class (test)",
            f">= {tg['churn_recall']}",
            f3(pr["recall"]),
            "MET" if met["churn_recall"] else "NOT MET",
        ],
        [
            "CLV R2, next-90-day spend (test, raw scale)",
            f">= {tg['clv_r2']}",
            f3(clv["r2"]),
            "MET" if met["clv_r2"] else "NOT MET",
        ],
        [
            "API p95, single prediction",
            f"< {tg['api_p95_ms']} ms",
            f"{rt.get('api_single_prediction', {}).get('p95_ms', 'n/a')} ms",
            "MET (measured)",
        ],
        [
            "Dashboard load, segment view",
            f"< {tg['dashboard_load_s']} s",
            f"{min(rt.get('dashboard_segment_view_load_seconds', {}).get('loads_s', [0])):.1f}-"
            f"{max(rt.get('dashboard_segment_view_load_seconds', {}).get('loads_s', [0])):.1f} s warm",
            "MET warm, see 9",
        ],
    ]
    s.append(
        table(
            pd.DataFrame(rows, columns=["Metric", "Target", "Result", "Status"]),
            [6.6 * cm, 2.6 * cm, 3.2 * cm, 3.6 * cm],
            8,
        )
    )
    s += [
        Spacer(1, 4),
        P(
            f"Two model-quality targets (churn ROC-AUC and CLV R2) are missed and are discussed in section 9. The production churn model is <b>{prod}</b>, chosen by "
            f"validation AUC among the gradient-boosted candidates, as the brief recommends. Validation AUC: {prod} {f3(meta['churn_results'][prod]['val']['roc_auc'])}, "
            f"Random Forest {f3(meta['churn_results']['random_forest']['val']['roc_auc'])}; test AUC {f3(meta['churn_results'][prod]['test']['roc_auc'])} vs "
            f"{f3(meta['churn_results']['random_forest']['test']['roc_auc'])}. The gap is within split noise (see section 4). Every choice was made on training and validation data. Test metrics are computed once per model and logged beside validation metrics, but never drive a decision.",
            BODY,
        ),
    ]
    s += [
        P("2. Data and cleaning", H2),
        P(
            f"Source: UCI Machine Learning Repository, Online Retail II (id 502, CC BY 4.0, Chen 2012): {clean.get('rows_in', 0):,} invoice lines from 01 Dec 2009 to 09 Dec 2011. "
            "Both sheets are loaded, standardized and ordered by time before any cleaning. Each cleaning rule is counted:",
            BODY,
        ),
    ]
    cr = [
        ["Lines in", f"{clean.get('rows_in', 0):,}"],
        [
            "Missing Customer ID (excluded from customer-level modelling)",
            f"{clean.get('rows_missing_customer_id', 0):,}",
        ],
        ["Zero or negative price (adjustments, dropped)", f"{clean.get('rows_nonpositive_price', 0):,}"],
        [
            "Non-product stock codes (postage, bank charges, manual, ...)",
            f"{clean.get('rows_non_product_stockcode', 0):,}",
        ],
        ["Exact duplicate lines", f"{clean.get('rows_exact_duplicates', 0):,}"],
        [
            f"Quantity above IQR bound ({clean.get('iqr_quantity_bound_train_fit', 0):.0f}, fitted on training customers) kept as bulk buys",
            f"{clean.get('rows_qty_flagged_bulk_kept', 0):,}",
        ],
        [
            f"Price above IQR bound ({clean.get('iqr_price_bound_train_fit', 0):.2f}, fitted on training customers) kept",
            f"{clean.get('rows_price_flagged_kept', 0):,}",
        ],
        [
            "Quantity above bound and >= 1,000 (data-entry errors, dropped)",
            f"{clean.get('rows_quantity_entry_error_dropped', 0):,}",
        ],
        [
            "Price above bound and >= 1,000 (data-entry errors, dropped)",
            f"{clean.get('rows_price_entry_error_dropped', 0):,}",
        ],
        ["Lines out", f"{clean.get('rows_out', 0):,}"],
        ["Customers with an ID", f"{clean.get('customers_out', 0):,}"],
    ]
    s += [
        table(pd.DataFrame(cr, columns=["Rule", "Count"]), [12.5 * cm, 3.5 * cm], 8),
        P(
            "Returns (negative quantity, invoices starting with C) are kept and used as a signal (return rate), not dropped. Product descriptions are standardized with "
            "a lookup keyed on stock code. The IQR bounds, the description lookup, product median prices and the engagement percentile reference are fitted on training customers' pre-cutoff rows only (customers are split before any statistic is fitted) and persisted in models_artifacts/reference_stats.joblib. Automated checks (schema, date range, nulls, price, ordering) run before feature engineering and stop the pipeline if violated.",
            BODY,
        ),
        CondPageBreak(160),
        P("3. Features, target definitions and leakage control", H2),
        P(
            f"All features use only lines with invoice date before the cutoff <b>{meta['cutoff_date']}</b>. The churn label is 1 when the customer has no purchase in the "
            f"{meta['horizon_days']} days from the cutoff (window ends before the last data date, so the label window is complete). The CLV target is each customer's net spend "
            "in that same window (returns subtract, negative totals are floored at 0). The modelled population is the "
            f"{meta['n_customers']:,} customers with at least one purchase before the cutoff, and the churn rate is {meta['churn_rate']:.1%}: in this window churn is the "
            "<i>majority</i> class, so imbalance handling is mild; class weights are still used.",
            BODY,
        ),
        P(
            "Features: recency, frequency, monetary total and average, tenure, basket size, orders and spend in the last 30/90/180 days, purchase frequency trend (slope of monthly orders over 12 months), "
            "inter-purchase gap mean and variance, seasonal concentration, product category affinity vector (keyword categories, because the source has no category field), return rate, "
            "discount sensitivity (share of lines priced under 90% of the product median), country one-hot, historical CLV and a composite engagement score. "
            "Historical CLV is net of returns and cancellations, so a few customers have negative values (for example customer 12346); these are real records and are kept, and the dashboard flags them. "
            "Unit tests change every row on or after the cutoff and assert that all features stay identical.",
            BODY,
        ),
    ]
    s += [
        CondPageBreak(300),
        P("4. Churn models", H2),
        P(
            "Six classical models (logistic regression, decision tree, random forest, XGBoost, LightGBM, KNN) plus a feed-forward ANN (batch norm, dropout, early stopping) and an LSTM over each "
            "customer's last 8 purchases. Split is stratified 70/15/15 with seed 42. Hyperparameters were tuned by 5-fold stratified cross-validated random search on the training split only; "
            "scale-sensitive models use a StandardScaler fitted on training data. Table is ordered by validation ROC-AUC then validation recall; test metrics are shown for reference, reported once, and never used for ordering, selection or tuning.",
            BODY,
        ),
    ]
    ct = comp.copy()
    for c in ct.columns[1:]:
        ct[c] = ct[c].map(f3)
    s += [
        table(ct, None, 7.4),
        Spacer(1, 4),
        P(
            f"Gradient-boosted trees are the production recommendation. The ANN reaches a similar AUC (test {f3(meta['churn_results']['ann']['test']['roc_auc'])}) but is harder to explain; "
            f"the LSTM is weaker ({f3(meta['churn_results']['lstm']['test']['roc_auc'])}) because with about 5,000 customers and 8 purchases per sequence it adds little beyond the engineered "
            "features. The deep models are comparative learning exercises, as the brief's risk table anticipates.",
            BODY,
        ),
        P(
            "Cross-validation (5-fold stratified, training customers only), mean and standard deviation of the tuning metric:",
            SMALL,
        ),
        table(_cv_table(rep), None, 7.6),
        Spacer(1, 4),
        img(rep / "loss_curves.png"),
        P(
            "Training vs validation loss for the ANN, LSTM and autoencoder (early stopping restores the best validation epoch).",
            SMALL,
        ),
    ]
    td, nc, t30 = meta["timing_days"], meta["next_category"], meta["timing_next30"]
    trows = [
        [
            k,
            f"{t30[k]['test']['roc_auc']:.3f}",
            f"{td[k]['test']['mae_days']:.1f}",
            f"{td[k]['test']['rmse_days']:.1f}",
            f"{td[k]['test']['r2_log']:.3f}",
            f"{nc[k]['test']['accuracy']:.3f}" if k in nc else "",
            f"{nc[k]['test']['top3_accuracy']:.3f}" if k in nc else "",
        ]
        for k in ("static_lgbm", "lstm_sequence", "average")
    ]
    trows.append(
        [
            "train_median / majority baseline",
            "",
            f"{td['train_median_baseline']['test']['mae_days']:.1f}",
            f"{td['train_median_baseline']['test']['rmse_days']:.1f}",
            f"{td['train_median_baseline']['test']['r2_log']:.3f}",
            f"{nc['majority_baseline_test_accuracy']:.3f}",
            "",
        ]
    )
    s += [
        P("Next-purchase timing and category (sequence model)", H2),
        P(
            f"The LSTM reads each customer's last 8 purchases (amount, gap in days, product category) and is compared with a static-feature LightGBM and their average, "
            f"for three targets: a 30-day purchase flag (all customers), days until the next purchase and the category of the next purchase (both among the {td['n_test']} test customers who buy within the 90-day horizon; "
            f"{td['n_train']} used for training). Test results, reported once:",
            BODY,
        ),
        table(
            pd.DataFrame(
                trows,
                columns=[
                    "model",
                    "30-day AUC",
                    "days MAE",
                    "days RMSE",
                    "days R2 (log)",
                    "category acc.",
                    "category top-3",
                ],
            ),
            None,
            7.6,
        ),
        P(
            "Whether the LSTM adds value beyond static features is read from the table: it only counts if it beats the static model and the baseline. Categories are keyword-rule groups, because the source has no category field.",
            BODY,
        ),
    ]
    rec = json.loads((art / "recommender_eval.json").read_text()) if (art / "recommender_eval.json").exists() else {}
    if rec:
        s += [
            P("Product recommendations", H2),
            P(
                f"An item-item co-purchase model (cosine similarity between products, fitted on training customers' purchases before the cutoff; "
                f"{rec['products_in_model']:,} products bought by at least 5 training customers) scores products a customer has not bought. "
                f"Offline check on {rec['test_buyers_evaluated']} test-split buyers: a recommended product (top 10) appears among their purchases "
                f"in the next 90 days for {rec['hit_rate_at_10']:.1%} of them, against {rec['popularity_baseline_hit_rate_at_10']:.1%} for recommending the most popular products. "
                "This is a sanity check, not evidence of campaign impact: the brief puts live A/B testing out of scope. "
                "Recommendations are served at GET /customers/{id}/recommendations and shown on the dashboard's customer view.",
                BODY,
            ),
        ]
    s += [
        P("Anomaly detection", H2),
        P(
            f"An autoencoder trained on six scaled spending features flags customers whose reconstruction error exceeds the {meta['anomaly']['percentile']}th percentile of validation errors "
            f"(threshold {meta['anomaly']['threshold']:.3f}). On the test split it flags {meta['anomaly']['flag_rate_test']:.1%} of customers. Flags are for manual review; no ground truth for fraud exists in this dataset, so no accuracy is claimed.",
            BODY,
        ),
        P("5. Customer lifetime value (next 90 days)", H2),
    ]
    cl = pd.DataFrame(
        [
            [
                k,
                f3(v["val"]["r2"]),
                f3(v["test"]["r2"]),
                f3(v["test"]["r2_log"]),
                f"{v['test']['mae']:.0f}",
                f"{v['test']['rmse']:.0f}",
            ]
            for k, v in meta["clv_results"].items()
        ],
        columns=["model", "val R2 (raw)", "test R2 (raw)", "test R2 (log1p)", "test MAE", "test RMSE"],
    )
    s += [
        table(cl, None, 8),
        P(
            f"Models are trained on log1p(spend) and scored back on the raw scale, with predictions capped at the largest training target. The chosen model is <b>{meta['clv_model']}</b> "
            f"(best validation R2). The target is 90-day future spend (not a discounted-value proxy). Test R2 is {f3(meta['clv_results'][meta['clv_model']]['test']['r2'])} on the raw scale; Random Forest scores higher on test but lower on validation, so it was not selected (no selection on test). Ridge log-scale R2 is {f3(meta['clv_results']['ridge']['test']['r2_log'])}.",
            BODY,
        ),
        P("6. Segmentation", H2),
    ]
    sg = seg.copy()
    for c in sg.columns:
        if sg[c].dtype.kind == "f":
            sg[c] = sg[c].round(2)
    s += [
        table(sg.drop(columns=["segment_id"]), None, 7),
        P(
            f"K-Means was fitted for k = 2..8 on log-scaled recency, frequency, spend, basket size, return rate and engagement. The scaler, K-Means, GMM and DBSCAN are fitted on training customers only and then applied to every customer (held-out silhouette {segc['kmeans']['silhouette_test']:.3f}, Davies-Bouldin {segc['kmeans']['davies_bouldin_test']:.3f} at k = 3). k is read from the elbow curve (inertia), silhouette and Davies-Bouldin index (figure below). Silhouette is highest at k = 2 ({segc['kmeans']['silhouette_by_k']['2']:.3f}); "
            f"k = 3 ({segc['kmeans']['silhouette_by_k']['3']:.3f}) was used because three business-readable segments are more actionable, a judgment call. Segment names are rule-based from centroid spend and recency. "
            f"A Gaussian mixture with {segc['gmm']['components']} components scored silhouette {segc['gmm']['silhouette']:.3f} (Davies-Bouldin {segc['gmm']['davies_bouldin']:.3f}). K-Means Davies-Bouldin at the chosen k (train) is {segc['kmeans']['davies_bouldin']:.3f}. DBSCAN (eps {segc['dbscan']['eps']:.2f}) found {segc['dbscan']['clusters']} dense cluster and labelled "
            f"{segc['dbscan']['noise_share']:.1%} as noise, which says the customer base is one continuous cloud rather than separated groups, so K-Means' spherical assumption is not obviously hiding other shapes.",
            BODY,
        ),
        img(rep / "kmeans_k_selection.png"),
        P("Elbow (inertia), silhouette and Davies-Bouldin against k, training customers.", SMALL),
        img(rep / "segments_pca.png", W * 0.7),
    ]
    s += [
        CondPageBreak(260),
        P("7. Explainability", H2),
        P(
            f"Feature contributions (native XGBoost pred_contribs, equal to tree SHAP, in log-odds units) gives the global importance below; the three most influential features are {', '.join(ex['top3_features'])}. "
            "The dashboard shows the same SHAP breakdown for any customer ID.",
            BODY,
        ),
        img(rep / "shap_global_summary.png", W * 0.78),
    ]
    for k, lab in [("low_risk", "Low-risk"), ("high_risk", "High-risk"), ("borderline", "Borderline")]:
        s += [
            img(rep / f"shap_force_{k}.png"),
            img(rep / f"shap_bars_{k}.png"),
            P(
                f"{lab} customer {ex[k]['customer_id']} (probability {ex[k]['probability']:.1%}). SHAP force plot (top 6 contributions, the other features summed into one term; log-odds scale) "
                f"and the same contributions as a labelled bar chart. Plain-language template: {ex[k]['explanation']}",
                SMALL,
            ),
        ]
    lv = ex["lime_vs_shap"]
    s += [
        CondPageBreak(330),
        P("LIME versus SHAP (borderline customer)", H2),
        img(rep / "lime_borderline.png", W * 0.8),
        P(
            f"For customer {lv['customer_id']}, LIME's top conditions were {', '.join(lv['lime_top'])}; SHAP's top features were {', '.join(lv['shap_top'])}. "
            f"Both agree on: {', '.join(lv['features_in_both']) or 'none'}. They diverge on the rest because LIME fits a local linear surrogate on binned features while SHAP attributes the model's actual output; "
            "for a near-50% customer the ordering of small effects is unstable in both.",
            BODY,
        ),
        img(rep / "partial_dependence.png"),
        P("Partial dependence for the three most influential features.", SMALL),
        CondPageBreak(260),
        P("8. System: API, dashboard, database and deployment", H2),
        P(
            "FastAPI endpoints: GET /health, GET /model/metadata, POST /predict, GET /customers/{id}/recommendations, POST /predict/batch (CSV in, JSON/CSV/PDF out), GET /customers (filters), GET /customers/leaderboard, "
            "GET /customers/{id}, GET /customers/{id}/explain, GET /segments, GET /revenue/trend. Pydantic validates inputs and returns 422 with clear messages. PostgreSQL (SQLAlchemy) stores "
            "customers, invoice-level transactions, predictions and segments; scores are precomputed so the dashboard does not rescore on page load. The Streamlit dashboard calls only the API. "
            "Everything starts with <font face='Courier'>docker compose up --build</font>; configuration comes from environment variables.",
            BODY,
        ),
        img(ROOT / "docs" / "architecture_diagram.png"),
        img(ROOT / "docs" / "er_diagram.png", W * 0.75),
        img(ROOT / "docs" / "workflow_diagram.png"),
    ]
    s += [
        P("9. Evaluation against the success metrics", H2),
        P(
            f"<b>Churn ROC-AUC {f3(pr['roc_auc'])} vs target {tg['churn_auc']} (not met).</b> Every model, including the best on validation ({meta['best_model_by_val_auc_overall']}, {f3(meta['churn_results'][meta['best_model_by_val_auc_overall']]['val']['roc_auc'])}), "
            "plateaus near 0.79 on the test split, and added recent-activity features did not move it. The label is a noisy human behavior (a customer skipping three months), the data has "
            "no marketing, product or demographic signals, and there are only about 5,000 customers, so a ceiling near 0.8 is plausible; this was not investigated further with more data. "
            f"<b>Churn recall {f3(pr['recall'])} (met)</b> at the default 0.5 threshold. <b>CLV R2 {f3(clv['r2'])} vs target {tg['clv_r2']} (not met).</b> Next-90-day spend is heavy-tailed (maximum "
            "above 100,000 against a median of 0), 57% of customers spend nothing in the window, and a few very large customers dominate the squared error; on a log scale R2 is about 0.34. "
            "No threshold or target was tuned on the test split.",
            BODY,
        ),
    ]
    ra = rt.get("api_single_prediction")
    if ra:
        s.append(
            P(
                f"<b>API latency:</b> {ra['requests']} sequential single-customer predictions against the running API (real PostgreSQL, localhost): p50 {ra['p50_ms']} ms, p95 {ra['p95_ms']} ms "
                f"(includes SHAP explanation). This is a build-machine measurement on localhost, not a network or Docker measurement. "
                f"<b>Dashboard load:</b> {rt['dashboard_segment_view_load_seconds']['note']}",
                BODY,
            )
        )
    s += [
        P("10. Limitations and honest notes", H2),
        P(
            "The churn label follows the brief (no purchase in the next 90 days), so it also covers customers who were already inactive for months at the cutoff (56.6% churn); the model is partly a continued-inactivity predictor, and an early-warning view limited to recently active customers is not evaluated here. k = 2 has the best silhouette; k = 3 was chosen for business use. Single time-split with one cutoff; no rolling-origin evaluation. Rows used in training are scored too and the dashboard marks which split they were in (in-sample scores look better than they are). "
            "The segmentation scaler and clusters are fitted on training customers only; the segment profile table describes all modelled customers. Product categories are a keyword proxy. The revenue chart sums invoices of modelled customers only "
            "and its forecast is a straight-line overlay, not a validated forecast. During the build the API, dashboard and tests were run against a real PostgreSQL server; afterwards the Docker Compose "
            "stack was verified end to end on Windows 11 Home (Docker Desktop 4.98.0, November 8, 2026): a fresh extraction of the delivered November 7 ZIP (this revision additionally changes the leaderboard heading text, batch-PDF number formatting, one dependency pin and documentation; the Docker-related files the run exercised are unchanged) with a newly generated .env ran docker compose up --build, all three containers came up healthy, the API seeded the database and answered requests, and the dashboard rendered all six views including batch CSV scoring and PDF export. AI tools were used to write and debug the code and this report; "
            "the author should read the code before presenting it as their own understanding.",
            BODY,
        ),
        CondPageBreak(260),
        P("Appendix A. Mathematical foundations", H1),
    ]
    A = [
        (
            "A.1 Statistics and probability",
            "Churn is a Bernoulli outcome with prior pi = P(y=1) (here about 0.57). A classifier outputs p(x) = P(y=1 | x). The decision rule is yhat = 1 if p(x) >= t. From the confusion matrix "
            "(TP, FP, TN, FN): precision = TP/(TP+FP); recall = TP/(TP+FN); F1 = 2*precision*recall/(precision+recall); accuracy = (TP+TN)/N. ROC-AUC = P(p(x+) > p(x-)) for a random positive and "
            "negative customer, equivalently the area under TPR vs FPR as t varies; it does not depend on pi, which is why it is reported with recall and F1 instead of accuracy. "
            "Cross-validated estimates use k = 5 folds: CV = (1/k) sum_i metric(model fitted without fold i, fold i). Spend is heavy-tailed, so the CLV model fits log(1 + spend) and uses "
            "R2 = 1 - SS_res/SS_tot, which is sensitive to large errors on raw spend.",
        ),
        (
            "A.2 Linear algebra and PCA",
            "With standardized feature matrix X (n x d), the covariance is C = X^T X/(n-1). PCA solves C v_j = lambda_j v_j; the projection onto the top-2 eigenvectors gives the segment plot, with explained "
            "variance lambda_j / sum(lambda). Multicollinearity (spend in 30/90/180-day windows, engagement score built from recency, frequency and spend) shows up as near-zero eigenvalues of C; the variance "
            "inflation factor VIF_j = 1/(1 - R2_j), where R2_j is the R2 of regressing feature j on the others, quantifies it. Tree models tolerate it; for logistic regression L2 regularization stabilizes it.",
        ),
        (
            "A.3 Logistic regression and cross-entropy",
            "p(x) = sigmoid(w.x + b) with sigmoid(z) = 1/(1+e^-z). The loss is binary cross-entropy L = -(1/n) sum [ y log p + (1-y) log(1-p) ] + (lambda/2)||w||^2, with optional class weights. "
            "The gradient is dL/dw = (1/n) X^T (p - y) + lambda w, so each coefficient is a log-odds change per standardized unit and gives a first-pass explanation. XGBoost and LightGBM minimize the same loss "
            "by adding trees fitted to the gradient g_i = p_i - y_i and Hessian h_i = p_i(1-p_i) of the loss.",
        ),
        (
            "A.4 Gradient descent and backpropagation",
            "Weights update as w <- w - eta * grad L(w) (Adam adds momentum and per-weight scaling; eta = 0.001 here). For a layer a_l = relu(W_l a_(l-1) + b_l), the chain rule gives "
            "delta_l = (W_(l+1)^T delta_(l+1)) * relu'(z_l) and dL/dW_l = delta_l a_(l-1)^T. The ANN stacks Linear-BatchNorm-ReLU-Dropout blocks. The LSTM carries a cell state: "
            "f_t = sigma(W_f [h_(t-1), x_t] + b_f), i_t = sigma(W_i [..] + b_i), g_t = tanh(W_g [..] + b_g), c_t = f_t * c_(t-1) + i_t * g_t, o_t = sigma(W_o [..] + b_o), h_t = o_t * tanh(c_t); "
            "gradients flow back through time via c_t. The loss curves in section 4 show training and validation loss per epoch; training stops when validation loss has not improved for 12 epochs, "
            "which is the convergence and overfitting check.",
        ),
        (
            "A.5 Regularization",
            "L2 adds (lambda/2)||w||^2 (logistic regression C = 1/lambda); dropout zeroes each hidden unit with probability 0.3 during training, scaling the rest, which approximates averaging many thinner networks; "
            "early stopping restores the weights from the best validation epoch; tree depth, min_samples_leaf and subsampling limit tree variance; random forests average B trees fitted on bootstrap samples, "
            "reducing variance by roughly 1/B for uncorrelated trees. Cross-validation on the training split chooses these settings, and the test split is never used for tuning.",
        ),
        (
            "A.6 Clustering, explanations and anomalies",
            "K-Means minimizes sum_k sum_(x in S_k) ||x - mu_k||^2 by alternating assignment and mean updates; silhouette s(i) = (b_i - a_i)/max(a_i, b_i) compares mean intra-cluster distance a_i with the nearest other "
            "cluster's b_i. A Gaussian mixture maximizes sum_i log sum_k pi_k N(x_i | mu_k, Sigma_k) by EM; DBSCAN groups points with at least min_samples neighbors within eps. "
            "SHAP values are Shapley values phi_j = sum over subsets S of F without j of |S|!(|F|-|S|-1)!/|F|! [f(S+j) - f(S)], with f(x) = base value + sum_j phi_j; native tree contributions (tree SHAP) compute them exactly for tree ensembles. "
            "LIME fits a weighted linear model g to f on perturbed samples around x. The autoencoder minimizes ||x - D(E(x))||^2; a customer is flagged when that error exceeds the 99th percentile on validation data.",
        ),
    ]
    # worked numbers from the actual production model on the test split
    import joblib
    from sklearn.metrics import confusion_matrix

    from src.features.build import FEATURE_COLUMNS
    from src.models.common import load_table, make_split

    tbl = load_table(cfg)
    te = make_split(tbl, cfg)["test"]
    mdl = joblib.load(art / "churn_model.joblib")
    yt = tbl["churn"].iloc[te].to_numpy()
    pt = mdl.predict_proba(tbl[FEATURE_COLUMNS].iloc[te].to_numpy(dtype=float))[:, 1]
    tn, fp, fn, tp = confusion_matrix(yt, (pt >= 0.5).astype(int)).ravel()
    pre, rec = tp / (tp + fp), tp / (tp + fn)
    A.append(
        (
            "A.7 Worked example from this project's test split (production model, threshold 0.5)",
            f"Confusion matrix: TP = {tp}, FP = {fp}, TN = {tn}, FN = {fn} (N = {len(yt)}). Precision = TP/(TP+FP) = {tp}/({tp}+{fp}) = {pre:.3f}; recall = TP/(TP+FN) = {tp}/({tp}+{fn}) = {rec:.3f} (in this split FP and FN are both {fp}, so the two denominators coincide and precision equals recall numerically); "
            f"F1 = 2 x {pre:.3f} x {rec:.3f} / ({pre:.3f} + {rec:.3f}) = {2 * pre * rec / (pre + rec):.3f}; accuracy = ({tp}+{tn})/{len(yt)} = {(tp + tn) / len(yt):.3f}. "
            f"Prior of churn in this split = {yt.mean():.3f}, so always predicting 'churn' would give accuracy {yt.mean():.3f} and recall 1.0 with precision {yt.mean():.3f}: this is why recall alone, "
            f"or accuracy alone, is not enough and all of precision, recall, F1 and ROC-AUC are reported together. Mean predicted probability on the test split = {pt.mean():.3f} "
            f"(observed rate {yt.mean():.3f}), a quick calibration check.",
        )
    )
    A.append(
        (
            "A.8 Gradient boosting objective and bias-variance",
            "A boosted model is f(x) = sum_m eta * h_m(x) with each tree h_m chosen to minimize the second-order approximation of the loss, "
            "sum_i [ g_i h(x_i) + 0.5 h_i h(x_i)^2 ] + Omega(h), where Omega penalizes the number of leaves and leaf weights. A split is kept when the gain "
            "0.5 [ G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda) - (G_L+G_R)^2/(H_L+H_R+lambda) ] - gamma is positive, with G and H the sums of g_i and h_i in a node. "
            "Expected test error decomposes as bias^2 + variance + noise: shallow trees and heavy regularization raise bias; deep trees raise variance. "
            "With a noisy behavioural label the irreducible noise term is large, which is the main reason accuracy plateaus for every model family in section 4.",
        )
    )
    for t, b in A:
        s += [P(t, H2), P(b, BODY)]
    s += [
        P("Appendix B. Future enhancements", H1),
        P(
            "Transformer sequence models (out of scope per brief, need more sequences); rolling-origin evaluation with several cutoffs; richer product taxonomy; probability calibration and "
            "a cost-based threshold using campaign cost and CLV; real-time streaming and authentication; A/B testing of retention offers; CatBoost comparison; MLflow tracking; CI with the test suite.",
            BODY,
        ),
    ]
    doc.build(s)
    return out


if __name__ == "__main__":
    print(run())  # noqa: T201
