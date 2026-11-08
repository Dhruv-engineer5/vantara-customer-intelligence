"""Train and compare all churn models, the CLV regressor and the deep/anomaly models."""

from __future__ import annotations

from collections.abc import Callable

from typing import Any

import json

import joblib
from sklearn.metrics import roc_auc_score
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.features.build import FEATURE_COLUMNS  # noqa: E402
from src.models import classical, deep  # noqa: E402
from src.models.common import RunLog, clf_metrics, load_table, make_split, reg_metrics, xy  # noqa: E402
from src.utils.config import load_config, rpath  # noqa: E402
from src.utils.logging_setup import get_logger  # noqa: E402

log = get_logger(__name__)


def _cv_deep(make: Callable[[], Any], X: Any, y: Any, cfg: dict, scale: bool, pw: float) -> dict:
    """5-fold stratified CV on TRAINING rows for a neural model (fixed architecture, no test data).

    Each fold: the scaler (if any) is fitted on the fold's training part, 15% of that part is held out for early
    stopping, and ROC-AUC is measured on the untouched fold.
    """
    from sklearn.model_selection import StratifiedKFold, train_test_split

    aucs = []
    for k, (a, b) in enumerate(
        StratifiedKFold(cfg["models"]["cv_folds"], shuffle=True, random_state=cfg["seed"]).split(X, y)
    ):
        fit_i, es_i = train_test_split(a, test_size=0.15, stratify=y[a], random_state=cfg["seed"] + k)
        Xf, Xe, Xh = X[fit_i], X[es_i], X[b]
        if scale:
            sc = StandardScaler().fit(Xf)
            Xf, Xe, Xh = sc.transform(Xf), sc.transform(Xe), sc.transform(Xh)
        m, _, _ = deep.fit_binary(make(), Xf, y[fit_i], Xe, y[es_i], cfg, pw)
        aucs.append(float(roc_auc_score(y[b], deep.predict_proba(m, Xh))))
    return {"cv_mean": round(float(np.mean(aucs)), 4), "cv_std": round(float(np.std(aucs)), 4), "cv_folds": len(aucs)}


def _cv_extra_models(cfg: dict, data: dict, opt: dict) -> dict:
    """5-fold CV (training rows only) for the static LightGBM and LSTM timing/category models.

    Metrics: ROC-AUC for the 30-day purchase flag, MAE in days for days-to-next-purchase, accuracy for the next
    category. Each fold: LightGBM is fitted on the fold's training part; the LSTM too, with 15% of that part held out
    for early stopping.
    """
    from lightgbm import LGBMClassifier, LGBMRegressor
    from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

    seed, folds = cfg["seed"], cfg["models"]["cv_folds"]
    out: dict = {}
    for task, (X, S, y, kind) in data.items():
        scores: dict[str, list[float]] = {"static_lgbm": [], "lstm_sequence": []}
        stratify = kind != "reg" and np.bincount(y.astype(int)).min() >= folds
        splitter = (StratifiedKFold if stratify else KFold)(folds, shuffle=True, random_state=seed)
        for k, (a, b) in enumerate(splitter.split(X, y if stratify else None)):
            fit_i, es_i = train_test_split(a, test_size=0.15, random_state=seed + k)
            if kind == "binary":
                st = LGBMClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    num_leaves=15,
                    class_weight="balanced",
                    random_state=seed,
                    n_jobs=2,
                    verbose=-1,
                ).fit(X[a], y[a])
                net = deep.LSTMClassifier(S.shape[2], opt["lstm_hidden"], opt["dropout"])
                m, _, _ = deep.fit_binary(net, S[fit_i], y[fit_i], S[es_i], y[es_i], cfg, opt["pw30"])
                scores["static_lgbm"].append(float(roc_auc_score(y[b], st.predict_proba(X[b])[:, 1])))
                scores["lstm_sequence"].append(float(roc_auc_score(y[b], deep.predict_proba(m, S[b]))))
            elif kind == "reg":
                st = LGBMRegressor(
                    n_estimators=200, learning_rate=0.05, num_leaves=15, random_state=seed, n_jobs=2, verbose=-1
                ).fit(X[a], y[a])
                net = deep.LSTMClassifier(S.shape[2], opt["lstm_hidden"], opt["dropout"])
                m, _, _ = deep.fit_binary(net, S[fit_i], y[fit_i], S[es_i], y[es_i], cfg, task="reg")

                def mae(p: Any, idx: Any = b, tgt: Any = y) -> float:
                    return float(np.mean(np.abs(np.expm1(tgt[idx]) - np.clip(np.expm1(p), 0, opt["clip"]))))

                scores["static_lgbm"].append(mae(st.predict(X[b])))
                scores["lstm_sequence"].append(mae(deep.predict_raw(m, S[b])))
            else:
                st = LGBMClassifier(
                    n_estimators=200, learning_rate=0.05, num_leaves=15, random_state=seed, n_jobs=2, verbose=-1
                ).fit(X[a], y[a])
                net = deep.LSTMClassifier(S.shape[2], opt["lstm_hidden"], opt["dropout"], n_out=opt["n_cat"])
                m, _, _ = deep.fit_binary(net, S[fit_i], y[fit_i], S[es_i], y[es_i], cfg, task="multi")
                scores["static_lgbm"].append(float(np.mean(st.classes_[st.predict_proba(X[b]).argmax(1)] == y[b])))
                scores["lstm_sequence"].append(float(np.mean(deep.predict_raw(m, S[b]).argmax(1) == y[b])))
        metric = {"binary": "ROC-AUC", "reg": "MAE days", "multi": "accuracy"}[kind]
        out[task] = {
            nm: {
                "metric": metric,
                "cv_mean": round(float(np.mean(v)), 4),
                "cv_std": round(float(np.std(v)), 4),
                "cv_folds": len(v),
            }
            for nm, v in scores.items()
        }
    return out


def run(cfg: dict | None = None) -> dict:
    """Full training run. Returns the metadata dict that is also saved to models_artifacts/metadata.json."""
    cfg = cfg or load_config()
    seed = cfg["seed"]
    deep.set_seed(seed)
    art = rpath(cfg, "artifacts")
    rep = rpath(cfg, "reports")
    art.mkdir(parents=True, exist_ok=True)
    rep.mkdir(parents=True, exist_ok=True)
    table = load_table(cfg)
    split = make_split(table, cfg)
    Xtr, ytr = xy(table, split["train"])
    Xva, yva = xy(table, split["val"])
    Xte, yte = xy(table, split["test"])
    runlog = RunLog(rep / "experiment_log.csv")
    scaler = StandardScaler().fit(Xtr)
    joblib.dump(scaler, art / "scaler.joblib")
    meta: dict = {
        "seed": seed,
        "n_customers": len(table),
        "split_sizes": {k: len(v) for k, v in split.items()},
        "churn_rate": float(table["churn"].mean()),
        "feature_columns": FEATURE_COLUMNS,
        "cutoff_date": cfg["features"]["cutoff_date"],
        "horizon_days": cfg["features"]["churn_horizon_days"],
    }

    # ---- classical churn models ----
    fitted: dict[str, object] = {}
    results: dict[str, dict] = {}
    for name, (est, grid) in classical.churn_candidates(cfg).items():
        best, params, secs = classical.tune(name, est, grid, Xtr, ytr, cfg)
        fitted[name] = best
        mv = clf_metrics(yva, best.predict_proba(Xva)[:, 1])
        mt = clf_metrics(yte, best.predict_proba(Xte)[:, 1])
        runlog.add("churn", name, "val", mv, secs, params)
        runlog.add("churn", name, "test", mt, secs, params)
        results[name] = {"val": mv, "test": mt}
        log.info("churn model=%s val_auc=%.4f test_auc=%.4f secs=%.1f", name, mv["roc_auc"], mt["roc_auc"], secs)

    (art / "candidates").mkdir(exist_ok=True)
    for _n, _m in fitted.items():  # every classical candidate is serialized, not only the production model
        joblib.dump(_m, art / "candidates" / f"churn_{_n}.joblib")

    # ---- deep churn models ----
    d = cfg["models"]["deep"]
    Xtr_s, Xva_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xva), scaler.transform(Xte)
    pw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    ann, hist_ann, secs = deep.fit_binary(
        deep.ANN(Xtr.shape[1], d["hidden"], d["dropout"]), Xtr_s, ytr, Xva_s, yva, cfg, pw
    )
    mv, mt = clf_metrics(yva, deep.predict_proba(ann, Xva_s)), clf_metrics(yte, deep.predict_proba(ann, Xte_s))
    ann_cv = _cv_deep(lambda: deep.ANN(Xtr.shape[1], d["hidden"], d["dropout"]), Xtr, ytr, cfg, True, pw)
    runlog.add("churn", "ann", "val", mv, secs, {"hidden": d["hidden"], **ann_cv})
    runlog.add("churn", "ann", "test", mt, secs, {"hidden": d["hidden"], **ann_cv})
    results["ann"] = {"val": mv, "test": mt}
    import torch

    torch.save(ann.state_dict(), art / "ann.pt")

    seq_npz = np.load(rpath(cfg, "sequences"))
    seq = seq_npz["seq"]
    S_tr, S_va, S_te = seq[split["train"]], seq[split["val"]], seq[split["test"]]
    lstm, hist_lstm, secs = deep.fit_binary(
        deep.LSTMClassifier(seq.shape[2], d["lstm_hidden"], d["dropout"]), S_tr, ytr, S_va, yva, cfg, pw
    )
    mv, mt = clf_metrics(yva, deep.predict_proba(lstm, S_va)), clf_metrics(yte, deep.predict_proba(lstm, S_te))
    lstm_cv = _cv_deep(
        lambda: deep.LSTMClassifier(seq.shape[2], d["lstm_hidden"], d["dropout"]), S_tr, ytr, cfg, False, pw
    )
    runlog.add("churn", "lstm", "val", mv, secs, {"hidden": d["lstm_hidden"], "seq_len": seq.shape[1], **lstm_cv})
    runlog.add("churn", "lstm", "test", mt, secs, {"hidden": d["lstm_hidden"], "seq_len": seq.shape[1], **lstm_cv})
    results["lstm"] = {"val": mv, "test": mt}
    torch.save(lstm.state_dict(), art / "lstm.pt")

    # next-purchase timing (brief Sec 11: LSTM should improve timing prediction beyond static features):
    # target = any purchase within 30 days after the cutoff. Compare static-feature LightGBM, LSTM on sequences
    # (amount, gap, category) and their average. Everything fitted on train only; test reported once.
    from lightgbm import LGBMClassifier

    t30 = table["next30"].to_numpy()
    y30tr, y30va, y30te = t30[split["train"]], t30[split["val"]], t30[split["test"]]
    pw30 = float((y30tr == 0).sum() / max((y30tr == 1).sum(), 1))
    base = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=15,
        class_weight="balanced",
        random_state=cfg["seed"],
        n_jobs=2,
        verbose=-1,
    )
    base.fit(Xtr, y30tr)
    lstm30, hist30, secs30 = deep.fit_binary(
        deep.LSTMClassifier(seq.shape[2], d["lstm_hidden"], d["dropout"]), S_tr, y30tr, S_va, y30va, cfg, pw30
    )
    timing: dict = {"positive_rate_train": float(y30tr.mean())}
    pv = {"static_lgbm": base.predict_proba(Xva)[:, 1], "lstm_sequence": deep.predict_proba(lstm30, S_va)}
    pt = {"static_lgbm": base.predict_proba(Xte)[:, 1], "lstm_sequence": deep.predict_proba(lstm30, S_te)}
    pv["average"], pt["average"] = (pv["static_lgbm"] + pv["lstm_sequence"]) / 2, (
        pt["static_lgbm"] + pt["lstm_sequence"]
    ) / 2
    for k in pv:
        mv30, mt30 = clf_metrics(y30va, pv[k]), clf_metrics(y30te, pt[k])
        runlog.add("timing30", k, "val", mv30, secs30, {})
        runlog.add("timing30", k, "test", mt30, secs30, {})
        timing[k] = {"val": mv30, "test": mt30}
    meta["timing_next30"] = timing
    torch.save(lstm30.state_dict(), art / "lstm_next30.pt")

    # next-purchase DAYS and CATEGORY among customers who buy within the horizon (brief: amount, gap, category sequences
    # for next-purchase timing; target variable "Next Purchase Category"). Baselines: train median / majority class.
    from lightgbm import LGBMClassifier as _LC
    from lightgbm import LGBMRegressor
    from sklearn.metrics import accuracy_score, f1_score

    buy = table["churn"].to_numpy() == 0
    tr_b, va_b, te_b = (split[k][buy[split[k]]] for k in ("train", "val", "test"))
    Xb = {
        k: table[FEATURE_COLUMNS].iloc[i].to_numpy(dtype=float) for k, i in (("tr", tr_b), ("va", va_b), ("te", te_b))
    }
    Sb = {k: seq[i] for k, i in (("tr", tr_b), ("va", va_b), ("te", te_b))}
    dtn = {k: np.log1p(table["days_to_next"].iloc[i].to_numpy()) for k, i in (("tr", tr_b), ("va", va_b), ("te", te_b))}
    reg_t: dict = {"n_train": int(len(tr_b)), "n_test": int(len(te_b))}
    lg = LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=15, random_state=seed, n_jobs=2, verbose=-1)
    lg.fit(Xb["tr"], dtn["tr"])
    lr_, _, secs_r = deep.fit_binary(
        deep.LSTMClassifier(seq.shape[2], d["lstm_hidden"], d["dropout"]),
        Sb["tr"],
        dtn["tr"],
        Sb["va"],
        dtn["va"],
        cfg,
        task="reg",
    )
    preds = {
        "static_lgbm": {k: lg.predict(Xb[k]) for k in ("va", "te")},
        "lstm_sequence": {k: deep.predict_raw(lr_, Sb[k]) for k in ("va", "te")},
    }
    preds["average"] = {k: (preds["static_lgbm"][k] + preds["lstm_sequence"][k]) / 2 for k in ("va", "te")}
    preds["train_median_baseline"] = {k: np.full(len(dtn[k]), np.median(dtn["tr"])) for k in ("va", "te")}
    for nm, pr in preds.items():
        out = {}
        for k, sp in (("va", "val"), ("te", "test")):
            yd, pd_ = np.expm1(dtn[k]), np.clip(np.expm1(pr[k]), 0, cfg["features"]["churn_horizon_days"])
            out[sp] = {
                "mae_days": float(np.mean(np.abs(yd - pd_))),
                "rmse_days": float(np.sqrt(np.mean((yd - pd_) ** 2))),
                "r2_log": float(1 - np.sum((dtn[k] - pr[k]) ** 2) / np.sum((dtn[k] - dtn[k].mean()) ** 2)),
            }
            runlog.add(
                "timing_days",
                nm,
                sp,
                {"mae": out[sp]["mae_days"], "rmse": out[sp]["rmse_days"], "r2": out[sp]["r2_log"]},
                secs_r,
                {},
            )
        reg_t[nm] = out
    meta["timing_days"] = reg_t
    torch.save(lr_.state_dict(), art / "lstm_days.pt")

    cat_all = table["next_category"].to_numpy()
    n_cat = int(cat_all.max()) + 1
    yc = {k: cat_all[i] for k, i in (("tr", tr_b), ("va", va_b), ("te", te_b))}
    cat_t: dict = {"n_classes": n_cat, "n_train": int(len(tr_b))}
    lgc = _LC(n_estimators=200, learning_rate=0.05, num_leaves=15, random_state=seed, n_jobs=2, verbose=-1)
    lgc.fit(Xb["tr"], yc["tr"])
    lc, _, secs_c = deep.fit_binary(
        deep.LSTMClassifier(seq.shape[2], d["lstm_hidden"], d["dropout"], n_out=n_cat),
        Sb["tr"],
        yc["tr"],
        Sb["va"],
        yc["va"],
        cfg,
        task="multi",
    )
    major = int(np.bincount(yc["tr"]).argmax())

    def _lg_pred(k: Any) -> Any:
        pr = np.zeros((len(yc[k]), n_cat))
        pr[:, lgc.classes_] = lgc.predict_proba(Xb[k])
        return pr

    probs = {"static_lgbm": {k: _lg_pred(k) for k in ("va", "te")}}
    probs["lstm_sequence"] = {
        k: torch.softmax(torch.tensor(deep.predict_raw(lc, Sb[k])), 1).numpy() for k in ("va", "te")
    }
    probs["average"] = {k: (probs["static_lgbm"][k] + probs["lstm_sequence"][k]) / 2 for k in ("va", "te")}
    for nm, pr in probs.items():
        cat_t[nm] = {}
        for k, sp in (("va", "val"), ("te", "test")):
            pc = pr[k].argmax(1)
            top3 = float(np.mean([yc[k][i] in np.argsort(-pr[k][i])[:3] for i in range(len(pc))]))
            cat_t[nm][sp] = {
                "accuracy": float(accuracy_score(yc[k], pc)),
                "macro_f1": float(f1_score(yc[k], pc, average="macro")),
                "top3_accuracy": top3,
            }
            runlog.add(
                "next_category",
                nm,
                sp,
                {"accuracy": cat_t[nm][sp]["accuracy"], "f1": cat_t[nm][sp]["macro_f1"]},
                secs_c,
                {},
            )
    cat_t["majority_baseline_test_accuracy"] = float(np.mean(yc["te"] == major))
    meta["next_category"] = cat_t
    torch.save(lc.state_dict(), art / "lstm_category.pt")

    # 5-fold cross-validation on TRAINING rows for the timing and category models too (brief: CV for all supervised
    # models). Fixed settings, no test data; deep folds hold out 15% of the fold's training part for early stopping.
    cv_extra = _cv_extra_models(
        cfg,
        {
            "timing30": (Xtr, S_tr, y30tr, "binary"),
            "timing_days": (Xb["tr"], Sb["tr"], dtn["tr"], "reg"),
            "next_category": (Xb["tr"], Sb["tr"], yc["tr"], "multi"),
        },
        {
            "pw30": pw30,
            "n_cat": n_cat,
            "lstm_hidden": d["lstm_hidden"],
            "dropout": d["dropout"],
            "clip": cfg["features"]["churn_horizon_days"],
        },
    )
    for task_, models_ in cv_extra.items():
        for nm_, res_ in models_.items():
            runlog.add(task_, nm_, "cv", {}, 0.0, res_)
    meta["cv_extra"] = cv_extra

    # autoencoder anomaly detector on spending-pattern features
    ae_cols = ["monetary_total", "monetary_avg", "avg_basket_size", "frequency", "return_rate", "gap_mean_days"]
    ae_idx = [FEATURE_COLUMNS.index(c) for c in ae_cols]
    ae_scaler = StandardScaler().fit(np.log1p(np.abs(Xtr[:, ae_idx])))
    A = lambda X: ae_scaler.transform(np.log1p(np.abs(X[:, ae_idx])))  # noqa: E731
    ae, ae_thr, hist_ae, secs = deep.fit_autoencoder(A(Xtr), A(Xva), cfg)
    flagged_test = float((deep.recon_error(ae, A(Xte)) > ae_thr).mean())
    runlog.add(
        "anomaly",
        "autoencoder",
        "test",
        {},
        secs,
        {"threshold": round(ae_thr, 4), "flag_rate_test": round(flagged_test, 4)},
    )
    torch.save(ae.state_dict(), art / "autoencoder.pt")
    joblib.dump({"scaler": ae_scaler, "cols": ae_cols, "threshold": ae_thr}, art / "autoencoder_meta.joblib")
    meta["anomaly"] = {"threshold": ae_thr, "percentile": d["anomaly_percentile"], "flag_rate_test": flagged_test}

    # loss curves (training vs validation) for ANN, LSTM, autoencoder
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, (nm, h) in zip(axes, [("ANN", hist_ann), ("LSTM", hist_lstm), ("Autoencoder", hist_ae)]):
        ax.plot(h["train_loss"], label="train")
        ax.plot(h["val_loss"], label="validation")
        ax.set_title(f"{nm} loss")
        ax.set_xlabel("epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(rep / "loss_curves.png", dpi=130)
    plt.close(fig)

    # ---- production model: best gradient-boosted tree by validation AUC ----
    gb = {k: v for k, v in results.items() if k in ("xgboost", "lightgbm")}
    prod = max(gb, key=lambda k: gb[k]["val"]["roc_auc"])
    joblib.dump(fitted[prod], art / "churn_model.joblib")
    meta["production_churn_model"] = prod
    meta["churn_results"] = results
    best_any = max(results, key=lambda k: results[k]["val"]["roc_auc"])
    meta["best_model_by_val_auc_overall"] = best_any

    # ---- CLV regression on log1p(next-window spend) ----
    ytr_c = np.log1p(table["future_spend"].iloc[split["train"]].to_numpy())
    yva_raw = table["future_spend"].iloc[split["val"]].to_numpy()
    yte_raw = table["future_spend"].iloc[split["test"]].to_numpy()
    clv_res, clv_fit = {}, {}
    for name, (est, grid) in classical.clv_candidates(cfg).items():
        best, params, secs = classical.tune(name, est, grid, Xtr, ytr_c, cfg, scoring="r2")
        clv_fit[name] = best
        cap = float(ytr_c.max())  # never extrapolate beyond the largest training target
        pv = np.expm1(np.clip(best.predict(Xva), 0, cap))
        pt = np.expm1(np.clip(best.predict(Xte), 0, cap))
        mv = {**reg_metrics(yva_raw, pv), "r2_log": float(reg_metrics(np.log1p(yva_raw), np.log1p(pv))["r2"])}
        mt = {**reg_metrics(yte_raw, pt), "r2_log": float(reg_metrics(np.log1p(yte_raw), np.log1p(pt))["r2"])}
        runlog.add("clv", name, "val", mv, secs, params)
        runlog.add("clv", name, "test", mt, secs, params)
        clv_res[name] = {"val": mv, "test": mt}
        log.info("clv model=%s val_r2=%.3f test_r2=%.3f test_r2_log=%.3f", name, mv["r2"], mt["r2"], mt["r2_log"])
    clv_best = max(clv_res, key=lambda k: clv_res[k]["val"]["r2"])
    for _n, _m in clv_fit.items():
        joblib.dump(_m, art / "candidates" / f"clv_{_n}.joblib")
    joblib.dump(clv_fit[clv_best], art / "clv_model.joblib")
    meta["clv_model"] = clv_best
    meta["clv_results"] = clv_res

    # ---- score everyone (flagging which rows were seen in training) ----
    Xall = table[FEATURE_COLUMNS].to_numpy(dtype=float)
    scores = pd.DataFrame(index=table.index)
    scores["churn_probability"] = fitted[prod].predict_proba(Xall)[:, 1]
    scores["clv_predicted"] = np.expm1(np.clip(clv_fit[clv_best].predict(Xall), 0, float(ytr_c.max())))
    err = deep.recon_error(ae, A(Xall))
    scores["anomaly_score"] = err
    scores["is_anomaly"] = (err > ae_thr).astype(int)
    split_col = np.empty(len(table), dtype=object)
    for k, v in split.items():
        split_col[v] = k
    scores["data_split"] = split_col
    scores.to_parquet(rpath(cfg, "scores"))

    # ---- consolidated comparison table (regenerated from results, ranked by AUC then recall) ----
    rows = [
        {
            "model": k,
            "val_roc_auc": v["val"]["roc_auc"],
            "test_roc_auc": v["test"]["roc_auc"],
            "test_recall": v["test"]["recall"],
            "test_precision": v["test"]["precision"],
            "test_f1": v["test"]["f1"],
            "test_accuracy": v["test"]["accuracy"],
            "test_tn": v["test"]["tn"],
            "test_fp": v["test"]["fp"],
            "test_fn": v["test"]["fn"],
            "test_tp": v["test"]["tp"],
            "val_recall": v["val"]["recall"],
        }
        for k, v in results.items()
    ]
    # ranked by VALIDATION metrics; test columns are reported once, never used for ordering
    comp = pd.DataFrame(rows).sort_values(["val_roc_auc", "val_recall"], ascending=False).reset_index(drop=True)
    comp.to_csv(rep / "model_comparison.csv", index=False)
    meta["targets"] = cfg["targets"]
    meta["targets_met"] = {
        "churn_auc": results[prod]["test"]["roc_auc"] >= cfg["targets"]["churn_auc"],
        "churn_recall": results[prod]["test"]["recall"] >= cfg["targets"]["churn_recall"],
        "clv_r2": clv_res[clv_best]["test"]["r2"] >= cfg["targets"]["clv_r2"],
    }
    (art / "metadata.json").write_text(json.dumps(meta, indent=2, default=float))
    log.info("production=%s targets_met=%s", prod, meta["targets_met"])
    return meta


if __name__ == "__main__":
    run()
