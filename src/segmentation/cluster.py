"""Customer segmentation: K-Means plus DBSCAN/GMM comparison, with business-readable profiles."""

from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.cluster import DBSCAN, KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import davies_bouldin_score, silhouette_score  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.utils.config import load_config, rpath  # noqa: E402
from src.utils.logging_setup import get_logger  # noqa: E402

log = get_logger(__name__)
RECENT_DAYS = load_config()["segmentation"]["recent_days"]
SEG_COLS = ["recency_days", "frequency", "monetary_total", "avg_basket_size", "return_rate", "engagement_score"]


def seg_matrix(table: pd.DataFrame) -> np.ndarray:
    """Log-transform skewed fields (signed) for clustering."""
    X = table[SEG_COLS].copy()
    for c in ("frequency", "monetary_total", "avg_basket_size", "recency_days"):
        X[c] = np.log1p(X[c].clip(lower=0))
    return X.to_numpy(dtype=float)


def name_segments(profile: pd.DataFrame) -> dict[int, str]:
    """Readable names from centroid spend rank, recency (<=90 days = recent) and order count."""
    names: dict[int, str] = {}
    order = profile["monetary_total"].rank(ascending=False, method="first")
    n = len(profile)
    for seg in profile.index:
        recent = profile.loc[seg, "recency_days"] <= RECENT_DAYS
        if order[seg] == 1:
            nm = "Champions" if recent else "At-risk big spenders"
        elif order[seg] == n:
            nm = "Active small buyers" if recent else "Lapsed one-timers"
        else:
            nm = "Active regulars" if recent else "Fading regulars"
        names[seg] = nm
    seen: dict[str, int] = {}
    for seg, nm in list(names.items()):  # keep names unique
        seen[nm] = seen.get(nm, 0) + 1
        if seen[nm] > 1:
            names[seg] = f"{nm} ({seen[nm]})"
    return names


def run(cfg: dict | None = None) -> pd.DataFrame:
    """Fit segmentations, save assignments, profiles, comparison and plot."""
    cfg = cfg or load_config()
    sc = cfg["segmentation"]
    table = pd.read_parquet(rpath(cfg, "features"))
    # Everything is FITTED on training customers only (scaler, k-means, GMM, DBSCAN) and then applied to all customers.
    tr = (table["data_split"] == "train").to_numpy()
    scaler = StandardScaler().fit(seg_matrix(table[tr]))
    Xall = scaler.transform(seg_matrix(table))
    X = Xall[tr]
    best_k, best_s, scores = None, -1.0, {}
    inertia, dbi = {}, {}
    for k in range(sc["k_range"][0], sc["k_range"][1] + 1):
        km = KMeans(k, n_init=10, random_state=cfg["seed"]).fit(X)
        scores[k] = float(silhouette_score(X, km.labels_))
        inertia[k] = float(km.inertia_)
        dbi[k] = float(davies_bouldin_score(X, km.labels_))
        if scores[k] > best_s:
            best_k, best_s = k, scores[k]
    k_best_silhouette = best_k
    # business rule (a judgment call, not a statistical result): at least 3 segments so profiles are actionable
    best_k = max(best_k, 3) if scores.get(3, -1) > 0 else best_k
    km = KMeans(best_k, n_init=10, random_state=cfg["seed"]).fit(X)
    kl = km.predict(Xall)
    gmm = GaussianMixture(sc["gmm_components"], random_state=cfg["seed"]).fit(X)
    gl = gmm.predict(Xall)
    nn = NearestNeighbors(n_neighbors=sc["dbscan_min_samples"]).fit(X)
    eps = float(np.quantile(nn.kneighbors(X)[0][:, -1], sc["dbscan_eps_quantile"]))
    db = DBSCAN(eps=eps, min_samples=sc["dbscan_min_samples"]).fit(X)
    n_db = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    # DBSCAN has no predict: label other customers by their nearest core sample if within eps, else noise
    core = db.components_
    core_lab = db.labels_[db.core_sample_indices_]
    dbl = np.full(len(Xall), -1)
    if len(core):
        nn_core = NearestNeighbors(n_neighbors=1).fit(core)
        dist, idx = nn_core.kneighbors(Xall)
        dbl = np.where(dist[:, 0] <= eps, core_lab[idx[:, 0]], -1)
    te = (table["data_split"] == "test").to_numpy()
    comparison = {
        "fit_on": "train customers only (applied to all customers)",
        "kmeans": {
            "k": best_k,
            "k_best_silhouette": k_best_silhouette,
            "k_rule": "k=3 minimum for actionable profiles (business judgment); best silhouette alone would pick "
            f"k={k_best_silhouette}",
            "silhouette": float(silhouette_score(X, km.labels_)),
            "silhouette_test": float(silhouette_score(Xall[te], kl[te])),
            "davies_bouldin": float(davies_bouldin_score(X, km.labels_)),
            "davies_bouldin_test": float(davies_bouldin_score(Xall[te], kl[te])),
            "silhouette_by_k": scores,
            "inertia_by_k": inertia,
            "davies_bouldin_by_k": dbi,
        },
        "gmm": {
            "components": sc["gmm_components"],
            "silhouette": float(silhouette_score(X, gmm.predict(X))),
            "davies_bouldin": float(davies_bouldin_score(X, gmm.predict(X))),
            "bic": float(gmm.bic(X)),
        },
        "dbscan": {
            "eps": eps,
            "clusters": n_db,
            "noise_share": float((db.labels_ == -1).mean()),
            "silhouette": (
                float(silhouette_score(X[db.labels_ != -1], db.labels_[db.labels_ != -1])) if n_db > 1 else None
            ),
        },
    }
    # elbow + silhouette + Davies-Bouldin figure used to choose k
    ks = sorted(inertia)
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.2))
    axs[0].plot(ks, [inertia[k] for k in ks], "o-")
    axs[0].set_title("Elbow: inertia vs k")
    axs[1].plot(ks, [scores[k] for k in ks], "o-")
    axs[1].set_title("Silhouette vs k")
    axs[2].plot(ks, [dbi[k] for k in ks], "o-")
    axs[2].set_title("Davies-Bouldin vs k (lower is better)")
    for a_ in axs:
        a_.set_xlabel("k")
        a_.axvline(best_k, color="grey", ls="--", lw=0.8)
    fig.tight_layout()
    fig.savefig(rpath(cfg, "reports") / "kmeans_k_selection.png", dpi=130)
    plt.close(fig)
    km.labels_ = kl
    db.labels_ = dbl
    table = table.assign(segment_id=km.labels_)
    profile = table.groupby("segment_id")[SEG_COLS + ["churn", "future_spend"]].mean()
    profile["customers"] = table.groupby("segment_id").size()
    names = name_segments(profile)
    profile.insert(0, "segment_name", profile.index.map(names))
    seg = pd.DataFrame(
        {
            "segment_id": km.labels_,
            "segment_name": [names[s] for s in km.labels_],
            "gmm_segment": gl,
            "dbscan_segment": db.labels_,
        },
        index=table.index,
    )
    seg.to_parquet(rpath(cfg, "segments"))
    profile.round(3).to_csv(rpath(cfg, "reports") / "segment_profiles.csv")
    (rpath(cfg, "reports") / "segmentation_comparison.json").write_text(json.dumps(comparison, indent=2))
    joblib.dump(
        {"kmeans": km, "scaler": scaler, "names": names, "gmm": gmm, "dbscan": db, "dbscan_eps": eps},
        rpath(cfg, "artifacts") / "segmentation.joblib",
    )
    xy = PCA(2, random_state=cfg["seed"]).fit(X).transform(Xall)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for s_id in sorted(names):
        m = km.labels_ == s_id
        ax.scatter(xy[m, 0], xy[m, 1], s=6, alpha=0.5, label=names[s_id])
    ax.legend(markerscale=2, fontsize=7)
    ax.set_title("K-Means segments (PCA projection)")
    fig.tight_layout()
    fig.savefig(rpath(cfg, "reports") / "segments_pca.png", dpi=130)
    plt.close(fig)
    log.info("segmentation k=%d silhouette=%.3f dbscan_clusters=%d", best_k, comparison["kmeans"]["silhouette"], n_db)
    return seg


if __name__ == "__main__":
    run()
