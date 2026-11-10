# Vantara: customer behavior prediction on Online Retail II

Student project for Internmo (PROJ-VANT-260704). It predicts 90-day churn, estimates 90-day customer value (CLV),
segments customers, explains predictions, and serves everything through a FastAPI service, a PostgreSQL database and a
Streamlit dashboard. Data: UCI Online Retail II (1,067,371 invoice lines, Dec 2009 to Dec 2011, CC BY 4.0).

## Results (generated from the saved metric files, not typed by hand)

<!-- METRICS:START -->
| Target | Required | Measured | Status |
|---|---|---|---|
| Churn ROC-AUC | >= 0.8 | 0.786 (xgboost) | NOT MET |
| Churn recall | >= 0.7 | 0.765 | met |
| CLV R2 | >= 0.6 | 0.415 (ridge) | NOT MET |
| API p95 latency | < 400 ms | 6.6 ms (p50 6.1 ms) | met |
| Dashboard load | < 3 s | 1.72-2.25 s warm; cold first load was slower | warm only |

Churn models, test split (reported once):

| Model | ROC-AUC | Recall | Precision | F1 | Accuracy |
|---|---|---|---|---|---|
| logistic_regression | 0.790 | 0.767 | 0.758 | 0.763 | 0.730 |
| random_forest | 0.790 | 0.783 | 0.760 | 0.771 | 0.737 |
| ann | 0.787 | 0.758 | 0.763 | 0.760 | 0.730 |
| xgboost | 0.786 | 0.765 | 0.765 | 0.765 | 0.734 |
| knn | 0.778 | 0.785 | 0.737 | 0.760 | 0.720 |
| lightgbm | 0.777 | 0.760 | 0.745 | 0.752 | 0.717 |
| decision_tree | 0.753 | 0.783 | 0.743 | 0.762 | 0.723 |
| lstm | 0.740 | 0.724 | 0.726 | 0.725 | 0.689 |
<!-- METRICS:END -->

Churn recall (>= 0.70) is met. Churn ROC-AUC (>= 0.80) and CLV R2 (>= 0.60) are **not met**. Many model families land near
test ROC-AUC 0.79 on this label, so this looks like a label-noise ceiling rather than a tuning gap. The production churn model
(xgboost, early stopping on a train-only fold) was picked on validation AUC among gradient-boosted candidates (validation AUC
0.811; Random Forest 0.808, test 0.790 vs 0.786). For CLV (90-day future spend) ridge was selected on
validation (test R2 0.415); Random Forest has a higher test R2 (0.463) but was not chosen, because the test split is never used
to choose anything. CLV is heavy-tailed (many customers spend nothing in the window). Outlier bounds, description and price
lookups, StockCode frequencies and engagement percentiles are fitted on train customers' pre-cutoff rows only. Cross-validation is 5-fold stratified.
The table above is ordered by test AUC for display only.
Full discussion: `docs/final_report.pdf`.

## Run with Docker (required path)

> Re-running the pipeline later is safe: on its next start the API compares a SHA-256 of the model/score artifacts with the latest `score_runs` row, archives the old prediction rows into `prediction_history`, and upserts the new scores. Nothing is deleted. Check `GET /model/metadata` (`score_run.in_sync`), or refresh explicitly with `docker compose exec api python -m src.db.session`.


```
python scripts/setup_env.py        # Windows without Python: powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
docker compose up --build
```

`setup_env` writes a git-ignored `.env` containing a random database password. No credentials are committed; compose refuses to start without one.

Open http://localhost:8501 (dashboard) and http://localhost:8000/docs (API). First start takes a few minutes: it builds the image
and seeds PostgreSQL with 5,248 customers and 42,554 invoices from the shipped `data/processed` files.

**Honest status:** the compose file was written and YAML-validated but could not be executed in the build environment (no
container runtime). The API was run against a real PostgreSQL 16.2 server (pgserver) and the dashboard against that API;
the container build itself is unverified until you run it. See `docs/WINDOWS_GUIDE.md`.

## Run without Docker (Linux/macOS, needs your own PostgreSQL)

```
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
export DATABASE_URL=postgresql+psycopg2://user:pass@localhost:5432/vantara
uvicorn api.main:app --port 8000 &
API_URL=http://localhost:8000 streamlit run frontend/dashboard.py
```

## Rebuild everything from raw data

```
# get the raw data first: download online_retail_II.xlsx (UCI dataset 502) into data/raw/
# direct link: https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip (unzip it)
# (the ucimlrepo fallback currently fails: UCI's API reports dataset 502 as "not available for import", checked November 8, 2026)
python -m src.pipeline
python -m src.report.readme_tables
pytest --cov=src --cov=api      # 62 tests, 77% coverage (docs/coverage.txt)
ruff check . && black --check .
```

## Layout

- `src/data` loading, cleaning, validation; `src/features` features and label (cutoff 2011-09-09, 90-day window)
- `src/models` classical and deep models, training, experiment log; `src/segmentation`; `src/explainability` (SHAP, LIME, PDP)
- `src/db` PostgreSQL schema; `api/` FastAPI; `frontend/dashboard.py`; `notebooks/` EDA, features, experiments
- `reports/` metrics and plots; `docs/` report, diagrams, guide; `tests/`

## Method notes

- Split is by customer (train/val/test 70/15/15, seed 42). All features use only data before the cutoff; a test rewrites every
  later line and checks features do not change. Scalers and encoders are fit on train only.
- Churn = no purchase in the 90 days after the cutoff. CLV target = net spend in that window, floored at 0.
- Anomaly detection is an autoencoder (99th-percentile threshold on validation reconstruction error).
- K-Means k=3 was chosen for interpretability; k=2 has a higher silhouette (0.362 vs 0.262). DBSCAN found one cluster.
- Dashboard "< 3 s" holds on warm loads only; the cold first load measured about 4.7 s.

## Audit-fix additions (second pass)

- Segmentation scaler, KMeans, GMM and DBSCAN are fitted on the training split only, then applied to all customers. Silhouette and Davies-Bouldin are reported for every k and for held-out customers (`reports/segmentation_comparison.json`, elbow figure `reports/kmeans_k_selection.png`).
- API medians and the CLV cap come from the training split only.
- LIME is available in the dashboard (Customer explanation view) and at `GET /customers/{id}/explain/lime`, next to SHAP.
- ANN and LSTM churn models also have 5-fold stratified CV; every candidate model is serialized in `models_artifacts/candidates/`.
- CLV: `clv_predicted_90d` is a 90-day horizon forecast; `clv_total_projected` = `clv_historical` + `clv_predicted_90d`, exactly the brief's "historical + predicted" (API and dashboard). No clamp: `clv_historical` is net of returns, so it can be negative for a few customers (the dashboard flags this), and their total can be negative too. `value_at_risk` on the leaderboard does floor historical value at 0, because a negative amount at risk has no meaning. `/segments` returns `predicted_*` (model outputs) next to `observed_*` columns (what actually happened after the cutoff, for evaluation only).
- Next purchase (buyers only, test): static LightGBM days-to-next MAE 20.6, LSTM 22.0, train-median baseline 22.2 days (the LSTM does not beat the static model for timing). Next-category accuracy: LSTM 0.503, static 0.471, majority baseline 0.418. Modest gains, reported as is.
- Coverage output: `docs/coverage.txt`.
- Round-two fixes: StockCode frequency encoding (`product_popularity_mean`, `niche_product_share`, fitted on training customers, persisted), an item-item product recommender (`GET /customers/{id}/recommendations`, dashboard panel; offline hit rate@10 on test buyers in `models_artifacts/recommender_eval.json`, no A/B test), configurable null-rate validation, L1/L2 penalties (logistic regression grid, ANN/LSTM loss), GMM and DBSCAN serialized, type hints on every function, `docs/FEATURES.md` per-feature justification, `purchase_probability_90d` in API and dashboard, and no committed credentials (`scripts/setup_env.py` writes a random password into a git-ignored `.env`).
- Historical customer value (`clv_historical`) is net of returns and cancellations, so a few customers show negative values (for example customer 12346, whose large cancelled order outweighs their purchases). This is the real record and is left unchanged; the dashboard flags it.
- Test note: database-backed tests start a throwaway PostgreSQL via the `pgserver` package when `DATABASE_URL` is not set (`pip install pgserver`, not needed for Docker); without either, those tests are skipped, so the 62-pass and 77% numbers need one of the two. Dashboard timing: a warm API call path measures about 1.7 to 2.2 s (`reports/runtime_checks.json`); the first cold browser load measured about 4.7 s in a separate check, so the under-3-second target is only met warm.
- Round-three fixes: thresholds and windows moved into `config/config.yaml`, `MODEL_ARTIFACTS_DIR` environment variable for artifact paths, ruff-enforced docstrings, 5-fold CV for the timing and category models (`cv_extra` in `models_artifacts/metadata.json`), `/segments` split into `predicted_*` and `observed_*` columns, `clv_total_projected` back to the brief's plain historical + predicted. See `docs/BRIEF_TRACEABILITY.md`.
- Fresh-install check: a brand-new Python 3.10 virtual environment on Linux installed `requirements.txt` (after the CPU build of PyTorch) and ran the 62 tests and the API and dashboard. That check found and fixed a real bug: `pyyaml==5.4.1` cannot be built on current toolchains, so it is now `pyyaml==6.0.2`. Docker check (November 8, 2026): on Windows 11 Home with Docker Desktop 4.98.0, a fresh extraction of the delivered November 7 ZIP (this revision additionally changes the leaderboard heading text, batch-PDF number formatting, one dependency pin and documentation; the Docker-related files the run exercised are unchanged) with a new `.env` from `scripts/setup_env.ps1` ran `docker compose up --build`; all three containers came up healthy, the API seeded PostgreSQL and answered requests, the dashboard rendered all six views, batch CSV scoring matched single-customer scores, and PDF export worked. Not yet exercised under Docker: the `/docs` page and data persistence across a container restart.
- Known limits: churn AUC and CLV R2 targets are not met; per-customer LSTM timing/category predictions are batch research outputs, not served by the API.

## Deployment (free-tier serving)

The public demo deployment serves the trained autoencoder's inference in NumPy
(`src/models/ae_serve.py`, weights in `models_artifacts/autoencoder_weights.json`)
instead of importing torch at serving time. Training is unchanged and still uses
torch (`src/models/deep.py`); this only shrinks the serving container so the API
fits a 512 MB free instance (measured ~476 MB under SHAP/LIME/predict load with
`uvicorn --limit-concurrency 2`, vs ~625 MB with torch). Outputs are validated
against the torch model before deploy: anomaly flags match on all 5,248 scored
customers. The deployment image installs `requirements-deploy.txt`
(= `requirements.txt` without torch). Weights: `models_artifacts/autoencoder_weights.json`.
## Multi-user mode (login + private workspaces)

The deployed app can run multi-user on top of Supabase Auth (free tier):

- Sign-up/login with email + password, email confirmation required. Supabase
  manages passwords and sessions; the app never stores or logs them.
- Each user gets a private workspace: customers, uploads, segments and
  analytics are stored with that user's id and are invisible to other users.
  The API enforces ownership on every route (queries filter by the caller's
  `user_id`; cross-user ids return 404).
- New accounts start with the same sample dataset as the public demo so the
  dashboard is populated immediately.
- Users can upload their own CSV if it matches the expected schema (same
  columns as the sample data). Limits: 5 MB / 50,000 rows per file, 3 datasets
  per user, and a global cap of 1,000,000 stored rows so the free database
  cannot be exhausted.

Auth verification: the API validates Supabase access tokens against the
project's JWKS endpoint (`<SUPABASE_URL>/auth/v1/.well-known/jwks.json`),
checking signature (ES256/RS256), issuer, audience and expiry. A local
test-only shared secret exists solely for the automated test suite and is
inert whenever `SUPABASE_URL` is configured.

Required environment variables (API + dashboard): `SUPABASE_URL`,
`SUPABASE_ANON_KEY`, `SUPABASE_JWT_AUD` (default `authenticated`).
The dashboard also uses the anon key client-side for the Supabase JS auth
flow. Without these variables the app runs in the original single-user demo
mode.
