# Brief traceability: requirement, evidence, status

Status values: **Met** (implemented and checked by a test or a saved output), **Partly** (implemented with a stated gap), **Not met** (target missed or not built), **Not verified** (could not be run in the build environment). Numbers come from `models_artifacts/metadata.json` and `reports/`; the README tables are generated from the same files.

| Brief section | Requirement | Evidence | Status |
|---|---|---|---|
| 3.2 | Churn ROC-AUC >= 0.80 | metadata.json `targets_met.churn_auc`; test AUC 0.786 (xgboost) | **Not met**, discussed in report section 9 |
| 3.2 | CLV R2 >= 0.60 | `targets_met.clv_r2`; test R2 0.415 (ridge) | **Not met**, discussed in report section 9 |
| 3.2 | Churn recall >= 0.70 | test recall 0.765 | Met |
| 3.2 | API p95 < 400 ms | `reports/runtime_checks.json` (p95 about 6.6 ms, localhost, includes SHAP) | Met (not measured under Docker) |
| 3.2 | Dashboard load < 3 s | warm 1.7-2.3 s, first cold load about 4.7 s | Partly |
| 3.2 | Single-command pipeline | `python -m src.pipeline`; config in `config/config.yaml` | Met |
| 4.1 | Six classical churn models: Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM and one distance-based model (the brief allows KNN **or** SVM) | `src/models/classical.py`, `reports/model_comparison.csv` | Met (KNN as the distance-based model; SVM is the brief's alternative option, not an additional requirement) |
| 4.1 | ANN, LSTM, autoencoder | `src/models/deep.py`, `models_artifacts/*.pt` | Met |
| 4.1 | K-Means plus GMM/DBSCAN | `src/segmentation/cluster.py`, `segmentation.joblib` holds all three | Met |
| 4.1 | SHAP for best churn model, LIME example | `src/explainability/explain.py`; SHAP is for the production model (xgboost), test AUCs are within 0.004 of each other (logistic regression 0.790, random forest 0.790, xgboost 0.786); xgboost was chosen on validation AUC, not test | Met, with that note |
| 4.1 | FastAPI, Streamlit, Docker | `api/`, `frontend/`, `Dockerfile`, `docker-compose.yml` | Met; compose verified November 8, 2026 on Windows 11 Home (Docker Desktop 4.98.0): fresh extraction of the delivered November 7 ZIP (this revision's extra changes - leaderboard heading text, batch-PDF number formatting, one dependency pin, documentation - do not touch the Docker files that run exercised), `docker compose up --build`, all three containers healthy, API seeded, all six dashboard views rendered |
| 1, 2.1 | Personalised product recommendations | `src/recommend/recommender.py`, `GET /customers/{id}/recommendations`, dashboard panel; offline hit rate@10 0.348 vs 0.263 popularity | Met as offline model; no A/B test (out of scope per 4.2) |
| 1 | Next-purchase probability | `purchase_probability_90d` = 1 - churn probability (the brief's section 16.1 says the classifier approximates it) | Met |
| 5.1 | Targets: churn, CLV, Next Purchase Category | `src/features/build.py` (`churn`, `future_spend`, `next_category`) | Met |
| 5.3 | Missing IDs, returns, bad prices, admin StockCodes, seasonality | `src/data/clean.py`, config `non_product_stockcodes`, `seasonal_concentration` | Met |
| 5.5 | UCI source, attribution | `src/data/load.py`, README, report | Met |
| 6.2 | Dedup keeps legitimate repeats | exact duplicate rows only are dropped; this cannot separate a genuine identical repeated line from a duplicate | Partly (documented rule) |
| 6.2 | IQR plus domain-rule outliers | `src/data/reference.py` (train-fitted bounds) | Met |
| 6.2 | Validation: schema, null-rate thresholds, date range, fail loudly | `src/data/validate.py`, `data.max_null_rate_raw` in config, `tests/test_round2.py` | Met |
| 6.3 | One-hot Country | `country_*` columns | Met |
| 6.3 | Frequency encoding of StockCode features | `product_popularity_mean`, `niche_product_share`, `reference_stats.joblib["stock_freq"]` (train customers only) | Met |
| 6.3 | StandardScaler for scale-sensitive models; persist scalers/encoders | pipelines inside saved candidates; `scaler.joblib`; `reference_stats.joblib` | Met |
| 6.4 | EDA: summaries, correlations, VIF, country/season, hypotheses | `notebooks/01_eda.ipynb` | Met |
| 7 | All listed features with one-line justification | `docs/FEATURES.md`, test checks every model column is documented | Met |
| 7 | Point-in-time discipline and leakage tests | `tests/test_features.py`, `assert_no_leakage` | Met |
| 8.1 | CV-tuned classical models | `src/models/classical.py` | Met |
| 8.2 | ANN with batch norm, dropout, early stopping | `src/models/deep.py` | Met |
| 8.2 | LSTM on amount, gap, category for timing | `build_sequences`; results in report: LSTM does not beat static LightGBM for days-to-next (MAE 22.0 vs 20.6 days) | Built; benefit **not shown** |
| 8.3 | Fixed seed, stratified 70/15/15 | `config.yaml`, `split_ids` | Met |
| 8.3 | CV on train only; test touched once | selection uses train/validation only; test metrics are computed once per model but inside the training script, not in a separate final stage | Partly (wording corrected in report) |
| 8.3 | Every run logged | `reports/experiment_log.csv` | Met |
| 9 | K by elbow and silhouette; GMM/DBSCAN; business profiles | `reports/kmeans_k_selection.png`, `segmentation_comparison.json`. k=2 has best silhouette; k=3 chosen for business use (stated as judgment) | Met, judgment stated |
| 9 | Hierarchical clustering (optional, "may") | not built | Not built (optional) |
| 10 | SHAP summary and 3 force plots, LIME vs SHAP, PDPs, plain-language template | `reports/shap_*`, `lime_borderline.png`, `partial_dependence.png`, `plain_language` | Met |
| 11 | Metrics per task incl. Davies-Bouldin; 5-fold stratified CV for all supervised models; comparison table | `model_comparison.csv`, `experiment_log.csv` (cv_mean/cv_std for churn models, ANN, LSTM, CLV models; since round 3 also rows with split=cv for the static LightGBM and LSTM 30-day, days-to-next and next-category models, and `cv_extra` in metadata.json; report section 4 CV table). The timing/category folds use fixed settings and early-stop on 15% of each fold's training part; the autoencoder and K-Means/GMM/DBSCAN are unsupervised and have no CV | Met |
| 12.1 | Prediction, batch CSV, metadata, health; PostgreSQL persistence; Pydantic validation | `api/`, `tests/test_api.py` | Met |
| 12.2 | Dashboard views, SHAP and LIME panel, CSV upload, PDF/CSV report | `frontend/dashboard.py`, screenshots in `docs/screens/` | Met |
| 12.3 | Single `docker compose up`, env-var configuration | compose file passes DATABASE_URL, LOG_LEVEL, MODEL_ARTIFACTS_DIR; needs `.env` from `scripts/setup_env.py`. A fresh unzip on Linux was started from artifacts against an empty PostgreSQL database (no Docker): API, seeding, explain, LIME, recommendations all worked. Verified under Docker on Windows 11 Home on November 8, 2026: fresh extraction of the delivered November 7 ZIP (this revision's extra changes do not touch the Docker files that run exercised), new `.env` from `scripts/setup_env.ps1`, `docker compose up --build`, all three containers healthy, seeded API answered single-customer and explanation requests, batch CSV scoring matched single-customer scores, PDF export worked. Not yet exercised: `/docs` page and data persistence across container restart | Met |
| 13 | Type hints, docstrings, linter clean | AST check (0 unannotated functions); `ruff check .` enforces docstrings on every public module, class and function in `src` and `api` (rules D100-D104, `noqa: D102` removed); tests are excluded from the docstring rule | Met |
| 13 | Structured logging; YAML config; seed and requirements | `src/utils/logging_setup.py`; feature thresholds and windows, risk cut-offs, segment recency and recommender settings are in `config/config.yaml` (tests check they are read from it). Still in code: category keyword lists and the fixed 12 x 30-day trend bucketing | Partly |
| 13 | Tests >= 70% coverage, leakage tests | 62 tests, 77% total coverage; coverage over `src/` alone is 72.8% (see docs/coverage.txt) | Met |
| 13 | No committed credentials | no password in compose/config; `.env` generated randomly and git-ignored; test in `tests/test_round2.py` | Met |
| 14, 18 | Architecture, ER, workflow diagrams | `docs/architecture_diagram.png`, `er_diagram.png`, `workflow_diagram.png` | Met |
| 16 | Mathematical appendix, loss curves, L1/L2 penalties | report appendix, `reports/loss_curves.png`; L1/L2 grid in logistic regression, L2 weight decay and L1 loss term in ANN/LSTM | Met |
| 17 | Recorded walkthrough | not produced | Not done (not part of this build) |
| 18 | Serialized artifacts for all models | `models_artifacts/` and `models_artifacts/candidates/` | Met |
| 18 | README for a clean-clone run | `README.md`, `docs/WINDOWS_GUIDE.md` | Met; Docker part verified on Windows November 8, 2026 |
| 20.5 | Runs via docker-compose on a clean machine | Windows 11 Home + Docker Desktop 4.98.0, fresh extraction of the delivered November 7 ZIP, November 8, 2026: `docker compose up --build` from an admin prompt, all containers healthy, API + dashboard verified | Met |
| 20.7 | Targets met or discussed | two targets missed, discussed in report section 9 | Met (discussion), targets themselves not met |

Not in scope or optional and not built: Transformer, CatBoost, SMOTE (class weights are used instead), Customer Personality dataset, hierarchical clustering, live A/B testing, real-time ingestion, authentication.

## Round-3 notes (semantics worth knowing)

- **CLV definition.** The brief defines CLV as "historical + predicted". `clv_total_projected` is exactly `clv_historical + clv_predicted_90d`, with no clamp. `clv_historical` is net of returns and cancellations, so a few customers (for example 12346) have negative historical and total values; the dashboard flags this. `value_at_risk` is the only place negatives are floored at 0, because an amount at risk cannot be negative.
- **Segment API.** `/segments` returns `predicted_churn_probability` and `predicted_clv_90d` (model outputs for scored customers) and `observed_churn_rate` and `observed_future_spend` (what actually happened in the 90 days after the cutoff). The observed columns are for evaluation only and are labelled so in the dashboard.
- **Autoencoder.** Adding the CV stage before the autoencoder changed the random-number sequence, so its validation threshold moved from 0.141 to 0.159 and the test flag rate from 0.5% to 0.8%. All churn, CLV, timing and category metrics are identical to the previous run. Nothing was tuned on the test split.
- **Not fixed, by decision or constraint:** recorded walkthrough, hierarchical clustering, exact-duplicate rule limit, test metrics computed inside each training loop. Docker on Windows was verified after the original build (November 8, 2026, see rows 12.3 and 20.5).
- **Fresh dependency install.** Verified on Linux, Python 3.10, in a new virtual environment: `pip install torch` (CPU index) then `pip install -r requirements.txt`, then 62 tests pass and the API and dashboard start from the packaged artifacts. This exposed `pyyaml==5.4.1` failing to build; fixed to 6.0.2. Docker (Python 3.11 image) on Windows 11 Home was verified on November 8, 2026 as described in rows 12.3 and 20.5.
- **Python 3.11 install (the Dockerfile base).** Verified on Linux in a new environment (uv, Python 3.11): `requirements.txt` installs, 62 tests pass, API and dashboard start. That environment got PyPI's default PyTorch build (CUDA wheels, runs on CPU) rather than the CPU-only build the Dockerfile installs first. `requirements.txt` now says `torch==2.14.1` (not `+cpu`) so both installs resolve; with the CPU build pre-installed, pip keeps it (checked on 3.10).
