# Windows guide: run Vantara step by step

Your laptop: Windows 11 Home, 16 GB RAM, virtualization must be on. Docker Desktop is not installed yet.

**What is not verified:** the build environment had no container runtime, so `docker compose up --build` has never been run on
this project, and nothing here has been run on Windows. What I did check, on Linux: a fresh Python 3.10 and a fresh Python 3.11
environment each installed `requirements.txt`, passed the tests and started the API and dashboard from the packaged files.

**Windows 11 Home and Docker (checked against Docker's own docs, 7 Nov 2026):** the install page
(https://docs.docker.com/desktop/setup/install/windows-install/) lists the WSL 2 backend requirements as Windows 11 64-bit
"Enterprise, Pro, or Education" 23H2 or higher, with WSL 2.1.5 or later, 8 GB RAM and virtualization turned on in BIOS. The same page also says
"Windows Home or Education editions only allow you to run Linux containers", which implies Home can run Docker Desktop for Linux
containers (this project only needs those). Docker's own pages therefore do not give a clear yes or no for Windows 11 Home. Step 2 is the real test.
If Docker Desktop refuses to install or start on Home, stop and send the exact error text and a screenshot. Do not buy anything.

## 1. Unzip
Right-click `vantara.zip` > Extract All, to `C:\vantara`. You should see `docker-compose.yml` directly inside `C:\vantara`.

## 2. Install Docker Desktop
1. Download from https://www.docker.com/products/docker-desktop/ (free for personal and education use).
2. Run the installer, keep "Use WSL 2 instead of Hyper-V" ticked. Restart when asked.
3. Open Docker Desktop, accept the terms, wait until the bottom-left shows "Engine running".
4. If it says virtualization is disabled: restart, enter BIOS (usually F2 or Del), enable "Intel Virtualization Technology", save.
5. Check: open PowerShell and run `docker --version` and `docker compose version`. Both must print a version.

## 3. Start the project
First create your private settings file. This generates a random database password for you; you never type or share it:
```
cd C:\vantara
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
```
It prints "Created .env with a random local database password." (If you run it again it says ".env already exists", which is fine.)
Then start everything:
```
docker compose up --build
```
If `docker compose` complains that `POSTGRES_PASSWORD` is not set, the `.env` step above was skipped.
First run takes about 5 to 15 minutes (downloads Python, PyTorch CPU, PostgreSQL). Leave the window open. It is ready when you see
`api` log "Application startup complete" and `dashboard` log "You can now view your Streamlit app".

## 4. Verify (first-time-user checks)
1. http://localhost:8000/health shows `"status":"ok"`, `"database":"ok"` and `"model_loaded":true`.
2. http://localhost:8000/docs lists the endpoints. Try `GET /customers?limit=5`.
3. http://localhost:8501 shows the dashboard. Click each of the six views in the left sidebar: Segments, Churn leaderboard, Revenue trends, Customer explanation, Batch scoring, Model report. Charts should load within a few seconds.
4. In "Customer explanation" type a customer ID (for example 12347) and click Explain: you should see churn probability, purchase probability, predicted 90-day value, two bar charts (SHAP and LIME) and a table of recommended products. Customer 12346 shows a yellow note that its historical value is negative: that is the real data (returns and cancellations exceed purchases), not an error.
5. Open a second PowerShell: `docker compose ps` shows `db`, `api`, `dashboard` as running/healthy.

## 5. Stop and restart
Stop: press Ctrl+C, then `docker compose down`. Start again: `docker compose up`. Your data stays in the database volume. Never add `-v` to `down`: that deletes the database.

## 6. Optional: run the tests inside Docker
```
docker compose run --rm api pytest -q -p no:warnings
```
Expected: all tests pass (the number is printed at the end; this step has never been run in Docker, so if it fails, tell me the last 20 lines and carry on, it does not affect the app). The API tests use a separate `vantara_test` database inside the same PostgreSQL container, so your app data is untouched.

## What the dashboard numbers mean
- Churn probability: how likely the customer buys nothing in the next 90 days. Purchase probability is simply 1 minus that.
- Predicted 90-day value: expected spend over the next 90 days. Historical value is what the customer already spent, net of returns.
- Segment tables show `predicted_*` (model output) and `observed_*` (what really happened after the cut-off date, only for judging the models).
- Known misses, stated in the report: churn ROC-AUC 0.786 (target 0.80) and CLV R2 0.415 (target 0.60).

## Common problems
- "port is already allocated": something uses 8000 or 8501. Close it, or edit the left number in `ports:` in `docker-compose.yml`.
- Build is very slow or fails downloading: check internet, run the command again (it resumes cached layers).
- "WSL 2 installation is incomplete": in PowerShell (Admin) run `wsl --update`, restart.
- Docker says Home edition is not supported: send the message to me; do not buy Windows Pro without checking first.


## If you re-run the pipeline later
Just restart the api container (`docker compose restart api`). It notices the new model files, keeps the old predictions in a history table, and loads the new scores. Do not delete the database volume.
