# Roadmap: NYC 311 Weekly Forecast Productionization

Scope: productionize the XGBoost weekly call-volume-per-board forecaster
(`notebooks/04_wow_pred.ipynb` + `notebooks/nyc_api.py`). The transformer
notebooks (02, 03) are out of scope.

## 1. Repo restructure

- `git init`, `.gitignore` (`.venv/`, `data/00_cache/`, `data/01_raw/*.parquet`,
  `data/02_artifacts/`, `__pycache__/`, `mlruns/`).
- `pyproject.toml` pinning working versions (pandas, xgboost, scikit-learn,
  requests, requests-cache, mlflow, pyspark). Notebook-only deps
  (plotly, matplotlib, jupyter) go in an optional `notebooks` extra.
- `src/modeling/` package: `fetch.py` (from `nyc_api.py`), `features.py`
  (join/lag/event/weather logic), `train.py`, `infer.py`, `tracking.py`
  (MLflow tracking-URI/experiment setup shared by train and infer).
  Notebooks stay for exploration only.

**Service boundary:** one `modeling` service = train + infer + the MLflow
server, all co-located (same package, same Docker context). Prediction
serving is a deliberately separate future service — once predictions are
exposed via REST API, that becomes its own directory (e.g. `serving/`) that
reads from MLflow/the inference output, not a merge into `modeling`.

## 2. Scripts + local MLflow tracking

- Port fetch logic as-is; fix the hardcoded relative cache path so it works
  regardless of cwd.
- `features.py`: Spark used only for the join/feature-engineering step over
  the already-aggregated (small) tables — Socrata-side aggregation stays as
  the ingestion-scale win, Spark isn't a good fit for raw ingestion here.
- `train.py`: fetch → Spark feature build → collect to pandas → same
  `XGBRegressor` hyperparams → log params/metrics/model to MLflow (local
  file store for now) → log the training snapshot itself as an artifact.
- `infer.py`: load latest MLflow model, fetch current week's data, join
  against the last training snapshot for lag context, predict, write output.
- Validate end-to-end locally before adding any infra.

## 3. Containerize

- `modeling/` holds everything for this service: the `src/modeling` package,
  its Dockerfile (python slim + build-essential/libgomp1 for xgboost/pyarrow,
  parameterized entrypoint — `python -m modeling.train` vs
  `python -m modeling.infer`), and the MLflow server config (official MLflow
  image, no custom build needed) — one service, one docker-compose stack.
- `docker-compose.yml`: train/infer container + MLflow server with a real
  backend/artifact store, for local validation before k8s.
- `MLFLOW_TRACKING_URI` as an env var so scripts are agnostic to file-store
  vs. server.

## 4. Local Kubernetes + Argo

- Install `kind`, `kubectl`, `helm`, Argo Workflows, ArgoCD locally (none
  currently installed).
- Weekly training = Argo `CronWorkflow`; on-demand inference = a separate
  `Workflow`/`WorkflowTemplate`. Both replace Kedro's orchestration role.
- ArgoCD manages the MLflow server Deployment/Service/PVC (still part of the
  `modeling` service) via a git-watched manifests path in this repo (GitOps).
- Images loaded into kind directly (`kind load docker-image`) — no registry
  needed for a local-only cluster.

## 5. CI

- GitHub Actions: narrow live-date-range pull (e.g. last 2 weeks) → run
  `train.py` end-to-end → run `infer.py` against the resulting CI model →
  assert non-empty predictions, expected schema, sane metric bounds.

## 6. CD (next session)

- Wire ArgoCD sync from this repo; verify the CronWorkflow fires on schedule
  in the local cluster; document the GitOps loop (commit → CI → image →
  manifest bump → ArgoCD sync).

## Known tradeoffs

- PySpark is oversized for ~22K rows of aggregated data — included for the
  stated learning goal, not efficiency.
- Full Argo + ArgoCD + k8s stack is heavy relative to app size — same reason.
- No API keys required by any current data source (311, events, weather).
