# AGENTS.md

## Repository Shape
- The repo root has no Python/package manifest; the only checked-in Python project is `sandbox/jmro/proposals` with `pyproject.toml` and `uv.lock` and requires Python `>=3.12`.
- `sandbox/` is personal/team exploration space. Most member README files are empty; do not infer shared workflow from them.
- `services/mlflow` plus root `docker-compose.yaml` is the MLflow/Postgres service stack, separate from the sandbox analysis code.
- `thesis.tex` is a course report/outline template and currently does not define a checked-in LaTeX build workflow.

## Commands
- Run the Twitch exploration pipeline from `sandbox/jmro/proposals`: `uv run python twitch_gamers/scripts/run_exploration.py`.
- The root Docker stack expects `.env` copied from `.env.example`: `docker compose up --build` starts MLflow and Postgres.
- Verify MLflow S3 artifact logging with `uv run services/mlflow/test_s3_artifacts.py` from the repo root, after exporting `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and optionally `MLFLOW_S3_ENDPOINT_URL`.
- There is no repo-level lint, typecheck, or test command configured; avoid inventing one unless you add the config.

## Data And Artifacts
- Large/raw/generated data is intentionally ignored by `.gitignore` (`**/*data*/`, datasets/raw/processed/interim dirs, outputs/artifacts/results, `mlruns`, `models`, etc.). Do not commit downloaded datasets, generated feature CSVs, MLflow runs, or model artifacts.
- The Twitch pipeline code reads `twitch_gamers/data/renamed_large_twitch_edges.csv` and `renamed_large_twitch_features.csv`; the README's older `large_twitch_*.csv` names are not what the current loader uses.
- `run_exploration.py` writes `twitch_gamers/data/enhanced_features.csv` and PNGs under `twitch_gamers/results/`; these are generated outputs.

## Twitch Pipeline Notes
- `twitch_gamers/scripts/run_exploration.py` is the real orchestrator: load data, quality checks, stats, basic plots, network features, network plots, save enhanced data, summary.
- The scripts use direct sibling imports like `from log import log`, so run the documented script path rather than assuming `python -m twitch_gamers...` works.
- Network-heavy work is intentionally implemented with `scipy.sparse`; only k-core uses NetworkX. Avoid replacing sparse operations with pure NetworkX for the full 168K-node/6.8M-edge graph.
- `plots.py` sets the non-interactive Matplotlib `Agg` backend and the language world map reads Natural Earth data from a URL, so that plot needs network access.

## MLflow Notes
- `docker-compose.yaml` passes S3 credentials and `MLFLOW_S3_ENDPOINT_URL` into the MLflow container, but local artifact logging also needs those env vars because the MLflow client uploads artifacts directly to S3.
- The MLflow Docker image is `ghcr.io/mlflow/mlflow:v3.10.0-full` with `mlflow[auth]`, `psycopg2-binary`, and `boto3` installed in `services/mlflow/Dockerfile`.
