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

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **ie500_data_mining_project** (537 symbols, 781 relationships, 13 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/ie500_data_mining_project/context` | Codebase overview, check index freshness |
| `gitnexus://repo/ie500_data_mining_project/clusters` | All functional areas |
| `gitnexus://repo/ie500_data_mining_project/processes` | All execution flows |
| `gitnexus://repo/ie500_data_mining_project/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
