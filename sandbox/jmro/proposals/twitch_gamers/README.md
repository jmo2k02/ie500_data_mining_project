
## ToC

- [SNAP Source](https://snap.stanford.edu/data/twitch_gamers.html)
- [Github](https://github.com/benedekrozemberczki/datasets)
- [Paper](https://arxiv.org/abs/2101.03091)

## Vertex Attributes

These are the attributes provided per Vertex

![alt text](./assets/image.png)


## Possible Applications

- Categorical attributes such as **DeadAccount**, **AffiliateStatus**, and **Explicit Content** can be used as targets for <u>**binary classification**</u>
- **Broadcaster Language** can be used for multi-class node classification with more than 20 categories.
- **View Count** and **Account Lifetime** can serve as target for **count data regression problems** at the node level.
- **Other possiblities:**
  - link prediction
  - community dectection with ground truth labels


## Project Structure

```
twitch_gamers/
├── data/                          # Dataset files
│   ├── large_twitch_edges.csv     # 6.8M edges (mutual follower relationships)
│   ├── large_twitch_features.csv  # 168K nodes, 9 attributes
│   └── enhanced_features.csv      # (generated) features + network metrics
├── scripts/                       # Modular exploration pipeline
│   ├── log.py                     # Logging: elapsed timer, section headers, step_timer()
│   ├── load.py                    # Data loading + path configuration
│   ├── stats.py                   # Quality checks, descriptive statistics, summary
│   ├── network.py                 # Network features via scipy.sparse (degree, clustering, PageRank, k-core)
│   ├── plots.py                   # All visualizations (basic + network feature plots)
│   └── run_exploration.py         # Orchestrator — runs the full pipeline
├── notebooks/
│   └── 01_data_exploration.ipynb  # Interactive exploration notebook
├── results/                       # Generated plots (PNG)
├── reports/                       # Outline / report drafts
├── first_exploration.ipynb        # Initial quick-look notebook
├── 2101.03091v2.pdf               # Reference paper
└── README.md                      # This file
```

### Running the pipeline

```bash
cd jmro/proposals
uv run python twitch_gamers/scripts/run_exploration.py
```

The script logs every step with timestamps and elapsed time so you can track progress:

```
14:32:01 [00:00.00] INFO  | ============================================================
14:32:01 [00:00.00] INFO  | 1. LOADING DATA
14:32:01 [00:00.00] INFO  | ============================================================
14:32:01 [00:00.01] INFO  |   Loading edges...
14:32:04 [00:03.21] INFO  |   Loading edges done (3.2s)
...
```

## Module Reference

### `log.py` — Logging utilities

Provides a pre-configured logger (`log`) that prints to stdout with a format showing wall-clock time, elapsed time since script start (`[MM:SS.ss]`), and log level. Keeps all output consistent across modules.

**Key exports:**

| Name | Type | Description |
|---|---|---|
| `log` | `logging.Logger` | Shared logger instance. Use `log.info(...)`, `log.warning(...)`, etc. |
| `section(num, title)` | function | Prints a numbered section header surrounded by `===` bars. |
| `step_timer(label)` | context manager | Wraps a block of work — logs `"<label>..."` on entry and `"<label> done (Xs)"` on exit with wall-clock duration. |
| `total_elapsed()` | function | Returns a human-readable string like `"2m 14.3s"` of total time since the script started. |

### `load.py` — Data loading and paths

Resolves all file paths relative to the script location so the pipeline works regardless of the working directory. Exposes the path constants and a single loader function.

**Key exports:**

| Name | Type | Description |
|---|---|---|
| `DATA_DIR` | `str` | Absolute path to `twitch_gamers/data/`. |
| `RESULTS_DIR` | `str` | Absolute path to `twitch_gamers/results/` (created if missing). |
| `load_data()` | function | Reads `large_twitch_edges.csv` and `large_twitch_features.csv`, returns `(edges_df, features_df)`. |

### `stats.py` — Data quality and descriptive statistics

Pure reporting module — reads the DataFrames, logs findings, never mutates data.

**Key exports:**

| Name | Description |
|---|---|
| `check_quality(features_df)` | Checks for missing values, duplicate `numeric_id`s, and logs column dtypes. |
| `describe_features(features_df)` | Logs `.describe()` output, binary feature class balance, language distribution (top 10), numerical feature summaries (mean/median/std/skew), and string column info. |
| `summarize(features_df, edges_df, density)` | Final summary after feature engineering — classification target balance, regression target stats, network characteristics. |

### `network.py` — Network analysis and feature engineering

Handles all graph computations. Uses **scipy.sparse** matrices (C/BLAS) instead of pure-Python NetworkX for the expensive operations, making degree, clustering coefficient, and PageRank orders of magnitude faster on our 168K-node / 6.8M-edge graph.

**Key exports:**

| Name | Description |
|---|---|
| `build_adjacency(edges_df, features_df)` | Builds a sparse symmetric binary adjacency matrix. Returns `(A, N)`. |
| `compute_components(A, N)` | Connected components via `scipy.sparse.csgraph`. Returns `(n_components, largest_size)`. |
| `compute_degree(A, N, features_df)` | Row-sums of the adjacency matrix. Adds `degree` column. |
| `compute_clustering(A, N, degree_array, features_df)` | Clustering coefficient via the sparse `A * A^2` element-wise trick: `cc[i] = (A .* A^2).sum(row=i) / (deg[i] * (deg[i]-1))`. Adds `clustering_coeff` column. |
| `compute_pagerank(A, N, degree_array, features_df)` | PageRank via sparse power iteration (alpha=0.85, tol=1e-6). Adds `pagerank` column. |
| `compute_core_number(edges_df, features_df)` | K-core decomposition via NetworkX (already O(m), fast). Adds `core_number` column. |
| `run_all(edges_df, features_df)` | Convenience function that runs all of the above in order. Returns a `NetworkResult` dataclass with graph-level stats. |

**`NetworkResult` dataclass fields:** `n_nodes`, `n_edges`, `density`, `n_components`, `largest_component`.

### `plots.py` — Visualizations

All matplotlib/seaborn plotting. Each function takes a DataFrame + output directory, saves a PNG, and logs the filename. Matplotlib is configured for non-interactive (`Agg`) backend.

**Basic (pre-network) plots** — called via `plot_all_basic(features_df, results_dir)`:

| Function | Output file | What it shows |
|---|---|---|
| `plot_binary_features()` | `binary_features_distribution.png` | Bar charts for mature, dead_account, affiliate |
| `plot_language()` | `language_distribution.png` | Horizontal bar chart of top 15 languages |
| `plot_views()` | `views_distribution.png` | Histograms of views (linear + log10 scale) |
| `plot_lifetime()` | `lifetime_distribution.png` | Histograms of life_time (linear + log10 scale) |
| `plot_correlation()` | `correlation_matrix.png` | Heatmap of numerical feature correlations |

**Network feature plots** — called via `plot_all_network(features_df, results_dir)`:

| Function | Output file | What it shows |
|---|---|---|
| `plot_degree()` | `degree_distribution.png` | Degree distribution (linear + log10) |
| `plot_clustering()` | `clustering_coefficient_distribution.png` | Clustering coefficient histogram |
| `plot_pagerank()` | `pagerank_distribution.png` | PageRank distribution (linear + log10) |
| `plot_core_number()` | `core_number_distribution.png` | K-core number histogram |
| `plot_correlation()` | `correlation_matrix_with_network.png` | Updated correlation heatmap including network features |

### `run_exploration.py` — Orchestrator

The entry point. Calls the other modules in sequence:

1. Load data (`load.py`)
2. Quality checks (`stats.py`)
3. Descriptive statistics (`stats.py`)
4. Basic visualizations (`plots.py`)
5. Network analysis + feature engineering (`network.py`)
6. Network feature visualizations (`plots.py`)
7. Save enhanced dataset to `data/enhanced_features.csv`
8. Print summary (`stats.py`)
