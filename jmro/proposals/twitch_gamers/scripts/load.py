"""
Data loading and path configuration.
"""

import os
import pandas as pd
from log import log, step_timer

# ── Resolve paths relative to this file ───────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_TWITCH_DIR = os.path.dirname(_SCRIPT_DIR)

DATA_DIR = os.path.join(_TWITCH_DIR, "data")
RESULTS_DIR = os.path.join(_TWITCH_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

EDGES_FILE = os.path.join(DATA_DIR, "renamed_large_twitch_edges.csv")
FEATURES_FILE = os.path.join(DATA_DIR, "renamed_large_twitch_features.csv")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load edges and features CSVs. Returns (edges_df, features_df)."""
    log.info(f"Data dir:    {DATA_DIR}")
    log.info(f"Results dir: {RESULTS_DIR}")

    with step_timer("Loading edges"):
        edges_df = pd.read_csv(EDGES_FILE)
    log.info(f"  Edges: {len(edges_df):,} rows, cols={edges_df.columns.tolist()}")

    with step_timer("Loading features"):
        features_df = pd.read_csv(FEATURES_FILE)
    log.info(
        f"  Features: {len(features_df):,} rows, cols={features_df.columns.tolist()}"
    )

    return edges_df, features_df
