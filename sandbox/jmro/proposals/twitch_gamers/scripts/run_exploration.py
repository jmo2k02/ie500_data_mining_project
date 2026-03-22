"""
Twitch Gamers Dataset - Exploration Pipeline (orchestrator).

Run with:
    uv run python twitch_gamers/scripts/run_exploration.py

Steps:
    1. Load data
    2. Data quality checks
    3. Descriptive statistics
    4. Basic visualizations
    5. Network analysis + feature engineering (scipy.sparse)
    6. Network feature visualizations
    7. Save enhanced dataset
    8. Summary
"""

import os

from log import log, section, total_elapsed
from load import load_data, DATA_DIR, RESULTS_DIR
from stats import check_quality, describe_features, summarize
from network import run_all as run_network
from plots import plot_all_basic, plot_all_network


def main() -> None:
    # ── 1. Load ───────────────────────────────────────────────────────
    section(1, "LOADING DATA")
    edges_df, features_df = load_data()

    # ── 2. Quality ────────────────────────────────────────────────────
    section(2, "DATA QUALITY CHECKS")
    check_quality(features_df)

    # ── 3. Statistics ─────────────────────────────────────────────────
    section(3, "DESCRIPTIVE STATISTICS")
    describe_features(features_df)

    # ── 4. Basic visualizations ───────────────────────────────────────
    section(4, "BASIC VISUALIZATIONS")
    plot_all_basic(features_df, RESULTS_DIR)

    # ── 5. Network analysis + feature engineering ─────────────────────
    section(5, "NETWORK ANALYSIS & FEATURE ENGINEERING")
    net_result = run_network(edges_df, features_df)

    # ── 6. Network feature visualizations ─────────────────────────────
    section(6, "NETWORK FEATURE VISUALIZATIONS")
    plot_all_network(features_df, RESULTS_DIR)

    # ── 7. Save ───────────────────────────────────────────────────────
    section(7, "SAVING ENHANCED DATASET")
    output_file = os.path.join(DATA_DIR, "enhanced_features.csv")
    features_df.to_csv(output_file, index=False)
    log.info(f"  Path:    {output_file}")
    log.info(f"  Shape:   {features_df.shape}")
    log.info(f"  Columns: {features_df.columns.tolist()}")

    # ── 8. Summary ────────────────────────────────────────────────────
    section(8, "SUMMARY")
    summarize(features_df, edges_df, net_result.density)

    log.info(f"Visualizations saved to: {RESULTS_DIR}")
    log.info(f"Enhanced dataset saved to: {output_file}")
    log.info(f"Total runtime: {total_elapsed()}")
    log.info("Done!")


if __name__ == "__main__":
    main()
