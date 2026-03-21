"""
Data quality checks and descriptive statistics.
"""

import pandas as pd
from log import log


def check_quality(features_df: pd.DataFrame) -> None:
    """Check for missing values, duplicates, and report data types."""
    # Missing values
    missing = features_df.isnull().sum()
    total_missing = missing.sum()
    if total_missing > 0:
        log.warning(f"  Missing values found: {total_missing}")
        for col in missing[missing > 0].index:
            pct = missing[col] / len(features_df) * 100
            log.warning(f"    {col}: {missing[col]} ({pct:.2f}%)")
    else:
        log.info("  No missing values found")

    # Duplicates
    if "id" in features_df.columns:
        dupes = features_df.duplicated(subset=["id"]).sum()
        log.info(f"  Duplicate node IDs: {dupes}")

    # Data types
    log.info("  Data types:")
    for col in features_df.columns:
        log.info(f"    {col:15s} -> {features_df[col].dtype}")


def describe_features(features_df: pd.DataFrame) -> None:
    """Log descriptive statistics for all feature types."""
    # Overall describe
    log.info("  Overall statistics:")
    for line in features_df.describe().to_string().split("\n"):
        log.info(f"    {line}")

    # Binary features
    for col in ["explicit_content", "dead_account", "affiliate_status"]:
        if col in features_df.columns:
            counts = features_df[col].value_counts()
            pcts = features_df[col].value_counts(normalize=True) * 100
            log.info(
                f"  {col.upper()}: "
                f"0={counts.get(0, 0):,} ({pcts.get(0, 0):.1f}%)  "
                f"1={counts.get(1, 0):,} ({pcts.get(1, 0):.1f}%)"
            )

    # Language distribution
    if "language" in features_df.columns:
        log.info(f"  Languages: {features_df['language'].nunique()} unique")
        for lang, count in features_df["language"].value_counts().head(10).items():
            pct = count / len(features_df) * 100
            log.info(f"    {lang:8s}: {count:>7,} ({pct:5.1f}%)")

    # Numerical features
    for col in ["view_count", "account_lifetime"]:
        if col in features_df.columns:
            s = features_df[col]
            log.info(
                f"  {col.upper():12s}: mean={s.mean():>12,.1f}  "
                f"median={s.median():>10,.1f}  std={s.std():>12,.1f}  "
                f"min={s.min():>8,}  max={s.max():>12,}  skew={s.skew():.2f}"
            )

    # String / timestamp columns
    for col in ["creation_date", "last_stream"]:
        if col in features_df.columns:
            log.info(
                f"  {col.upper()}: dtype={features_df[col].dtype}, "
                f"sample={features_df[col].iloc[0]}, "
                f"unique={features_df[col].nunique():,}"
            )


def summarize(
    features_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    density: float,
) -> None:
    """Log a final summary of the dataset after feature engineering."""
    n_network = sum(
        1
        for c in ["degree", "clustering_coeff", "pagerank", "core_number"]
        if c in features_df.columns
    )
    n_original = len(features_df.columns) - n_network

    log.info(f"Dataset: {len(features_df):,} nodes, {len(edges_df):,} edges")
    log.info(
        f"Total features: {len(features_df.columns)} ({n_original} original + {n_network} network)"
    )

    log.info("Binary classification targets:")
    for col in ["affiliate_status", "explicit_content", "dead_account"]:
        if col in features_df.columns:
            pos = features_df[col].mean() * 100
            log.info(f"  {col}: {pos:.1f}% positive class")

    log.info("Multi-class target:")
    if "language" in features_df.columns:
        log.info(f"  language: {features_df['language'].nunique()} classes")
        for lang, count in features_df["language"].value_counts().head(3).items():
            log.info(f"    {lang}: {count:,} ({count / len(features_df) * 100:.1f}%)")

    log.info("Regression targets:")
    for col in ["view_count", "account_lifetime"]:
        if col in features_df.columns:
            s = features_df[col]
            log.info(
                f"  {col}: mean={s.mean():,.0f}, median={s.median():,.0f}, skew={s.skew():.2f}"
            )

    log.info("Network characteristics:")
    if "degree" in features_df.columns:
        log.info(f"  Avg degree: {features_df['degree'].mean():.2f}")
    if "clustering_coeff" in features_df.columns:
        log.info(f"  Avg clustering: {features_df['clustering_coeff'].mean():.4f}")
    if "core_number" in features_df.columns:
        log.info(f"  Avg core number: {features_df['core_number'].mean():.2f}")
    log.info(f"  Network density: {density:.6f}")
