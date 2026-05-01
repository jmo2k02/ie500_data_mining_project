from __future__ import annotations

import pandas as pd


DELAY_CLASS_ORDER = ["no_delay", "small_delay", "medium_delay", "large_delay"]


def add_delay_class_target(
    dataframe: pd.DataFrame,
    delay_column: str = "DepDelayMinutes",
    target_column: str = "delay_class",
) -> pd.DataFrame:
    """Add ordered delay classes for multiclass departure-delay prediction."""
    if delay_column not in dataframe.columns:
        raise ValueError(f"Missing required delay column: {delay_column}")

    result = dataframe.copy()
    result[target_column] = pd.cut(
        result[delay_column],
        bins=[float("-inf"), 15, 30, 60, float("inf")],
        labels=DELAY_CLASS_ORDER,
        right=True,
    )
    return result.dropna(subset=[target_column])
