from __future__ import annotations

import pandas as pd


def chronological_train_val_test_split(
    dataframe: pd.DataFrame,
    time_column: str,
    train_size: float = 0.7,
    val_size: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split rows chronologically into train, validation, and test sets."""
    if time_column not in dataframe.columns:
        raise ValueError(f"Missing required time column: {time_column}")
    if not 0 < train_size < 1 or not 0 < val_size < 1:
        raise ValueError("train_size and val_size must be between 0 and 1")
    if train_size + val_size >= 1:
        raise ValueError("train_size + val_size must leave room for a test split")

    sorted_df = dataframe.sort_values(time_column).reset_index(drop=True)
    train_end = int(len(sorted_df) * train_size)
    val_end = int(len(sorted_df) * (train_size + val_size))

    return (
        sorted_df.iloc[:train_end].copy(),
        sorted_df.iloc[train_end:val_end].copy(),
        sorted_df.iloc[val_end:].copy(),
    )
