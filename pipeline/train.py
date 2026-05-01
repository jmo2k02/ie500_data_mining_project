from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from evaluate import classification_report_frame, evaluate_classifier
from features import build_feature_matrix
from loader import LoaderStorage
from models import default_param_distributions, make_model
from split import chronological_train_val_test_split
from targets import add_delay_class_target


@dataclass(frozen=True)
class TrainingConfig:
    data_root: str
    input_path: str
    output_dir: str = "outputs/training"
    model_name: str = "hist_gradient_boosting"
    delay_column: str = "DepDelayMinutes"
    target_column: str = "delay_class"
    time_column: str = "CRSDepDateTime_UTC"
    sample_frac: float = 1.0
    tune: bool = False
    n_iter: int = 20
    cv_splits: int = 3
    mlflow_experiment: str | None = None


def run_training(config: TrainingConfig) -> dict[str, float]:
    """Train, evaluate, and persist a multiclass delay classifier."""
    storage = LoaderStorage(config.data_root)
    dataframe = _load_dataframe(storage, config.input_path)
    if config.sample_frac < 1.0:
        dataframe = dataframe.sample(frac=config.sample_frac, random_state=42)

    dataframe = add_delay_class_target(
        dataframe,
        delay_column=config.delay_column,
        target_column=config.target_column,
    )
    train_df, val_df, test_df = chronological_train_val_test_split(
        dataframe,
        time_column=config.time_column,
    )

    x_train, y_train = build_feature_matrix(
        train_df,
        target_column=config.target_column,
        time_column=config.time_column,
    )
    x_val, y_val = build_feature_matrix(
        val_df,
        target_column=config.target_column,
        time_column=config.time_column,
    )
    x_test, y_test = build_feature_matrix(
        test_df,
        target_column=config.target_column,
        time_column=config.time_column,
    )
    x_val = x_val.reindex(columns=x_train.columns, fill_value=0)
    x_test = x_test.reindex(columns=x_train.columns, fill_value=0)

    model = make_model(config.model_name)
    if config.tune:
        model = _tune_model(model, config, x_train, y_train)
    else:
        model.fit(x_train, y_train)

    val_metrics = evaluate_classifier(model, x_val, y_val)
    test_metrics = evaluate_classifier(model, x_test, y_test)
    _persist_outputs(config, model, val_metrics, test_metrics, x_train.columns, x_test, y_test)
    _log_mlflow(config, val_metrics, test_metrics)
    return {f"val_{key}": value for key, value in val_metrics.items()} | {
        f"test_{key}": value for key, value in test_metrics.items()
    }


def _load_dataframe(storage: LoaderStorage, input_path: str) -> pd.DataFrame:
    if input_path.endswith(".parquet"):
        return storage.read_parquet(input_path)
    if input_path.endswith(".csv"):
        return storage.read_csv(input_path)
    raise ValueError("Only .parquet and .csv inputs are supported")


def _tune_model(model, config: TrainingConfig, x_train: pd.DataFrame, y_train: pd.Series):
    param_distributions = default_param_distributions(config.model_name)
    if not param_distributions:
        model.fit(x_train, y_train)
        return model

    search = RandomizedSearchCV(
        model,
        param_distributions=param_distributions,
        n_iter=config.n_iter,
        scoring="f1_macro",
        cv=TimeSeriesSplit(n_splits=config.cv_splits),
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )
    search.fit(x_train, y_train)
    return search.best_estimator_


def _persist_outputs(
    config: TrainingConfig,
    model,
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
    feature_columns,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, output_dir / f"{config.model_name}.joblib")
    (output_dir / "metrics.json").write_text(
        json.dumps({"validation": val_metrics, "test": test_metrics}, indent=2)
    )
    (output_dir / "features.json").write_text(json.dumps(list(feature_columns), indent=2))
    classification_report_frame(model, x_test, y_test).to_csv(
        output_dir / "classification_report_test.csv"
    )


def _log_mlflow(
    config: TrainingConfig,
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
) -> None:
    if not config.mlflow_experiment:
        return

    import mlflow

    mlflow.set_experiment(config.mlflow_experiment)
    with mlflow.start_run(run_name=config.model_name):
        mlflow.log_params(
            {
                "model_name": config.model_name,
                "input_path": config.input_path,
                "delay_column": config.delay_column,
                "target_column": config.target_column,
                "time_column": config.time_column,
                "sample_frac": config.sample_frac,
                "tune": config.tune,
            }
        )
        for key, value in val_metrics.items():
            mlflow.log_metric(f"val_{key}", value)
        for key, value in test_metrics.items():
            mlflow.log_metric(f"test_{key}", value)
