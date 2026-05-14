from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
import inspect
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import uniform
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, fbeta_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

from evaluate import (
    classification_report_frame,
    evaluate_classifier,
    plotConfusionMatrix,
)
from features import fit_transform, transform
from loader import LoaderStorage
from models import (
    default_param_distributions,
    make_model,
    plot_feature_importances,
    get_feature_importances,
)
from split import chronological_train_val_test_split
from targets import DELAY_CLASS_ORDER, add_delay_class_target


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
    weights: dict[int, float] | None | str = "balanced"
    tune: bool = False
    n_iter: int = 10
    cv_splits: int = 3
    mlflow_experiment: str | None = None


CLASS_WEIGHT_PARAM_DISTRIBUTIONS = {
    "class_weight_0": uniform(0.1, 0.4),
    "class_weight_1": uniform(0.5, 0.5),
    "class_weight_2": uniform(1.0, 1.0),
    "class_weight_3": uniform(1.0, 3.0),
}


class ClassWeightTunedClassifier(BaseEstimator, ClassifierMixin):
    """Expose per-class sample weights as RandomizedSearchCV parameters."""

    def __init__(
        self,
        estimator,
        class_weight_0: float = 0.3,
        class_weight_1: float = 0.7,
        class_weight_2: float = 1.0,
        class_weight_3: float = 1.5,
    ):
        self.estimator = estimator
        self.class_weight_0 = class_weight_0
        self.class_weight_1 = class_weight_1
        self.class_weight_2 = class_weight_2
        self.class_weight_3 = class_weight_3

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator)
        _disable_builtin_class_weight(self.estimator_)
        self.class_weight_ = {
            0: self.class_weight_0,
            1: self.class_weight_1,
            2: self.class_weight_2,
            3: self.class_weight_3,
        }
        sample_weight = compute_sample_weight(class_weight=self.class_weight_, y=y)
        _fit_estimator_with_sample_weight(self.estimator_, X, y, sample_weight)
        self.classes_ = getattr(self.estimator_, "classes_", np.unique(y))
        self.n_features_in_ = getattr(self.estimator_, "n_features_in_", X.shape[1])
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    @property
    def feature_importances_(self):
        return self.estimator_.feature_importances_


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
    train_df = train_df.drop(columns=[config.time_column])
    val_df = val_df.drop(columns=[config.time_column])
    test_df = test_df.drop(columns=[config.time_column])
    x_train, y_train = fit_transform(train_df)
    x_val, y_val = transform(val_df)
    x_test, y_test = transform(test_df)

    weights = _resolve_class_weights(config.weights, y_train)
    sample_weights = compute_sample_weight(class_weight=weights, y=y_train) if weights else None

    model = make_model(config.model_name)
    if config.tune:
        model = _tune_model(
            model,
            config,
            x_train,
            y_train,
        )
    else:
        _fit_estimator_with_sample_weight(model, x_train, y_train, sample_weights)

    val_metrics, val_predictions = evaluate_classifier(model, x_val, y_val)
    test_metrics, test_predictions = evaluate_classifier(model, x_test, y_test)
    _persist_outputs(
        config,
        model,
        val_metrics,
        test_metrics,
        val_predictions,
        test_predictions,
        x_train.columns,
        x_val,
        y_val,
        x_test,
        y_test,
        train_rows=len(train_df),
        val_rows=len(val_df),
        test_rows=len(test_df),
        weights=weights,
    )
    _log_mlflow(config, val_metrics, test_metrics)
    plotConfusionMatrix(predictions=val_predictions, y_true=y_val)
    feature_importances = get_feature_importances(model, feature_names=x_train.columns)
    if not feature_importances.empty:
        plot_feature_importances(feature_importances, top_n=30)
    # if this was hyperparamter tuning, return the best hyperparameters
    if config.tune:
        return {f"val_{key}": value for key, value in val_metrics.items()} | {
            f"test_{key}": value for key, value in test_metrics.items()
        }, getattr(model, "_tuning_best_params", model.get_params())
    return {f"val_{key}": value for key, value in val_metrics.items()} | {
        f"test_{key}": value for key, value in test_metrics.items()
    }, None


def _load_dataframe(storage: LoaderStorage, input_path: str) -> pd.DataFrame:
    if input_path.endswith(".parquet"):
        return storage.read_parquet(input_path)
    if input_path.endswith(".csv"):
        return storage.read_csv(input_path)
    raise ValueError("Only .parquet and .csv inputs are supported")


def _tune_model(
    model,
    config: TrainingConfig,
    x_train: pd.DataFrame,
    y_train: pd.Series,
):
    model = ClassWeightTunedClassifier(model)
    param_distributions = _tuning_param_distributions(config.model_name)

    search = RandomizedSearchCV(
        model,
        param_distributions=param_distributions,
        n_iter=config.n_iter,
        scoring=make_scorer(fbeta_score, beta=2, average="macro", zero_division=0),
        cv=TimeSeriesSplit(n_splits=config.cv_splits),
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )
    search.fit(x_train, y_train)
    best_model = search.best_estimator_
    best_model._tuning_best_params = search.best_params_
    best_model._tuning_best_score = search.best_score_
    best_model._tuning_cv_results = pd.DataFrame(search.cv_results_)
    return best_model


def _tuning_param_distributions(model_name: str) -> dict[str, object]:
    model_param_distributions = {
        f"estimator__{name}": values
        for name, values in default_param_distributions(model_name).items()
    }
    return model_param_distributions | CLASS_WEIGHT_PARAM_DISTRIBUTIONS


def _resolve_class_weights(weights, y_train: pd.Series):
    if isinstance(weights, str):
        class_labels = np.unique(y_train)
        class_weights = compute_class_weight(
            class_weight=weights,
            classes=class_labels,
            y=y_train,
        )
        return dict(zip(class_labels, class_weights))
    if isinstance(weights, dict):
        return weights
    if weights is None:
        return {0: 0.3, 1: 0.70, 2: 1.0, 3: 1.5}
    raise ValueError("weights must be either a dict, None, or the string 'balanced'")


def _fit_estimator_with_sample_weight(model, x_train, y_train, sample_weight):
    if sample_weight is None:
        model.fit(x_train, y_train)
        return model

    if isinstance(model, Pipeline):
        final_step_name, final_estimator = model.steps[-1]
        if "sample_weight" in inspect.signature(final_estimator.fit).parameters:
            model.fit(x_train, y_train, **{f"{final_step_name}__sample_weight": sample_weight})
        else:
            model.fit(x_train, y_train)
        return model

    if "sample_weight" in inspect.signature(model.fit).parameters:
        model.fit(x_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(x_train, y_train)
    return model


def _disable_builtin_class_weight(model) -> None:
    """Avoid multiplying tuned sample weights by estimator class_weight settings."""
    if isinstance(model, Pipeline):
        final_step_name, final_estimator = model.steps[-1]
        if "class_weight" in final_estimator.get_params():
            model.set_params(**{f"{final_step_name}__class_weight": None})
        return

    if hasattr(model, "get_params") and "class_weight" in model.get_params():
        model.set_params(class_weight=None)


def _fit_params_for_model(model, y_train: pd.Series) -> dict[str, object]:
    """Return sample-weight fit params when the estimator does not already balance classes."""
    if _uses_builtin_class_weight(model):
        return {}

    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    if isinstance(model, Pipeline):
        final_step_name, final_estimator = model.steps[-1]
        if "sample_weight" in inspect.signature(final_estimator.fit).parameters:
            return {f"{final_step_name}__sample_weight": sample_weight}
        return {}

    if "sample_weight" in inspect.signature(model.fit).parameters:
        return {"sample_weight": sample_weight}

    return {}


def _uses_builtin_class_weight(model) -> bool:
    if isinstance(model, Pipeline):
        final_estimator = model.steps[-1][1]
        return getattr(final_estimator, "class_weight", None) is not None

    return getattr(model, "class_weight", None) is not None


def _persist_outputs(
    config: TrainingConfig,
    model,
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
    val_predictions,
    test_predictions,
    feature_columns,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    train_rows: int,
    val_rows: int,
    test_rows: int,
    weights,
) -> None:
    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{run_timestamp}_{config.model_name}"
    run_dir = Path(config.output_dir) / "runs" / run_id
    images_dir = run_dir / "images"
    run_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    model_path = run_dir / "model.joblib"
    run_info_path = run_dir / "run_info.json"
    features_path = run_dir / "features.json"
    validation_report_path = run_dir / "classification_report_validation.csv"
    test_report_path = run_dir / "classification_report_test.csv"

    joblib.dump(model, model_path)
    features_path.write_text(json.dumps(list(feature_columns), indent=2))

    validation_report = classification_report_frame(model, x_val, y_val)
    test_report = classification_report_frame(model, x_test, y_test)
    validation_report.to_csv(validation_report_path)
    test_report.to_csv(test_report_path)

    saved_images = {}
    _save_confusion_matrix_image(
        val_predictions,
        y_val,
        images_dir / "validation_confusion_matrix.png",
        "Validation Confusion Matrix",
    )
    saved_images["validation_confusion_matrix"] = "images/validation_confusion_matrix.png"
    _save_confusion_matrix_image(
        test_predictions,
        y_test,
        images_dir / "test_confusion_matrix.png",
        "Test Confusion Matrix",
    )
    saved_images["test_confusion_matrix"] = "images/test_confusion_matrix.png"

    feature_importances = get_feature_importances(model, feature_names=feature_columns)
    if not feature_importances.empty:
        _save_feature_importance_image(
            feature_importances,
            images_dir / "feature_importances_top_30.png",
            top_n=30,
        )
        saved_images["feature_importances_top_30"] = "images/feature_importances_top_30.png"

    cv_results_path = None
    cv_results = getattr(model, "_tuning_cv_results", None)
    if cv_results is not None:
        cv_results_path = run_dir / "tuning_cv_results.csv"
        cv_results.to_csv(cv_results_path, index=False)

    run_info = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data": {
            "data_root": config.data_root,
            "input_path": config.input_path,
            "sample_frac": config.sample_frac,
            "target_column": config.target_column,
            "delay_column": config.delay_column,
            "time_column": config.time_column,
        },
        "splits": {
            "train_rows": train_rows,
            "validation_rows": val_rows,
            "test_rows": test_rows,
        },
        "features": {
            "count": len(feature_columns),
            "names": list(feature_columns),
        },
        "target_classes": list(DELAY_CLASS_ORDER),
        "artifacts": {
            "model_path": "model.joblib",
            "run_info_path": "run_info.json",
            "features_path": "features.json",
            "validation_report_path": "classification_report_validation.csv",
            "test_report_path": "classification_report_test.csv",
            "tuning_cv_results_path": "tuning_cv_results.csv" if cv_results_path else None,
            "images": saved_images,
        },
        "model": {
            "name": config.model_name,
            "class_name": type(model).__name__,
            "parameters": model.get_params(deep=False) if hasattr(model, "get_params") else {},
            "class_weights": getattr(model, "class_weight_", weights),
        },
        "tuning": {
            "enabled": config.tune,
            "scoring": "fbeta_macro_beta_2" if config.tune else None,
            "n_iter": config.n_iter if config.tune else None,
            "cv_splits": config.cv_splits if config.tune else None,
            "class_weight_intervals": {
                "0": [0.1, 0.5],
                "1": [0.5, 1.0],
                "2": [1.0, 2.0],
                "3": [1.0, 4.0],
            } if config.tune else None,
            "best_params": getattr(model, "_tuning_best_params", None),
            "best_score": getattr(model, "_tuning_best_score", None),
        },
        "metrics": {
            "validation": val_metrics,
            "test": test_metrics,
            "validation_classification_report": validation_report,
            "test_classification_report": test_report,
        },
        "feature_importances_top_30": feature_importances.head(30),
    }
    run_info_path.write_text(json.dumps(_make_json_safe(run_info), indent=2))


def _make_json_safe(value):
    if isinstance(value, dict):
        return {str(key): _make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_make_json_safe(item) for item in value]
    if isinstance(value, pd.Series):
        return _make_json_safe(value.to_dict())
    if isinstance(value, pd.DataFrame):
        return _make_json_safe(value.to_dict(orient="index"))
    if isinstance(value, np.ndarray):
        return _make_json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _save_confusion_matrix_image(predictions, y_true, path: Path, title: str) -> None:
    cm = confusion_matrix(y_true=y_true, y_pred=predictions, labels=range(len(DELAY_CLASS_ORDER)))
    fig, ax = plt.subplots(figsize=(8, 6))
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=DELAY_CLASS_ORDER,
    )
    display.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_feature_importance_image(
    feature_importances: pd.Series,
    path: Path,
    top_n: int = 30,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    feature_importances.head(top_n).plot(kind="barh", ax=ax)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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

# TODO - > Train Test split -> 
#   TRAIN: 2014 -> 2018 
#   TEST: 2019
# TODO - > Add these weights
weights = [
    # 0 -> [0.1 - 0.5]
    # 1 -> [0.5 - 1]
    # 2 -> [1 - 2]
    # 3 -> [1 - 4]
    # 10 random executions
    {0: 0.3, 1: 0.70, 2: 1.0, 3: 1.5},

    {0: 0.3, 1: 0.70, 2: 1.0, 3: 1.5},

    {0: 0.3, 1: 0.70, 2: 1.0, 3: 1.5},

    {0: 0.3, 1: 0.70, 2: 1.0, 3: 1.5},
]
