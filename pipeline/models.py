from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42


def make_model(name: str):
    """Build a supported multiclass classifier by name."""
    if name == "dummy":
        return DummyClassifier(strategy="most_frequent")
    if name == "logistic_regression":
        return Pipeline(
            [
                ("scaler", StandardScaler(with_mean=False)),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=500,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=200,
            random_state=RANDOM_STATE,
        )
    raise ValueError(f"Unsupported model: {name}")


def default_param_distributions(name: str) -> dict[str, list[object]]:
    """Small search spaces suitable for a first tuning pass."""
    if name == "logistic_regression":
        return {
            "classifier__C": [0.01, 0.1, 1.0, 10.0],
        }
    if name == "random_forest":
        return {
            "n_estimators": [100, 200, 400],
            "max_depth": [None, 12, 24],
            "min_samples_leaf": [5, 10, 25],
            "max_features": ["sqrt", "log2"],
        }
    if name == "hist_gradient_boosting":
        return {
            "learning_rate": [0.03, 0.06, 0.1],
            "max_iter": [100, 200, 400],
            "max_leaf_nodes": [15, 31, 63],
            "l2_regularization": [0.0, 0.01, 0.1],
        }
    return {}
