from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
)
from scipy.stats import loguniform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

N_JOBS = 1
RANDOM_STATE = 42
# extends a standart classifer to predict the most frequent class, so it is a simple baseline model
class BaselineModel(BaseEstimator, ClassifierMixin):
    delay_feature = "R_dep_avg_DepDelayMinutes_sfd"

    # model that predicts using the 2h_prev_avg_delay binned into the same 4 classes as the target variable, so it is a simple heuristic based on the average delay of the previous 2 hours
    def fit(self, X, y):
        # keep sklearn-compatible attributes
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        # the bins are 0 to 15 minutes, 15 to 60 minutes, 60 to 180 minutes, and more than 180 minutes
        self.class_bins = [0, 15, 60, 180, np.inf]
        # print out the disrtibution of how many flights would get predicted into each class based on the 2h_prev_avg_delay_so_far_day feature, to get an idea of how good this heuristic is
        print(f"Distribution of predicted classes based on the {self.delay_feature} feature:")
        print(pd.cut(pd.to_numeric(X[self.delay_feature], errors="coerce").fillna(0), bins=self.class_bins, labels=[0, 1, 2, 3], include_lowest=True).value_counts())
        return self

    def predict(self, X):
        # predict the class based on the average delay of the previous 2 hours
        delays = pd.to_numeric(X[self.delay_feature], errors="coerce").fillna(0)
        return pd.cut(delays, bins=self.class_bins, labels=[0, 1, 2, 3], include_lowest=True).astype(int)
def make_model(name: str):
    """Build a supported multiclass classifier by name."""
    if name == "dummy":
        return DummyClassifier(strategy="most_frequent")
    elif name =="Baseline":
        return BaselineModel()
    elif name == "logistic_regression":
        return LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS
        )     
    elif name == "logistic_regression_pipeline":
        return Pipeline(
            [
                ("scaler", StandardScaler(with_mean=False)),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=N_JOBS
                    ),
                ),
            ]
        )

    elif name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=10,
            max_depth=16,
            class_weight="balanced_subsample",
            n_jobs=N_JOBS,
            random_state=RANDOM_STATE,
        )
    elif name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=RANDOM_STATE,
        )
    # xgboost
    elif name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            objective="multi:softprob",
            tree_method="hist",
            learning_rate=0.06,
            n_estimators=400,
            max_depth=7,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_lambda=10.0,
            random_state=RANDOM_STATE,
            eval_metric="mlogloss",
            n_jobs=N_JOBS,
        )
    elif name == "naive_bayes":
        from sklearn.naive_bayes import MultinomialNB
        return MultinomialNB()
    elif name in {"svc", "support_vector_classifier"}:
        from sklearn.svm import SVC
        return Pipeline(
            [
                ("scaler", StandardScaler(with_mean=False)),
                ("classifier", SVC(random_state=RANDOM_STATE)),
            ]
        )
    raise ValueError(f"Unsupported model: {name}")

def make_neural_network_model(input_shape: int, num_classes: int,
                               metric: str="accuracy", loss: str="sparse_categorical_crossentropy"):
    """Build a simple feedforward neural network for multiclass classification."""
    from tensorflow import keras
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_shape,)),
            keras.layers.Dense(512, activation="relu"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=loss,
        metrics=[metric,"f1_score"],
    )
    return model

def get_feature_importances(model, feature_names: list[str]):
    # if the model is a tree based model, we can get the feature importances
    # so for random forest and hist gradient boosting and xgboost, we can get the feature importances directly from the model
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    # for logistic regression, we can get the feature importances from the coefficients
    elif hasattr(model, "coef_"):
        # we can normalize the coefficents by taking the absolute value and dividing by the sum of the absolute values
        importances = abs(model.coef_).sum(axis=0) / abs(model.coef_).sum()
    # for neural network, we can look at the weights of the first layer as a proxy for feature importance, but this is not very reliable
    elif hasattr(model, "layers") and hasattr(model, "get_weights"):
        importances = abs(model.layers[0].get_weights()[0]).sum(axis=1) / abs(model.layers[0].get_weights()[0]).sum() 
    else:
        print("Model type not supported for feature importance")
        return pd.Series(dtype=float)
    feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    return feature_importances

def plot_feature_importances(feature_importances: pd.Series, top_n: int = 20):
    if feature_importances.empty:
        print("No feature importances available for this model type")
        return
    plt.figure(figsize=(10, 6))
    feature_importances.head(top_n).plot(kind="barh")
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.title("Top Feature Importances")
    plt.show()


def default_param_distributions(name: str) -> dict[str, list[object]|object]:
    """Small search spaces suitable for a first tuning pass."""
    if name == "logistic_regression":
        return {
            "C": loguniform(10**(-3), 10),
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
            "min_samples_leaf": [20, 50, 100],
            "l2_regularization": [0.0, 0.01, 0.1],
        }
    if name == "xgboost":
        return {
        "learning_rate": [0.02, 0.03, 0.05, 0.06, 0.08, 0.1],
        "n_estimators": [300, 400, 600, 800],
        "max_depth": [4, 5, 6, 7, 8],
        "min_child_weight": [1, 3, 5, 10],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
        "reg_lambda": [1.0, 3.0, 10.0, 30.0],
        "reg_alpha": [0.0, 0.1, 1.0],
        "gamma": [0.0, 0.5, 1.0, 2.0],
        "max_delta_step": [0, 1, 5],
    }
    if name in {"svc", "support_vector_classifier"}:
        return {
            "classifier__C": [0.1, 1.0, 10.0],
            "classifier__kernel": ["rbf", "linear"],
            "classifier__gamma": ["scale", "auto"],
        }
    return {}
