from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report

from targets import DELAY_CLASS_ORDER


def evaluate_classifier(model, x, y) -> dict[str, float]:
    """Compute headline multiclass metrics."""
    predictions = model.predict(x)
    report = classification_report(
        y,
        predictions,
        labels=DELAY_CLASS_ORDER,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": accuracy_score(y, predictions),
        "balanced_accuracy": balanced_accuracy_score(y, predictions),
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "large_delay_recall": report["large_delay"]["recall"],
    }


def classification_report_frame(model, x, y) -> pd.DataFrame:
    """Return the full per-class classification report as a dataframe."""
    predictions = model.predict(x)
    report = classification_report(
        y,
        predictions,
        labels=DELAY_CLASS_ORDER,
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(report).transpose()
