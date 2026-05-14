from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score, fbeta_score,confusion_matrix

from targets import DELAY_CLASS_ORDER
import seaborn as sns
import matplotlib.pyplot as plt

def f1_macro_score(model, x, y) -> float:
    """Compute the f1-macro score."""
    predictions = model.predict(x)
    # chech the shape of predictions, if it is (n_samples, num_classes), take the argmax to get the predicted class labels
    if predictions.ndim == 2 and predictions.shape[1] > 1:
        predictions = predictions.argmax(axis=1)
    f1_macro = f1_score(y, predictions, average="macro", zero_division=0)
    return f1_macro

def f2_macro_score(model, x, y) -> float:
    """Compute the F-beta macro score with beta=2."""
    predictions = model.predict(x)
    if predictions.ndim == 2 and predictions.shape[1] > 1:
        predictions = predictions.argmax(axis=1)
    return fbeta_score(y, predictions, beta=2, average="macro", zero_division=0)

def evaluate_classifier(model, x, y) -> dict[str, float]:
    """Compute headline multiclass metrics."""
    predictions = model.predict(x)
    # chech the shape of predictions, if it is (n_samples, num_classes), take the argmax to get the predicted class labels
    if predictions.ndim == 2 and predictions.shape[1] > 1:
        predictions = predictions.argmax(axis=1)
    report = classification_report(
        y,
        predictions,
        labels=range(len(DELAY_CLASS_ORDER)),
        target_names=DELAY_CLASS_ORDER,
        output_dict=True,
        zero_division=0,
    )
    f2_macro = fbeta_score(y, predictions, beta=2, average="macro", zero_division=0)
    # print the F2-macro score, which weights recall higher than precision
    print(f"F1-macro: {report['macro avg']['f1-score']:.4f}")
    print(f"F2-macro: {f2_macro:.4f}")
    return {
    "accuracy": accuracy_score(y, predictions),
    "balanced_accuracy": balanced_accuracy_score(y, predictions),
    "macro_f1": report["macro avg"]["f1-score"],
    "macro_f2": f2_macro,
    "weighted_f1": report["weighted avg"]["f1-score"],
    "small_delay_precision": report["small_delay"]["precision"],
    "small_delay_recall": report["small_delay"]["recall"],
    "medium_delay_precision": report["medium_delay"]["precision"],
    "medium_delay_recall": report["medium_delay"]["recall"],
    "large_delay_precision": report["large_delay"]["precision"],
    "large_delay_recall": report["large_delay"]["recall"],
    "no_delay_precision": report["no_delay"]["precision"],
"no_delay_recall": report["no_delay"]["recall"],
}, predictions


def classification_report_frame(model, x, y) -> pd.DataFrame:
    """Return the full per-class classification report as a dataframe."""
    predictions = model.predict(x)
    report = classification_report(
        y,
        predictions,
        labels=range(len(DELAY_CLASS_ORDER)),
        target_names=DELAY_CLASS_ORDER,
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(report).transpose()

 
def plotConfusionMatrix(predictions, y_true, classes=DELAY_CLASS_ORDER):
    cm = confusion_matrix(y_true=y_true, y_pred=predictions, labels=range(len(classes)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()
