
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def calculate_metrics(y_true, y_pred):
    """Calculate standard binary classification metrics."""
    return {
        "accuracy": round(
            float(accuracy_score(y_true, y_pred)), 4
        ),
        "precision": round(
            float(precision_score(
                y_true, y_pred, zero_division=0
            )), 4
        ),
        "recall": round(
            float(recall_score(
                y_true, y_pred, zero_division=0
            )), 4
        ),
        "f1_score": round(
            float(f1_score(
                y_true, y_pred, zero_division=0
            )), 4
        )
    }
