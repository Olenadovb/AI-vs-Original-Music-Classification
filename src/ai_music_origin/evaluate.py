"""Metrics, cross-validation, and the confusion-matrix figure."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score



def compute_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    labels: list | None = None,
) -> dict[str, Any]:
    """Compute classification metrics for model evaluation."""

    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "macro_f1": float(
            f1_score(
                y_true, y_pred,
                average="macro",
                zero_division=0
            )
        ),
        "macro_precision": float(
            precision_score(
                y_true, y_pred,
                average="macro",
                zero_division=0
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true, y_pred,
                average="macro",
                zero_division=0
            )
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=labels
        ).tolist(),
    }



def cross_validate_accuracy(
    model: BaseEstimator, X: pd.DataFrame, y: pd.Series, n_splits: int, seed: int
) -> dict[str, float]:
    """Report mean +/- std accuracy across ``n_splits`` stratified folds.

    Why this matters: a single train/test split gives you one number, and one number
    can't tell you whether a model is reliably good or just got a lucky split. The
    standard deviation across folds is what tells you how much to trust the mean.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(clone(model), X, y, cv=cv, scoring="accuracy")
    return {"cv_mean_accuracy": float(scores.mean()), "cv_std_accuracy": float(scores.std())}


def plot_confusion_matrix(
    cm: list[list[int]], class_names: list[str], output_path: str | Path
) -> None:
    """Render and save a confusion-matrix heatmap."""
    cm_array = np.array(cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_array, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")
    for i in range(cm_array.shape[0]):
        for j in range(cm_array.shape[1]):
            ax.text(j, i, str(cm_array[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
