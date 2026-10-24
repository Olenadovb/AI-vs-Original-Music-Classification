
"""Reproducibility tests for music classification models."""

import numpy as np
import pytest

from sklearn.datasets import make_classification

from ai_music_origin.config import load_config
from ai_music_origin.models import (
    build_logistic_regression,
    build_svm,
    build_knn,
)
from ai_music_origin.evaluate import compute_metrics


@pytest.fixture
def sample_data():
    """Generate the same synthetic dataset for each test."""
    X, y = make_classification(
        n_samples=150,
        n_features=20,
        n_informative=15,
        n_classes=5,
        random_state=42,
    )

    return X[:100], X[100:], y[:100], y[100:]


@pytest.fixture
def config():
    return load_config("config/default.yaml")


@pytest.mark.parametrize("builder", [
    build_logistic_regression,
    build_svm,
    build_knn,
])
def test_same_seed_gives_identical_predictions(
    builder, config, sample_data
):
    """Two runs with the same configuration must agree."""
    X_train, X_val, y_train, y_val = sample_data

    model_1 = builder(config)
    model_2 = builder(config)

    model_1.fit(X_train, y_train)
    model_2.fit(X_train, y_train)

    pred_1 = model_1.predict(X_val)
    pred_2 = model_2.predict(X_val)

    np.testing.assert_array_equal(pred_1, pred_2)

    metrics_1 = compute_metrics(y_val, pred_1)
    metrics_2 = compute_metrics(y_val, pred_2)

    assert metrics_1 == metrics_2


@pytest.mark.parametrize("builder", [
    build_logistic_regression,
    build_svm,
    build_knn,
])
def test_repeated_training_is_stable(builder, config, sample_data):
    """Repeated training must produce consistent results."""
    X_train, X_val, y_train, _ = sample_data

    results = []

    for _ in range(3):
        model = builder(config)
        model.fit(X_train, y_train)

        results.append(model.predict(X_val))

    for predictions in results[1:]:
        np.testing.assert_array_equal(
            results[0],
            predictions
        )
