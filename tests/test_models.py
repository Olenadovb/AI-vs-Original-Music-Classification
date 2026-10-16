
"""Tests for the music classification baseline models."""

import numpy as np
import pytest

from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ai_music_origin.config import load_config
from ai_music_origin.models import (
    build_baseline,
    build_logistic_regression,
    build_svm,
    build_knn,
)
from ai_music_origin.evaluate import compute_metrics


@pytest.fixture
def sample_data():
    """Create a small synthetic classification dataset."""
    X, y = make_classification(
        n_samples=100,
        n_features=20,
        n_informative=15,
        n_classes=5,
        random_state=42,
    )
    return X, y


@pytest.fixture
def config():
    return load_config("config/default.yaml")


@pytest.mark.parametrize("builder", [
    build_logistic_regression,
    build_svm,
    build_knn,
])
def test_model_creation(builder, config):
    """Check that each model has a scaling pipeline."""
    model = builder(config)

    assert isinstance(model, Pipeline)
    assert isinstance(model.named_steps["scaler"], StandardScaler)
    assert "clf" in model.named_steps


@pytest.mark.parametrize("builder", [
    build_logistic_regression,
    build_svm,
    build_knn,
])
def test_model_training_and_prediction(builder, config, sample_data):
    """Check that models can train and predict."""
    X, y = sample_data

    model = builder(config)
    model.fit(X[:80], y[:80])

    predictions = model.predict(X[80:])

    assert predictions.shape == (20,)
    assert set(predictions).issubset(set(y[:80]))


def test_dummy_classifier(sample_data):
    """Check that the majority-class baseline works."""
    X, y = sample_data

    model = build_baseline()
    model.fit(X[:80], y[:80])

    predictions = model.predict(X[80:])

    assert predictions.shape == (20,)


def test_evaluation_metrics():
    """Check evaluation metrics on known labels."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 2, 2])

    metrics = compute_metrics(y_true, y_pred)

    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["balanced_accuracy"] == pytest.approx(2 / 3)
    assert metrics["macro_f1"] == pytest.approx(2 / 3)
    assert metrics["macro_precision"] == pytest.approx(2 / 3)
    assert metrics["macro_recall"] == pytest.approx(2 / 3)

    assert np.array(metrics["confusion_matrix"]).shape == (3, 3)
