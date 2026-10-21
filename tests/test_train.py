from __future__ import annotations

import numpy as np
import pytest

from src.features import build_supervised_frame, feature_matrix
from src.preprocessing import split_random
from src.train import (
    build_model,
    class_weight_report,
    fit_model,
    predict_proba,
    select_threshold,
    threshold_curve,
)


def test_build_model_rejects_unknown_options():
    with pytest.raises(ValueError):
        build_model("xgboost")
    with pytest.raises(ValueError):
        build_model("logistic_regression", strategy="smote")


def test_logistic_pipeline_predicts_probabilities(demo_df):
    supervised = build_supervised_frame(demo_df)
    train, val, _ = split_random(supervised)
    X_train, y_train = feature_matrix(train, "safe")
    X_val, _ = feature_matrix(val, "safe")
    pipeline = build_model("logistic_regression", strategy="class_weight")
    fitted, seconds = fit_model(pipeline, X_train, y_train)
    assert seconds >= 0
    proba = predict_proba(fitted, X_val)
    assert proba.shape == (len(val),)
    assert np.isfinite(proba).all()
    assert proba.min() >= 0 and proba.max() <= 1


def test_threshold_selection_never_needs_test_data():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.10, 0.20, 0.80, 0.90])
    result = select_threshold(y, p)
    assert result["f1"] == pytest.approx(1.0)
    assert 0.0 <= result["threshold"] <= 1.0


def test_threshold_selection_requires_two_classes():
    with pytest.raises(ValueError):
        select_threshold(np.zeros(5), np.linspace(0, 1, 5))


def test_threshold_curve_is_monotone_in_predicted_rate():
    curve = threshold_curve(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.6, 0.9]))
    rates = [row["predicted_positive_rate"] for row in curve]
    assert all(a >= b for a, b in zip(rates, rates[1:]))


def test_fit_model_rejects_single_class(demo_df):
    train, _, _ = split_random(demo_df)
    pipeline = build_model("logistic_regression")
    y = np.zeros(len(train), dtype=int)
    with pytest.raises(ValueError):
        fit_model(pipeline, train.drop(columns=["is_late"]), y)


def test_class_weight_report(demo_df):
    report = class_weight_report(demo_df["is_late"])
    assert report["n_train"] == len(demo_df)
    assert report["positive_rate"] == pytest.approx(float(demo_df["is_late"].mean()), abs=1e-4)
    assert report["imbalance_ratio"] > 1
