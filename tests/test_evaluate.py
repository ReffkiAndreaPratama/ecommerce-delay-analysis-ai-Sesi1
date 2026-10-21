from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    augment_with_predictions,
    build_summary,
    compute_binary_metrics,
    confusion_frame,
    error_breakdown,
    performance_drop,
)


def test_perfect_predictions():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = compute_binary_metrics(y, p, threshold=0.5)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["tp"] == 2 and metrics["tn"] == 2
    assert metrics["fp"] == 0 and metrics["fn"] == 0


def test_all_wrong_predictions():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.9, 0.8, 0.2, 0.1])
    metrics = compute_binary_metrics(y, p, threshold=0.5)
    assert metrics["f1"] == 0.0
    assert metrics["fp"] == 2 and metrics["fn"] == 2


def test_metric_input_validation():
    with pytest.raises(ValueError):
        compute_binary_metrics([], [])
    with pytest.raises(ValueError):
        compute_binary_metrics([0, 1], [0.2])
    with pytest.raises(ValueError):
        compute_binary_metrics([0, 1], [0.2, float("nan")])
    with pytest.raises(ValueError):
        compute_binary_metrics([0, 1], [0.2, 0.8], threshold=1.5)
    with pytest.raises(ValueError):
        compute_binary_metrics([0, 2], [0.2, 0.8])


def test_confusion_frame_shape():
    metrics = compute_binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    frame = confusion_frame(metrics)
    assert frame.shape == (2, 2)
    assert frame.iloc[1, 1] == 2


def test_augment_and_error_breakdown():
    df = pd.DataFrame(
        {
            "is_late": [0, 1, 1, 0],
            "customer_state": ["SP", "SP", "BA", "BA"],
            "order_id": ["1", "2", "3", "4"],
        }
    )
    pred = augment_with_predictions(df, np.array([0.1, 0.4, 0.9, 0.8]), threshold=0.5)
    assert pred["error_type"].tolist() == ["correct", "false_negative", "correct", "false_positive"]
    table = error_breakdown(pred, by="customer_state")
    sp = table[table["customer_state"] == "SP"].iloc[0]
    assert sp["false_negatives"] == 1
    ba = table[table["customer_state"] == "BA"].iloc[0]
    assert ba["false_positives"] == 1


def test_performance_drop_matches_configurations():
    results = pd.DataFrame(
        [
            {"feature_set": "safe", "model": "logistic_regression", "strategy": "none", "split": "random", "pr_auc": 0.50},
            {"feature_set": "safe", "model": "logistic_regression", "strategy": "none", "split": "temporal", "pr_auc": 0.40},
            {"feature_set": "safe", "model": "random_forest", "strategy": "none", "split": "random", "pr_auc": 0.60},
            {"feature_set": "safe", "model": "random_forest", "strategy": "none", "split": "temporal", "pr_auc": 0.45},
        ]
    )
    drops = performance_drop(results)
    assert len(drops) == 2
    assert drops.iloc[0]["absolute_drop"] == pytest.approx(0.15)
    assert "relative_drop_pct" in drops.columns


def test_build_summary_is_derived_from_results():
    results = pd.DataFrame(
        [
            {"feature_set": "safe", "model": "logistic_regression", "strategy": "none", "split": "random",
             "pr_auc": 0.5, "f1": 0.4, "val_pr_auc": 0.55, "val_f1": 0.45, "threshold": 0.3},
            {"feature_set": "safe", "model": "logistic_regression", "strategy": "none", "split": "temporal",
             "pr_auc": 0.4, "f1": 0.35, "val_pr_auc": 0.45, "val_f1": 0.4, "threshold": 0.3},
            {"feature_set": "leaky", "model": "logistic_regression", "strategy": "none", "split": "random",
             "pr_auc": 0.9, "f1": 0.8, "val_pr_auc": 0.9, "val_f1": 0.8, "threshold": 0.3},
            {"feature_set": "leaky", "model": "logistic_regression", "strategy": "none", "split": "temporal",
             "pr_auc": 0.85, "f1": 0.8, "val_pr_auc": 0.85, "val_f1": 0.8, "threshold": 0.3},
        ]
    )
    summary = build_summary(results)
    assert summary["leakage_inflation"]["absolute_difference"] == pytest.approx(0.425, abs=1e-6)
    assert summary["best_safe_by_validation_pr_auc"]["model"] == "logistic_regression"
