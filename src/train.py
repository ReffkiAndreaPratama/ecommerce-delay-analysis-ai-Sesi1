"""Model construction and validation-only threshold selection."""

from __future__ import annotations

import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE, SUPPORTED_MODELS, SUPPORTED_STRATEGIES, TARGET_COL
from src.features import feature_columns
from src.preprocessing import make_preprocessor


def build_model(
    model_name: str,
    strategy: str = "none",
    feature_set: str = "safe",
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Build an unfitted preprocessing + classifier pipeline.

    ``strategy`` is either ``none`` or ``class_weight``. No resampling is
    applied, so no synthetic observations can leak across partitions.
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"model_name must be one of {SUPPORTED_MODELS}")
    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(f"strategy must be one of {SUPPORTED_STRATEGIES}")

    numeric, categorical = feature_columns(feature_set)
    preprocessor = make_preprocessor(numeric, categorical)

    if model_name == "logistic_regression":
        classifier = LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            class_weight="balanced" if strategy == "class_weight" else None,
            random_state=random_state,
        )
    else:
        classifier = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced_subsample" if strategy == "class_weight" else None,
            n_jobs=-1,
            random_state=random_state,
        )

    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def fit_model(pipeline: Pipeline, X_train, y_train) -> tuple[Pipeline, float]:
    """Fit on the training partition only and return elapsed seconds."""
    if len(X_train) == 0:
        raise ValueError("Training partition is empty")
    if np.asarray(y_train).min() == np.asarray(y_train).max():
        raise ValueError("Training target has a single class")
    start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    return pipeline, float(time.perf_counter() - start)


def predict_proba(pipeline: Pipeline, X) -> np.ndarray:
    """Return P(late) with a stable shape."""
    proba = pipeline.predict_proba(X)
    classes = list(pipeline.classes_)
    if 1 not in classes:
        raise ValueError("Fitted pipeline has no positive class")
    return proba[:, classes.index(1)]


def select_threshold(
    y_true,
    y_prob,
    grid: np.ndarray | None = None,
) -> dict[str, float]:
    """Choose the F1-maximising threshold on the validation set.

    Ties are broken by the threshold closest to 0.5, making the procedure
    deterministic without consulting the test set.
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob, dtype=float)
    if y.size == 0 or y.size != p.size:
        raise ValueError("y_true and y_prob must be non-empty and aligned")
    if not np.isfinite(p).all():
        raise ValueError("y_prob contains non-finite values")
    if np.unique(y).size < 2:
        raise ValueError("Threshold selection requires both classes")

    thresholds = np.round(np.arange(0.05, 0.951, 0.01), 2) if grid is None else np.asarray(grid)
    best = {"threshold": 0.5, "f1": -1.0, "precision": 0.0, "recall": 0.0}
    for threshold in thresholds:
        pred = (p >= threshold).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if f1 > best["f1"] + 1e-12 or (
            abs(f1 - best["f1"]) <= 1e-12 and abs(threshold - 0.5) < abs(best["threshold"] - 0.5)
        ):
            best = {
                "threshold": float(threshold),
                "f1": float(f1),
                "precision": float(precision),
                "recall": float(recall),
            }
    return best


def threshold_curve(y_true, y_prob) -> list[dict[str, float]]:
    """Precision/recall/F1 across thresholds, for dashboard plots."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob, dtype=float)
    rows = []
    for threshold in np.round(np.arange(0.05, 0.951, 0.01), 2):
        pred = (p >= threshold).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "threshold": float(threshold),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "predicted_positive_rate": float(pred.mean()),
            }
        )
    return rows


def class_weight_report(y_train) -> dict[str, float]:
    """Describe the imbalance of the training partition."""
    y = np.asarray(y_train).astype(int)
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    return {
        "n_train": int(len(y)),
        "n_positive": positives,
        "n_negative": negatives,
        "positive_rate": round(float(y.mean()), 4),
        "imbalance_ratio": round(negatives / positives, 2) if positives else float("inf"),
    }
