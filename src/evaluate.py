"""Evaluation metrics, error analysis, and result persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import TARGET_COL, TIME_COL


def _validate_inputs(y_true, y_prob, threshold: float) -> tuple[np.ndarray, np.ndarray, float]:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob, dtype=float)
    if y.size == 0:
        raise ValueError("Cannot evaluate an empty target")
    if y.size != p.size:
        raise ValueError("y_true and y_prob must have the same length")
    if not np.isfinite(p).all():
        raise ValueError("y_prob contains non-finite values")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be within [0, 1]")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("y_true must be binary (0/1)")
    return y, p, float(threshold)


def compute_binary_metrics(y_true, y_prob, threshold: float = 0.5) -> dict[str, Any]:
    """Compute discrimination, ranking, and calibration metrics.

    PR-AUC (average precision) is the headline metric because late parcels are
    the minority class.
    """
    y, p, threshold = _validate_inputs(y_true, y_prob, threshold)
    y_pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()

    two_classes = np.unique(y).size == 2
    metrics: dict[str, Any] = {
        "n_test": int(y.size),
        "n_positive": int(y.sum()),
        "late_rate": round(float(y.mean()), 4),
        "threshold": round(threshold, 2),
        "accuracy": round(float(accuracy_score(y, y_pred)), 4),
        "precision": round(float(precision_score(y, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, y_pred, zero_division=0)), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4) if two_classes else None,
        "roc_auc": round(float(roc_auc_score(y, p)), 4) if two_classes else None,
        "brier": round(float(brier_score_loss(y, p)), 4),
        "predicted_positive_rate": round(float(y_pred.mean()), 4),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return metrics


def confusion_frame(metrics: dict[str, Any]) -> pd.DataFrame:
    """Pretty confusion matrix for the dashboard."""
    return pd.DataFrame(
        [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]],
        index=["Aktual: tepat waktu", "Aktual: terlambat"],
        columns=["Prediksi: tepat waktu", "Prediksi: terlambat"],
    )


def augment_with_predictions(
    df: pd.DataFrame, y_prob, threshold: float = 0.5
) -> pd.DataFrame:
    """Attach predictions and key context columns for error review."""
    if len(df) != len(y_prob):
        raise ValueError("Prediction vector does not match the test frame")
    out = df.copy()
    out["predicted_probability"] = np.asarray(y_prob, dtype=float)
    out["predicted_late"] = (out["predicted_probability"] >= threshold).astype(int)
    if TARGET_COL in out.columns:
        out["actual_late"] = out[TARGET_COL].astype(int)
        out["error_type"] = "correct"
        out.loc[(out["actual_late"] == 1) & (out["predicted_late"] == 0), "error_type"] = "false_negative"
        out.loc[(out["actual_late"] == 0) & (out["predicted_late"] == 1), "error_type"] = "false_positive"
    return out


def error_breakdown(predictions: pd.DataFrame, by: str = "customer_state") -> pd.DataFrame:
    """Error rates and late rate grouped by a context column."""
    if by not in predictions.columns:
        raise ValueError(f"Column '{by}' is not available for error analysis")
    frame = predictions.copy()
    frame[by] = frame[by].fillna("unknown").astype(str)
    grouped = frame.groupby(by, observed=True).agg(
        n_orders=("predicted_late", "size"),
        actual_late_rate=("actual_late", "mean"),
        predicted_late_rate=("predicted_late", "mean"),
        false_negatives=("error_type", lambda s: int((s == "false_negative").sum())),
        false_positives=("error_type", lambda s: int((s == "false_positive").sum())),
    )
    grouped["false_negative_rate"] = grouped["false_negatives"] / grouped["n_orders"]
    grouped["false_positive_rate"] = grouped["false_positives"] / grouped["n_orders"]
    return grouped.sort_values("n_orders", ascending=False).reset_index()


def performance_drop(
    results: pd.DataFrame,
    metric: str = "pr_auc",
    baseline_split: str = "random",
    comparison_split: str = "temporal",
) -> pd.DataFrame:
    """Compare random vs temporal performance for matched configurations."""
    if metric not in results.columns:
        raise ValueError(f"Metric '{metric}' is not in the results table")
    keys = ["feature_set", "model", "strategy"]
    missing = [k for k in keys + ["split"] if k not in results.columns]
    if missing:
        raise ValueError(f"Results table is missing columns: {missing}")

    base = results[results["split"] == baseline_split][keys + [metric]].rename(
        columns={metric: f"{baseline_split}_{metric}"}
    )
    comp = results[results["split"] == comparison_split][keys + [metric]].rename(
        columns={metric: f"{comparison_split}_{metric}"}
    )
    merged = base.merge(comp, on=keys, how="inner")
    if merged.empty:
        raise ValueError("No matched configurations for the requested splits")
    merged["absolute_drop"] = merged[f"{baseline_split}_{metric}"] - merged[f"{comparison_split}_{metric}"]
    merged["relative_drop_pct"] = (
        100.0 * merged["absolute_drop"] / merged[f"{baseline_split}_{metric}"].replace(0, np.nan)
    ).round(2)
    return merged.sort_values("absolute_drop", ascending=False).reset_index(drop=True)


def save_metrics_csv(rows: list[dict[str, Any]], path: Path | str) -> pd.DataFrame:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


def save_json(payload: dict[str, Any], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def build_summary(results: pd.DataFrame) -> dict[str, Any]:
    """Derive headline numbers from the actual results table.

    Nothing here is hard-coded: every value comes from the experiment run.
    """
    if results.empty:
        return {}
    safe = results[results["feature_set"] == "safe"]
    leakage = results[results["feature_set"] == "leaky"]
    summary: dict[str, Any] = {
        "n_configurations": int(len(results)),
        "best_safe_by_validation_pr_auc": None,
        "leakage_inflation": None,
        "temporal_degradation": None,
    }
    if not safe.empty and "val_pr_auc" in safe.columns:
        best = safe.sort_values(["val_pr_auc", "model"], ascending=[False, True]).iloc[0]
        summary["best_safe_by_validation_pr_auc"] = {
            "model": best["model"],
            "strategy": best["strategy"],
            "split": best["split"],
            "val_pr_auc": float(best["val_pr_auc"]),
            "test_pr_auc": float(best["pr_auc"]),
            "test_f1": float(best["f1"]),
            "threshold": float(best["threshold"]),
        }
    if not safe.empty and not leakage.empty and "pr_auc" in results.columns:
        safe_mean = float(safe["pr_auc"].dropna().mean())
        leaky_mean = float(leakage["pr_auc"].dropna().mean())
        summary["leakage_inflation"] = {
            "safe_mean_test_pr_auc": round(safe_mean, 4),
            "leaky_mean_test_pr_auc": round(leaky_mean, 4),
            "absolute_difference": round(leaky_mean - safe_mean, 4),
        }
    try:
        drops = performance_drop(results, metric="pr_auc")
        summary["temporal_degradation"] = {
            "mean_absolute_drop": round(float(drops["absolute_drop"].mean()), 4),
            "max_absolute_drop": round(float(drops["absolute_drop"].max()), 4),
            "n_matched_configurations": int(len(drops)),
        }
    except ValueError:
        pass
    return summary


def latest_timestamp(df: pd.DataFrame) -> str:
    if TIME_COL not in df.columns:
        return "unknown"
    return str(pd.to_datetime(df[TIME_COL]).max())
