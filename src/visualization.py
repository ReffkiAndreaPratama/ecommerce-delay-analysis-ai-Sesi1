"""Matplotlib figures for the report and the Streamlit dashboard."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve

from src.config import (
    FEATURE_SET_LABELS,
    MODEL_LABELS,
    SPLIT_LABELS,
    STRATEGY_LABELS,
    TARGET_COL,
)

COLORS = ["#2563eb", "#f97316", "#0f766e", "#b91c1c"]
plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 9,
    }
)


def _save(fig: plt.Figure, path: Path | str) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output


def plot_class_distribution(df: pd.DataFrame, path: Path | str) -> Path:
    counts = df[TARGET_COL].astype(int).value_counts().reindex([0, 1], fill_value=0)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    bars = ax.bar(["Tepat waktu", "Terlambat"], counts.values, color=["#64748b", "#dc2626"])
    ax.set_ylabel("Jumlah pesanan")
    ax.set_title("Distribusi kelas target")
    total = int(counts.sum())
    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:,}\n({value / total:.1%})",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylim(0, max(counts.values) * 1.22)
    return _save(fig, path)


def _label(row: pd.Series) -> str:
    model = MODEL_LABELS.get(row["model"], row["model"])
    strategy = "CW" if row["strategy"] == "class_weight" else "none"
    split = "temporal" if row["split"] == "temporal" else "random"
    return f"{model}\n{strategy} / {split}"


def plot_model_comparison(
    results: pd.DataFrame, path: Path | str, metric: str = "pr_auc", feature_set: str = "safe"
) -> Path:
    frame = results[results["feature_set"] == feature_set].copy()
    if frame.empty:
        raise ValueError("No results available for the requested feature set")
    frame["label"] = frame.apply(_label, axis=1)
    frame = frame.sort_values(metric)
    fig, ax = plt.subplots(figsize=(8.4, max(3.0, 0.38 * len(frame))))
    colors = [COLORS[1] if s == "temporal" else COLORS[0] for s in frame["split"]]
    ax.barh(frame["label"], frame[metric], color=colors)
    ax.set_xlabel(f"{metric} (test set)")
    ax.set_title(f"Perbandingan model - fitur {FEATURE_SET_LABELS.get(feature_set, feature_set)}")
    ax.set_xlim(0, min(1.0, max(0.6, frame[metric].max() * 1.15)))
    for y, value in enumerate(frame[metric]):
        ax.text(value + 0.01, y, f"{value:.3f}", va="center", fontsize=8)
    return _save(fig, path)


def plot_validation_comparison(
    results: pd.DataFrame, path: Path | str, metric: str = "pr_auc", feature_set: str = "safe"
) -> Path:
    frame = results[
        (results["feature_set"] == feature_set) & results[metric].notna()
    ].copy()
    if frame.empty:
        raise ValueError("No results available for validation comparison")
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for i, split in enumerate(["random", "temporal"]):
        subset = frame[frame["split"] == split]
        if subset.empty:
            continue
        values = subset[metric].to_numpy()
        x = np.full(len(values), i) + np.linspace(-0.12, 0.12, len(values))
        ax.scatter(x, values, s=42, color=COLORS[i], label=SPLIT_LABELS[split], zorder=3)
        ax.hlines(values.mean(), i - 0.22, i + 0.22, color=COLORS[i], linewidth=2)
    ax.set_xticks([0, 1], [SPLIT_LABELS["random"], SPLIT_LABELS["temporal"]])
    ax.set_ylabel(f"{metric} (test set)")
    ax.set_title("Random vs temporal validation")
    ax.legend(frameon=False)
    return _save(fig, path)


def plot_leakage_comparison(results: pd.DataFrame, path: Path | str, metric: str = "pr_auc") -> Path:
    frame = results[results[metric].notna()].copy()
    if frame.empty:
        raise ValueError("No results available for leakage comparison")
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for i, feature_set in enumerate(["safe", "leaky"]):
        subset = frame[frame["feature_set"] == feature_set]
        if subset.empty:
            continue
        values = subset[metric].to_numpy()
        x = np.full(len(values), i) + np.linspace(-0.12, 0.12, len(values))
        ax.scatter(x, values, s=42, color=COLORS[i], zorder=3, label=FEATURE_SET_LABELS[feature_set])
        ax.hlines(values.mean(), i - 0.22, i + 0.22, color=COLORS[i], linewidth=2)
    ax.set_xticks([0, 1], ["Fitur aman", "Fitur bocor"])
    ax.set_ylabel(f"{metric} (test set)")
    ax.set_title("Dampak data leakage terhadap metrik offline")
    ax.legend(frameon=False)
    return _save(fig, path)


def plot_pr_curve(y_true, y_prob, path: Path | str, label: str = "Model") -> Path:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob, dtype=float)
    precision, recall, _ = precision_recall_curve(y, p)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(recall, precision, color=COLORS[0], label=label)
    ax.axhline(y.mean(), color="#64748b", linestyle="--", linewidth=1, label=f"Baseline ({y.mean():.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve (test set)")
    ax.legend(frameon=False)
    return _save(fig, path)


def plot_calibration(y_true, y_prob, path: Path | str, n_bins: int = 10) -> Path:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob, dtype=float)
    if np.unique(y).size < 2:
        raise ValueError("Calibration plot requires both classes")
    fraction, mean_predicted = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#64748b", label="Kalibrasi sempurna")
    ax.plot(mean_predicted, fraction, marker="o", color=COLORS[2], label="Model")
    ax.set_xlabel("Rata-rata probabilitas prediksi")
    ax.set_ylabel("Fraksi aktual terlambat")
    ax.set_title("Calibration curve (test set)")
    ax.legend(frameon=False)
    return _save(fig, path)


def plot_threshold_curve(curve: pd.DataFrame, path: Path | str, selected: float | None = None) -> Path:
    fig, ax = plt.subplots(figsize=(6.2, 3.5))
    for column, color in (("precision", COLORS[0]), ("recall", COLORS[1]), ("f1", COLORS[2])):
        ax.plot(curve["threshold"], curve[column], label=column.title(), color=color)
    if selected is not None:
        ax.axvline(selected, color="#111827", linestyle=":", label=f"threshold={selected:.2f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Skor")
    ax.set_ylim(0, 1.02)
    ax.set_title("Pemilihan threshold pada validation set")
    ax.legend(frameon=False, ncol=4)
    return _save(fig, path)


def plot_error_by_state(table: pd.DataFrame, path: Path | str, top_n: int = 12) -> Path:
    frame = table.head(top_n).copy()
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    x = np.arange(len(frame))
    width = 0.38
    ax.bar(x - width / 2, frame["false_negative_rate"], width, color=COLORS[3], label="False negative")
    ax.bar(x + width / 2, frame["false_positive_rate"], width, color=COLORS[1], label="False positive")
    ax.set_xticks(x, frame["customer_state"], rotation=45, ha="right")
    ax.set_ylabel("Proporsi pesanan")
    ax.set_title("Jenis kesalahan per negara bagian (top volume)")
    ax.legend(frameon=False)
    return _save(fig, path)


def plot_missingness(df: pd.DataFrame, path: Path | str, top_n: int = 15) -> Path:
    missing = (df.isna().mean() * 100).sort_values(ascending=False).head(top_n)
    missing = missing[missing > 0]
    fig, ax = plt.subplots(figsize=(6.4, max(2.8, 0.3 * len(missing))))
    if missing.empty:
        ax.text(0.5, 0.5, "Tidak ada nilai kosong", ha="center", va="center")
        ax.axis("off")
    else:
        ax.barh(missing.index[::-1], missing.values[::-1], color=COLORS[0])
        ax.set_xlabel("Persentase nilai kosong (%)")
        ax.set_title("Kolom dengan nilai kosong terbanyak")
    return _save(fig, path)


def plot_split_timeline(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, path: Path | str
) -> Path:
    fig, ax = plt.subplots(figsize=(8.0, 2.6))
    for i, (name, frame, color) in enumerate(
        (("Train", train, COLORS[0]), ("Validation", val, COLORS[1]), ("Test", test, COLORS[2]))
    ):
        dates = pd.to_datetime(frame["order_purchase_timestamp"])
        ax.scatter(dates, np.full(len(dates), i), s=4, alpha=0.35, color=color)
        ax.text(dates.min(), i + 0.18, name, fontsize=8, color=color)
    ax.set_yticks([])
    ax.set_xlabel("Tanggal pembelian")
    ax.set_title("Pembagian temporal: train lebih lama, test paling baru")
    return _save(fig, path)
