"""Train-only preprocessing and leakage-safe data splitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    ID_COL,
    RANDOM_STATE,
    TARGET_COL,
    TEMPORAL_TRAIN_FRACTION,
    TEMPORAL_VAL_FRACTION,
    TIME_COL,
)


def make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """Build an unfitted preprocessing transformer.

    Imputation, scaling, and encoding are all fitted inside the pipeline, so
    they only ever see the training partition.
    """
    if not numeric and not categorical:
        raise ValueError("At least one numeric or categorical feature is required")

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False),
            ),
        ]
    )
    transformers = []
    if numeric:
        transformers.append(("num", numeric_pipe, numeric))
    if categorical:
        transformers.append(("cat", categorical_pipe, categorical))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def _check_classes(frame: pd.DataFrame, name: str) -> None:
    counts = frame[TARGET_COL].value_counts()
    if len(counts) < 2 or counts.min() < 2:
        raise ValueError(
            f"{name} split must contain both classes with at least 2 rows each; got {counts.to_dict()}"
        )


def split_random(
    df: pd.DataFrame,
    train_size: float = TEMPORAL_TRAIN_FRACTION,
    val_size: float = TEMPORAL_VAL_FRACTION,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified random 70/10/20 split (baseline)."""
    if not 0.5 <= train_size < 1.0:
        raise ValueError("train_size must be in [0.5, 1)")
    if not 0.0 < val_size < 1.0 - train_size:
        raise ValueError("val_size leaves no room for a test set")
    if df.empty:
        raise ValueError("Cannot split an empty dataset")

    test_size = 1.0 - train_size - val_size
    train, rest = train_test_split(
        df,
        test_size=test_size + val_size,
        stratify=df[TARGET_COL],
        random_state=random_state,
    )
    relative_val = val_size / (val_size + test_size)
    val, test = train_test_split(
        rest,
        test_size=1.0 - relative_val,
        stratify=rest[TARGET_COL],
        random_state=random_state,
    )
    train, val, test = (f.reset_index(drop=True) for f in (train, val, test))
    validate_splits(train, val, test, split_type="random")
    return train, val, test


def split_temporal(
    df: pd.DataFrame,
    train_size: float = TEMPORAL_TRAIN_FRACTION,
    val_size: float = TEMPORAL_VAL_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological 70/10/20 split: train -> validation -> test.

    The test period is always the most recent one, mirroring deployment.
    """
    if df.empty:
        raise ValueError("Cannot split an empty dataset")
    if TIME_COL not in df.columns:
        raise ValueError(f"Temporal split requires '{TIME_COL}'")
    if not 0.5 <= train_size < 1.0:
        raise ValueError("train_size must be in [0.5, 1)")
    if not 0.0 < val_size < 1.0 - train_size:
        raise ValueError("val_size leaves no room for a test set")

    ordered = df.copy()
    ordered[TIME_COL] = pd.to_datetime(ordered[TIME_COL], errors="coerce")
    ordered = ordered.dropna(subset=[TIME_COL]).sort_values(TIME_COL).reset_index(drop=True)
    n = len(ordered)
    if n < 30:
        raise ValueError("Temporal split needs at least 30 labelled orders")

    train_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))
    train = ordered.iloc[:train_end].reset_index(drop=True)
    val = ordered.iloc[train_end:val_end].reset_index(drop=True)
    test = ordered.iloc[val_end:].reset_index(drop=True)
    validate_splits(train, val, test, split_type="temporal")
    return train, val, test


def validate_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    split_type: str = "random",
) -> None:
    """Check class presence, disjoint order IDs, and temporal ordering."""
    for name, frame in (("train", train), ("validation", val), ("test", test)):
        if frame.empty:
            raise ValueError(f"{name} split is empty")
        _check_classes(frame, name)

    if ID_COL in train.columns:
        ids = [set(train[ID_COL]), set(val[ID_COL]), set(test[ID_COL])]
        if any(ids[i] & ids[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("Order IDs overlap between splits")

    if split_type not in {"random", "temporal"}:
        raise ValueError("split_type must be 'random' or 'temporal'")
    if split_type == "temporal":
        train_max = pd.to_datetime(train[TIME_COL]).max()
        val_min = pd.to_datetime(val[TIME_COL]).min()
        val_max = pd.to_datetime(val[TIME_COL]).max()
        test_min = pd.to_datetime(test[TIME_COL]).min()
        if not (train_max <= val_min <= val_max <= test_min):
            raise ValueError(
                "Temporal split is not chronological: "
                f"train_max={train_max}, val=[{val_min}, {val_max}], test_min={test_min}"
            )


def split_summary(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> dict[str, dict[str, float | str | int]]:
    """Compact split statistics used in reports and the dashboard."""
    summary: dict[str, dict[str, float | str | int]] = {}
    for name, frame in (("train", train), ("validation", val), ("test", test)):
        summary[name] = {
            "n_rows": int(len(frame)),
            "late_rate": round(float(frame[TARGET_COL].mean()), 4),
            "n_late": int(frame[TARGET_COL].sum()),
            "date_min": str(pd.to_datetime(frame[TIME_COL]).min()),
            "date_max": str(pd.to_datetime(frame[TIME_COL]).max()),
        }
    return summary


def splitter_for(split_type: str):
    """Return the split function for a configured strategy name."""
    if split_type == "random":
        return split_random
    if split_type == "temporal":
        return split_temporal
    raise ValueError("split_type must be 'random' or 'temporal'")


def positive_rate(y: np.ndarray | pd.Series) -> float:
    arr = np.asarray(y)
    if arr.size == 0:
        raise ValueError("Cannot compute positive rate of an empty target")
    return float(arr.mean())
