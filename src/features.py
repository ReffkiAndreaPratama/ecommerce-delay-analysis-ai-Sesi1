"""Leakage-aware feature engineering.

Only information available at or shortly after checkout is used in the
``safe`` feature set. Post-delivery fields are kept in a separate ``leaky``
set purely to demonstrate how leakage inflates offline metrics.
"""

from __future__ import annotations

import pandas as pd

from src.config import (
    FORBIDDEN_FEATURE_COLUMNS,
    LEAKY_EXTRA_FEATURES,
    SAFE_CATEGORICAL_FEATURES,
    SAFE_NUMERIC_FEATURES,
    TARGET_COL,
    TIME_COL,
)


def _hours_between(later: pd.Series, earlier: pd.Series) -> pd.Series:
    return (later - earlier).dt.total_seconds() / 3600.0


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features and post-delivery diagnostics from raw timestamps."""
    required = [TIME_COL, "order_estimated_delivery_date", "order_delivered_customer_date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    purchase = pd.to_datetime(out[TIME_COL], errors="coerce")
    estimated = pd.to_datetime(out["order_estimated_delivery_date"], errors="coerce")
    delivered = pd.to_datetime(out["order_delivered_customer_date"], errors="coerce")

    def _optional_datetime(column: str) -> pd.Series:
        if column in out.columns:
            return pd.to_datetime(out[column], errors="coerce")
        return pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")

    carrier = _optional_datetime("order_delivered_carrier_date")
    approved = _optional_datetime("order_approved_at")

    out["purchase_month"] = purchase.dt.month
    out["purchase_dayofweek"] = purchase.dt.dayofweek
    out["purchase_hour"] = purchase.dt.hour
    out["purchase_is_weekend"] = purchase.dt.dayofweek.isin([5, 6]).astype(int)

    out["estimated_delivery_days"] = (
        estimated.dt.normalize() - purchase.dt.normalize()
    ).dt.days

    approval_delay = _hours_between(approved, purchase)
    out["approval_missing"] = approved.isna().astype(int)
    out["approval_delay_hours"] = approval_delay.clip(lower=0)

    for col in ["price", "freight_value", "payment_value"]:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["order_total_value"] = out["price"].fillna(0) + out["freight_value"].fillna(0)
    denominator = out["order_total_value"].replace(0, pd.NA)
    out["freight_ratio"] = (out["freight_value"] / denominator).fillna(0.0)

    for col in ["n_items", "n_sellers", "payment_installments"]:
        if col not in out.columns:
            out[col] = 1
        out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in ["product_weight_g", "product_photos_qty", "product_description_lenght"]:
        if col not in out.columns:
            out[col] = pd.NA
        out[col] = pd.to_numeric(out[col], errors="coerce")

    def _optional_numeric(column: str) -> pd.Series:
        if column in out.columns:
            return pd.to_numeric(out[column], errors="coerce")
        return pd.Series(float("nan"), index=out.index, dtype=float)

    length = _optional_numeric("product_length_cm")
    height = _optional_numeric("product_height_cm")
    width = _optional_numeric("product_width_cm")
    out["product_volume_cm3"] = length * height * width
    out["product_description_length"] = _optional_numeric("product_description_lenght")

    out["same_state"] = (
        out["customer_state"].fillna("unknown") == out["seller_state"].fillna("unknown")
    ).map({True: "yes", False: "no"})
    out["is_holiday_season"] = out["purchase_month"].isin([11, 12]).map(
        {True: "yes", False: "no"}
    )

    for col in ["customer_state", "seller_state", "product_category_name", "payment_type"]:
        if col not in out.columns:
            out[col] = "unknown"
        out[col] = out[col].fillna("unknown").astype(str)

    # Post-delivery diagnostics. Never included in the safe feature set.
    out["actual_delivery_days"] = (
        delivered.dt.normalize() - purchase.dt.normalize()
    ).dt.days
    out["carrier_transit_days"] = _hours_between(delivered, carrier) / 24.0
    out["delivery_error_days"] = out["actual_delivery_days"] - out["estimated_delivery_days"]
    if "review_score" not in out.columns:
        out["review_score"] = pd.NA
    out["review_score"] = pd.to_numeric(out["review_score"], errors="coerce")

    return out


def feature_columns(feature_set: str = "safe") -> tuple[list[str], list[str]]:
    """Return ``(numeric, categorical)`` column lists for a feature set."""
    if feature_set not in {"safe", "leaky"}:
        raise ValueError("feature_set must be 'safe' or 'leaky'")
    numeric = list(SAFE_NUMERIC_FEATURES)
    categorical = list(SAFE_CATEGORICAL_FEATURES)
    if feature_set == "leaky":
        numeric.extend(LEAKY_EXTRA_FEATURES)
    return numeric, categorical


def assert_feature_safety(columns: list[str], feature_set: str = "safe") -> None:
    """Fail loudly if a post-delivery column enters the safe feature set."""
    if feature_set == "leaky":
        return
    illegal = sorted(set(columns) & FORBIDDEN_FEATURE_COLUMNS)
    if illegal:
        raise ValueError(f"Leakage detected in safe features: {illegal}")


def build_supervised_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return the engineered frame with a valid binary target.

    Splits must be created from this frame so every feature set is evaluated
    on exactly the same orders.
    """
    if TARGET_COL not in df.columns:
        raise ValueError(f"Dataset must contain the target '{TARGET_COL}'")
    if df.empty:
        raise ValueError("Cannot build features from an empty dataset")

    engineered = add_engineered_features(df)
    engineered = engineered[engineered[TARGET_COL].notna()].copy()
    engineered = engineered[
        engineered["estimated_delivery_days"].notna()
        & (engineered["estimated_delivery_days"] >= 0)
    ].copy()
    if engineered.empty:
        raise ValueError("No rows remain after feature validation")
    engineered[TARGET_COL] = engineered[TARGET_COL].astype(int)
    if engineered[TARGET_COL].nunique() < 2:
        raise ValueError("Target has a single class after feature construction")
    return engineered.reset_index(drop=True)


def feature_matrix(frame: pd.DataFrame, feature_set: str = "safe") -> tuple[pd.DataFrame, pd.Series]:
    """Select feature columns from an already-engineered frame."""
    numeric, categorical = feature_columns(feature_set)
    columns = numeric + categorical
    assert_feature_safety(columns, feature_set)
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise ValueError(f"Feature columns missing from frame: {missing}")
    X = frame[columns].copy()
    y = frame[TARGET_COL].astype(int).copy()
    X.index = pd.RangeIndex(len(X))
    y.index = pd.RangeIndex(len(y))
    return X, y


def build_model_frame(
    df: pd.DataFrame, feature_set: str = "safe"
) -> tuple[pd.DataFrame, pd.Series]:
    """Build the model matrix and binary target without any fitting."""
    return feature_matrix(build_supervised_frame(df), feature_set)


def engineered_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Public helper used by error analysis and the dashboard."""
    return add_engineered_features(df)
