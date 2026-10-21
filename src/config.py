"""Project-wide configuration for the e-commerce late-delivery study.

All paths are derived from the repository root so scripts behave the same
whether they are launched from the project root or from ``scripts/``.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
DEMO_DATA_DIR = DATA_DIR / "demo"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURE_DIR = RESULTS_DIR / "figures"
PREDICTION_DIR = RESULTS_DIR / "predictions"

RANDOM_STATE = 42
TARGET_COL = "is_late"
TIME_COL = "order_purchase_timestamp"
ID_COL = "order_id"

# Fraction of the timeline used for each split. The remaining 20% is the test
# set, which is never used for threshold selection or model selection.
TEMPORAL_TRAIN_FRACTION = 0.70
TEMPORAL_VAL_FRACTION = 0.10

# Demo data defaults. The demo is synthetic and always labelled DEMO.
DEMO_N_ORDERS = 15_000
DEMO_TARGET_LATE_RATE = 0.11

# Olist files required to build the real dataset. The public dataset is
# distributed on Kaggle under CC BY-NC-SA 4.0 (see data/README.md).
OLIST_FILES = {
    "orders": "olist_orders_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}

# Columns that must never be used as deployment-time features.
FORBIDDEN_FEATURE_COLUMNS = {
    "order_delivered_customer_date",
    "order_delivered_carrier_date",
    "order_approved_at",
    "order_status",
    "review_score",
}

SAFE_NUMERIC_FEATURES = [
    "purchase_month",
    "purchase_dayofweek",
    "purchase_hour",
    "purchase_is_weekend",
    "estimated_delivery_days",
    "approval_delay_hours",
    "approval_missing",
    "price",
    "freight_value",
    "order_total_value",
    "freight_ratio",
    "n_items",
    "n_sellers",
    "payment_installments",
    "payment_value",
    "product_weight_g",
    "product_volume_cm3",
    "product_photos_qty",
    "product_description_length",
]

SAFE_CATEGORICAL_FEATURES = [
    "customer_state",
    "seller_state",
    "product_category_name",
    "payment_type",
    "same_state",
    "is_holiday_season",
]

# Post-delivery columns used only in the deliberately leaky comparison. They
# would not exist at prediction time and are included to quantify optimism.
LEAKY_EXTRA_FEATURES = [
    "actual_delivery_days",
    "carrier_transit_days",
    "review_score",
    "delivery_error_days",
]

SUPPORTED_MODELS = ("logistic_regression", "random_forest")
SUPPORTED_STRATEGIES = ("none", "class_weight")
SUPPORTED_FEATURE_SETS = ("safe", "leaky")
SUPPORTED_SPLITS = ("random", "temporal")

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
}
STRATEGY_LABELS = {
    "none": "Tanpa penyeimbangan",
    "class_weight": "Class weighting",
}
SPLIT_LABELS = {
    "random": "Random split",
    "temporal": "Temporal split",
}
FEATURE_SET_LABELS = {
    "safe": "Fitur aman (deployment-safe)",
    "leaky": "Fitur bocor (leakage demo)",
}
