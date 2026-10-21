"""Dataset loading, schema validation, and quality audit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import (
    DEMO_N_ORDERS,
    ID_COL,
    OLIST_FILES,
    RAW_DATA_DIR,
    RANDOM_STATE,
    TARGET_COL,
    TIME_COL,
)
from src.demo_data import generate_demo_orders

DATE_COLUMNS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]

REQUIRED_COLUMNS = [
    "order_id",
    "customer_id",
    "order_status",
    "order_purchase_timestamp",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
    "customer_state",
    "seller_state",
    "product_category_name",
    "price",
    "freight_value",
    "payment_type",
    "payment_installments",
    "payment_value",
]


class DataNotFoundError(FileNotFoundError):
    """Raised when the real Olist CSV files are not available locally."""


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise DataNotFoundError(f"Missing dataset file: {path}")
    return pd.read_csv(path, low_memory=False)


def olist_files_present(raw_dir: Path | str = RAW_DATA_DIR) -> bool:
    raw = Path(raw_dir)
    return all((raw / name).exists() for name in OLIST_FILES.values())


def load_raw_olist(raw_dir: Path | str = RAW_DATA_DIR) -> pd.DataFrame:
    """Load and join the public Olist relational tables into one order table.

    The function does not download anything and never fabricates a URL. See
    ``scripts/download_data.py`` and ``data/README.md`` for acquisition.
    """
    raw = Path(raw_dir)
    missing = [name for name in OLIST_FILES.values() if not (raw / name).exists()]
    if missing:
        raise DataNotFoundError(
            "The real Olist dataset is not available in "
            f"{raw}. Missing: {', '.join(missing)}. "
            "Run `python scripts/download_data.py --help` or use the DEMO mode."
        )

    orders = _read_csv(raw / OLIST_FILES["orders"])
    customers = _read_csv(raw / OLIST_FILES["customers"])
    items = _read_csv(raw / OLIST_FILES["order_items"])
    payments = _read_csv(raw / OLIST_FILES["payments"])
    reviews = _read_csv(raw / OLIST_FILES["reviews"])
    products = _read_csv(raw / OLIST_FILES["products"])
    sellers = _read_csv(raw / OLIST_FILES["sellers"])

    translation_path = raw / OLIST_FILES["category_translation"]
    if translation_path.exists():
        translation = _read_csv(translation_path)
        products = products.merge(
            translation, on="product_category_name", how="left"
        )
        products["product_category_name"] = products[
            "product_category_name_english"
        ].fillna(products["product_category_name"])
        products = products.drop(columns=["product_category_name_english"])

    product_cols = [
        "product_id",
        "product_category_name",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
        "product_photos_qty",
        "product_description_lenght",
    ]
    items = items.merge(products[product_cols], on="product_id", how="left")
    items = items.sort_values(["order_id", "order_item_id"])

    items_agg = items.groupby("order_id", as_index=False).agg(
        n_items=("order_item_id", "count"),
        price=("price", "sum"),
        freight_value=("freight_value", "sum"),
        n_sellers=("seller_id", "nunique"),
        product_id=("product_id", "first"),
        seller_id=("seller_id", "first"),
        product_category_name=("product_category_name", "first"),
        product_weight_g=("product_weight_g", "first"),
        product_length_cm=("product_length_cm", "first"),
        product_height_cm=("product_height_cm", "first"),
        product_width_cm=("product_width_cm", "first"),
        product_photos_qty=("product_photos_qty", "first"),
        product_description_lenght=("product_description_lenght", "first"),
    )

    payments_agg = (
        payments.sort_values(["order_id", "payment_sequential"])
        .groupby("order_id", as_index=False)
        .agg(
            payment_type=("payment_type", "first"),
            payment_installments=("payment_installments", "max"),
            payment_value=("payment_value", "sum"),
            n_payments=("payment_sequential", "count"),
        )
    )

    reviews_agg = reviews.groupby("order_id", as_index=False).agg(
        review_score=("review_score", "mean"),
        n_reviews=("review_id", "count"),
    )

    sellers_small = sellers[["seller_id", "seller_state"]].drop_duplicates("seller_id")
    customers_small = customers[
        ["customer_id", "customer_state", "customer_city", "customer_zip_code_prefix"]
    ].drop_duplicates("customer_id")

    df = orders.merge(customers_small, on="customer_id", how="left")
    df = df.merge(items_agg, on="order_id", how="left")
    df = df.merge(sellers_small, on="seller_id", how="left")
    df = df.merge(payments_agg, on="order_id", how="left")
    df = df.merge(reviews_agg, on="order_id", how="left")
    df["product_category_name"] = df["product_category_name"].fillna("unknown")
    df.attrs["data_source"] = "OLIST"
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Define late delivery at date granularity.

    A parcel is late when the customer delivery date is strictly after the
    estimated delivery date. Time-of-day is ignored because the marketplace
    publishes the estimate as a calendar date.
    """
    out = df.copy()
    delivered = pd.to_datetime(out["order_delivered_customer_date"], errors="coerce")
    estimated = pd.to_datetime(out["order_estimated_delivery_date"], errors="coerce")
    out[TARGET_COL] = (delivered.dt.normalize() > estimated.dt.normalize()).astype("Int64")
    out.loc[delivered.isna() | estimated.isna(), TARGET_COL] = pd.NA
    return out


def clean_orders(df: pd.DataFrame) -> pd.DataFrame:
    """Parse timestamps, remove unusable labels, and de-duplicate orders."""
    if ID_COL not in df.columns:
        raise ValueError(f"Dataset must contain '{ID_COL}'")

    out = df.copy()
    for col in DATE_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")

    out = out.drop_duplicates(subset=[ID_COL], keep="last")
    out = add_target(out)

    # Labels require a purchase timestamp, an estimate, and a delivery.
    before = len(out)
    out = out.dropna(subset=[TIME_COL, "order_estimated_delivery_date", "order_delivered_customer_date"])
    out = out[out[TARGET_COL].notna()].copy()

    # Chronologically impossible rows cannot be used for temporal validation.
    purchase = pd.to_datetime(out[TIME_COL])
    estimated = pd.to_datetime(out["order_estimated_delivery_date"])
    delivered = pd.to_datetime(out["order_delivered_customer_date"])
    out = out[(estimated >= purchase.dt.normalize()) & (delivered >= purchase)].copy()
    out[TARGET_COL] = out[TARGET_COL].astype(int)
    out = out.sort_values(TIME_COL).reset_index(drop=True)
    out.attrs["dropped_rows"] = int(before - len(out))
    out.attrs["data_source"] = df.attrs.get("data_source", "UNKNOWN")
    return out


def audit_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Return a JSON-serialisable audit of schema and data quality."""
    if df.empty:
        raise ValueError("Cannot audit an empty dataset")

    missing = {col: int(n) for col, n in df.isna().sum().items() if n > 0}
    purchase = pd.to_datetime(df[TIME_COL], errors="coerce")
    delivered = pd.to_datetime(df["order_delivered_customer_date"], errors="coerce")
    estimated = pd.to_datetime(df["order_estimated_delivery_date"], errors="coerce")
    target = df[TARGET_COL].astype(int)

    late_rate = float(target.mean())
    audit = {
        "data_source": df.attrs.get("data_source", "UNKNOWN"),
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "n_unique_orders": int(df[ID_COL].nunique()),
        "n_duplicate_order_ids": int(df[ID_COL].duplicated().sum()),
        "n_unique_customers": int(df["customer_id"].nunique()),
        "n_unique_sellers": int(df["seller_id"].nunique()),
        "date_min": str(purchase.min()),
        "date_max": str(purchase.max()),
        "n_late": int(target.sum()),
        "n_on_time": int((1 - target).sum()),
        "late_rate": round(late_rate, 6),
        "missing_values": missing,
        "n_missing_cells": int(df.isna().sum().sum()),
        "n_delivered_before_purchase": int((delivered < purchase).sum()),
        "n_estimated_before_purchase": int((estimated < purchase).sum()),
        "n_missing_category": int((df["product_category_name"].fillna("unknown") == "unknown").sum()),
        "target_definition": "order_delivered_customer_date::date > order_estimated_delivery_date::date",
    }
    return audit


def load_dataset(
    source: str = "auto",
    raw_dir: Path | str = RAW_DATA_DIR,
    n_demo: int = DEMO_N_ORDERS,
    seed: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the real Olist data if present, otherwise the labelled DEMO data.

    Returns the cleaned order table and a metadata dictionary. ``source`` can
    be ``auto``, ``demo``, or ``olist``. ``olist`` raises instead of silently
    falling back, so experiments cannot be mislabelled.
    """
    source = source.lower()
    if source not in {"auto", "demo", "olist"}:
        raise ValueError("source must be one of: auto, demo, olist")

    if source == "olist" or (source == "auto" and olist_files_present(raw_dir)):
        raw = load_raw_olist(raw_dir)
        data_source = "OLIST"
        origin = str(Path(raw_dir))
    elif source in {"auto", "demo"}:
        raw = generate_demo_orders(n_orders=n_demo, seed=seed)
        data_source = "DEMO"
        origin = "synthetic generator (src/demo_data.py)"
    else:
        raise DataNotFoundError(
            f"Real Olist files were not found in {raw_dir}. "
            "Download them from Kaggle or run with --source demo."
        )

    df = clean_orders(raw)
    df.attrs["data_source"] = data_source
    meta = {
        "data_source": data_source,
        "origin": origin,
        "n_rows": int(len(df)),
        "n_demo_requested": int(n_demo) if data_source == "DEMO" else None,
        "seed": int(seed) if data_source == "DEMO" else None,
        "is_demo": data_source == "DEMO",
        "warning": (
            "DEMO SYNTHETIC DATA - not real Olist observations and not research evidence."
            if data_source == "DEMO"
            else "Real Olist data. Cite the original dataset and respect CC BY-NC-SA 4.0."
        ),
    }
    return df, meta


def save_audit(audit: dict[str, Any], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
