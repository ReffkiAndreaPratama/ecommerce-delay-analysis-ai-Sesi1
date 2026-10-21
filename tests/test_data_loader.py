from __future__ import annotations

import pandas as pd
import pytest

from src.config import TARGET_COL
from src.data_loader import (
    DataNotFoundError,
    add_target,
    audit_dataset,
    clean_orders,
    load_dataset,
)


def test_target_uses_date_granularity():
    df = pd.DataFrame(
        {
            "order_id": ["a", "b", "c"],
            "order_delivered_customer_date": [
                "2018-01-10 23:00:00",
                "2018-01-11 00:01:00",
                "2018-01-10 23:59:00",
            ],
            "order_estimated_delivery_date": [
                "2018-01-10 00:00:00",
                "2018-01-10 00:00:00",
                "2018-01-10 00:00:00",
            ],
        }
    )
    out = add_target(df)
    assert out[TARGET_COL].tolist() == [0, 1, 0]


def test_clean_orders_drops_missing_and_duplicates():
    raw = pd.DataFrame(
        {
            "order_id": ["1", "1", "2", "3"],
            "customer_id": ["c1", "c1", "c2", "c3"],
            "order_status": ["delivered"] * 4,
            "order_purchase_timestamp": ["2018-01-01"] * 4,
            "order_estimated_delivery_date": ["2018-01-10", "2018-01-10", "2018-01-10", None],
            "order_delivered_customer_date": ["2018-01-09", "2018-01-09", "2018-01-12", "2018-01-12"],
            "customer_state": ["SP"] * 4,
            "seller_state": ["SP"] * 4,
            "product_category_name": ["toys"] * 4,
            "price": [10.0] * 4,
            "freight_value": [1.0] * 4,
            "payment_type": ["credit_card"] * 4,
            "payment_installments": [1] * 4,
            "payment_value": [11.0] * 4,
        }
    )
    out = clean_orders(raw)
    assert len(out) == 2
    assert out["order_id"].is_unique
    assert out[TARGET_COL].tolist() == [0, 1]


def test_clean_orders_rejects_missing_id():
    with pytest.raises(ValueError):
        clean_orders(pd.DataFrame({"x": [1]}))


def test_audit_returns_json_safe_payload(demo_df):
    audit = audit_dataset(demo_df)
    assert audit["n_rows"] == len(demo_df)
    assert audit["n_duplicate_order_ids"] == 0
    assert 0.0 < audit["late_rate"] < 1.0
    assert isinstance(audit["missing_values"], dict)


def test_audit_rejects_empty_frame():
    with pytest.raises(ValueError):
        audit_dataset(pd.DataFrame())


def test_load_dataset_demo_metadata():
    df, meta = load_dataset(source="demo", n_demo=80, seed=2)
    assert meta["data_source"] == "DEMO"
    assert meta["is_demo"] is True
    assert "not real" in meta["warning"].lower()
    assert df.attrs["data_source"] == "DEMO"


def test_load_dataset_olist_without_files_raises(tmp_path):
    with pytest.raises(DataNotFoundError):
        load_dataset(source="olist", raw_dir=tmp_path)
