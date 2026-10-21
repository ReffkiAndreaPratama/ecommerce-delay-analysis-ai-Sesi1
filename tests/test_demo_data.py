from __future__ import annotations

import pandas as pd
import pytest

from src.demo_data import generate_demo_orders


def test_demo_schema_and_determinism():
    first = generate_demo_orders(n_orders=120, seed=3)
    second = generate_demo_orders(n_orders=120, seed=3)
    assert len(first) == 120
    assert first[["order_id", "customer_id", "order_purchase_timestamp"]].equals(
        second[["order_id", "customer_id", "order_purchase_timestamp"]]
    )
    for column in [
        "order_purchase_timestamp",
        "order_estimated_delivery_date",
        "order_delivered_customer_date",
        "customer_state",
        "seller_state",
        "price",
        "review_score",
    ]:
        assert column in first.columns


def test_demo_has_minority_late_class():
    df = generate_demo_orders(n_orders=1200, seed=11)
    delivered = pd.to_datetime(df["order_delivered_customer_date"]).dt.normalize()
    estimated = pd.to_datetime(df["order_estimated_delivery_date"]).dt.normalize()
    late_rate = float((delivered > estimated).mean())
    assert 0.04 < late_rate < 0.25


def test_demo_timestamps_are_ordered():
    df = generate_demo_orders(n_orders=200, seed=5)
    purchase = pd.to_datetime(df["order_purchase_timestamp"])
    delivered = pd.to_datetime(df["order_delivered_customer_date"])
    estimated = pd.to_datetime(df["order_estimated_delivery_date"])
    assert (delivered >= purchase).all()
    assert (estimated >= purchase.dt.normalize()).all()


def test_demo_rejects_invalid_arguments():
    with pytest.raises(ValueError):
        generate_demo_orders(n_orders=10)
    with pytest.raises(ValueError):
        generate_demo_orders(n_orders=100, target_late_rate=0.9)
