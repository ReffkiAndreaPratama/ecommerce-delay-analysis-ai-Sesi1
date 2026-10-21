from __future__ import annotations

import pandas as pd
import pytest

from src.config import FORBIDDEN_FEATURE_COLUMNS, TARGET_COL
from src.features import (
    add_engineered_features,
    assert_feature_safety,
    build_model_frame,
    build_supervised_frame,
    feature_columns,
    feature_matrix,
)


def test_engineered_columns_and_target(demo_df):
    out = add_engineered_features(demo_df)
    for column in [
        "purchase_month",
        "estimated_delivery_days",
        "freight_ratio",
        "same_state",
        "actual_delivery_days",
        "delivery_error_days",
    ]:
        assert column in out.columns
    assert out[TARGET_COL].notna().all()


def test_safe_feature_set_excludes_post_delivery_columns(demo_df):
    numeric, categorical = feature_columns("safe")
    columns = set(numeric + categorical)
    assert not (columns & FORBIDDEN_FEATURE_COLUMNS)
    with pytest.raises(ValueError):
        assert_feature_safety(["review_score"], "safe")
    X, y = build_model_frame(demo_df, "safe")
    assert len(X) == len(y)
    assert set(X.columns) == columns


def test_leaky_feature_set_contains_post_delivery_columns(demo_df):
    numeric, _ = feature_columns("leaky")
    assert "review_score" in numeric
    X, _ = build_model_frame(demo_df, "leaky")
    assert "actual_delivery_days" in X.columns


def test_estimated_delivery_days_is_non_negative(demo_df):
    frame = build_supervised_frame(demo_df)
    assert (frame["estimated_delivery_days"] >= 0).all()


def test_feature_matrix_rejects_missing_columns(demo_df):
    frame = build_supervised_frame(demo_df).drop(columns=["customer_state"])
    with pytest.raises(ValueError):
        feature_matrix(frame, "safe")


def test_missing_estimated_date_rows_are_dropped():
    df = pd.DataFrame(
        {
            "order_id": ["1", "2", "3"],
            "order_purchase_timestamp": ["2018-01-01", "2018-01-01", "2018-01-01"],
            "order_estimated_delivery_date": ["2018-01-10", None, "2018-01-10"],
            "order_delivered_customer_date": ["2018-01-09", "2018-01-09", "2018-01-12"],
            TARGET_COL: [0, 1, 1],
            "customer_state": ["SP", "SP", "SP"],
            "seller_state": ["SP", "SP", "SP"],
            "product_category_name": ["toys", "toys", "toys"],
            "payment_type": ["credit_card", "credit_card", "credit_card"],
            "price": [10.0, 10.0, 10.0],
            "freight_value": [1.0, 1.0, 1.0],
            "payment_value": [11.0, 11.0, 11.0],
        }
    )
    frame = build_supervised_frame(df)
    assert len(frame) == 2
    assert frame["order_id"].tolist() == ["1", "3"]


def test_same_state_flag(demo_df):
    frame = add_engineered_features(demo_df)
    expected = (frame["customer_state"] == frame["seller_state"]).map({True: "yes", False: "no"})
    assert frame["same_state"].equals(expected)
