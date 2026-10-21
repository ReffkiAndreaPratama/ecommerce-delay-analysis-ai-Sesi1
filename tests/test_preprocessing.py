from __future__ import annotations

import pandas as pd
import pytest

from src.config import TARGET_COL
from src.preprocessing import (
    make_preprocessor,
    split_random,
    split_temporal,
    splitter_for,
    validate_splits,
)


def test_random_split_sizes_and_disjoint_ids(demo_df):
    train, val, test = split_random(demo_df)
    total = len(demo_df)
    assert abs(len(train) / total - 0.70) < 0.02
    assert abs(len(val) / total - 0.10) < 0.02
    assert abs(len(test) / total - 0.20) < 0.02
    assert set(train["order_id"]).isdisjoint(val["order_id"])
    assert set(train["order_id"]).isdisjoint(test["order_id"])
    assert set(val["order_id"]).isdisjoint(test["order_id"])


def test_temporal_split_is_chronological(demo_df):
    train, val, test = split_temporal(demo_df)
    train_max = pd.to_datetime(train["order_purchase_timestamp"]).max()
    val_min = pd.to_datetime(val["order_purchase_timestamp"]).min()
    val_max = pd.to_datetime(val["order_purchase_timestamp"]).max()
    test_min = pd.to_datetime(test["order_purchase_timestamp"]).min()
    assert train_max <= val_min
    assert val_max <= test_min


def test_validate_splits_detects_overlap(demo_df):
    train, val, test = split_random(demo_df)
    leaked = pd.concat([train, val.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        validate_splits(leaked, val, test, split_type="random")


def test_validate_splits_rejects_single_class(demo_df):
    train, val, test = split_random(demo_df)
    train = train.copy()
    train[TARGET_COL] = 0
    with pytest.raises(ValueError):
        validate_splits(train, val, test)


def test_preprocessor_fits_unknown_categories_safely(demo_df):
    train, _, test = split_random(demo_df)
    numeric = ["price", "freight_value"]
    categorical = ["customer_state"]
    pre = make_preprocessor(numeric, categorical)
    transformed_train = pre.fit_transform(train[numeric + categorical])
    unseen = test[numeric + categorical].copy()
    unseen.loc[unseen.index[0], "customer_state"] = "ZZ"
    transformed_test = pre.transform(unseen)
    assert transformed_train.shape[1] == transformed_test.shape[1]
    assert transformed_test.shape[0] == len(unseen)


def test_missing_values_are_imputed(demo_df):
    train, _, _ = split_random(demo_df)
    frame = train[["price", "freight_value", "customer_state"]].copy()
    frame.loc[frame.index[0], "price"] = None
    frame.loc[frame.index[1], "customer_state"] = None
    pre = make_preprocessor(["price", "freight_value"], ["customer_state"])
    transformed = pre.fit_transform(frame)
    assert not pd.isna(transformed).any()


def test_empty_split_input_raises():
    empty = pd.DataFrame()
    with pytest.raises(ValueError):
        split_random(empty)
    with pytest.raises(ValueError):
        split_temporal(empty)


def test_splitter_factory():
    assert splitter_for("random") is split_random
    assert splitter_for("temporal") is split_temporal
    with pytest.raises(ValueError):
        splitter_for("unknown")
