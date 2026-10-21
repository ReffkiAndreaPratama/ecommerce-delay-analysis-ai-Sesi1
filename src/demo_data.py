"""Synthetic, Olist-shaped demo data.

The demo exists so the full pipeline can be exercised without Kaggle
credentials or a large download. It is **not** a substitute for the real
Olist data and every artefact derived from it is labelled ``DEMO``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import DEMO_TARGET_LATE_RATE

# Approximate geography used only to give the synthetic data a realistic
# distance/latency structure. It is not a source of research claims.
STATE_WEIGHTS = {
    "SP": 0.42,
    "RJ": 0.13,
    "MG": 0.12,
    "RS": 0.055,
    "PR": 0.05,
    "SC": 0.04,
    "BA": 0.034,
    "DF": 0.02,
    "ES": 0.02,
    "GO": 0.02,
    "PE": 0.017,
    "CE": 0.016,
    "PA": 0.01,
    "MT": 0.009,
    "MA": 0.007,
    "MS": 0.007,
    "PB": 0.006,
    "PI": 0.005,
    "RN": 0.005,
    "AL": 0.004,
    "SE": 0.004,
    "TO": 0.003,
    "RO": 0.003,
    "AM": 0.003,
    "AC": 0.001,
    "AP": 0.001,
    "RR": 0.001,
}

# Extra days added to the promised lead time, by destination state.
STATE_LEAD_DAYS = {
    "SP": 0.0,
    "RJ": 1.0,
    "MG": 1.0,
    "PR": 1.5,
    "SC": 2.0,
    "RS": 2.5,
    "DF": 3.0,
    "ES": 2.0,
    "GO": 3.0,
    "BA": 4.5,
    "PE": 4.5,
    "CE": 4.5,
    "PA": 6.0,
    "MT": 5.0,
    "MA": 5.5,
    "MS": 5.0,
    "PB": 5.0,
    "PI": 5.0,
    "RN": 5.0,
    "AL": 5.0,
    "SE": 5.0,
    "TO": 5.0,
    "RO": 6.0,
    "AM": 6.5,
    "AC": 7.0,
    "AP": 7.0,
    "RR": 7.5,
}

CATEGORIES = [
    "bed_bath_table",
    "health_beauty",
    "sports_leisure",
    "furniture_decor",
    "computers_accessories",
    "housewares",
    "watches_gifts",
    "telephony",
    "garden_tools",
    "auto",
    "toys",
    "cool_stuff",
    "perfumery",
    "baby",
    "electronics",
    "stationery",
    "fashion_bags_accessories",
    "pet_shop",
    "office_furniture",
    "unknown",
]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _calibrate_intercept(logits_without_intercept: np.ndarray, target_rate: float) -> float:
    """Find the intercept that makes the mean predicted probability match target."""
    low, high = -20.0, 20.0
    for _ in range(80):
        mid = (low + high) / 2.0
        rate = float(_sigmoid(logits_without_intercept + mid).mean())
        if rate > target_rate:
            high = mid
        else:
            low = mid
    return (low + high) / 2.0


def generate_demo_orders(
    n_orders: int = 15_000,
    seed: int = 42,
    target_late_rate: float = DEMO_TARGET_LATE_RATE,
) -> pd.DataFrame:
    """Generate an order-level table that mimics the joined Olist schema.

    The target is defined exactly as in the real pipeline:
    a delivery is late when the customer delivery date is later than the
    estimated delivery date (date granularity).
    """
    if n_orders < 50:
        raise ValueError("n_orders must be at least 50")
    if not 0.01 <= target_late_rate <= 0.5:
        raise ValueError("target_late_rate must be between 0.01 and 0.5")

    rng = np.random.default_rng(seed)

    states = list(STATE_WEIGHTS)
    state_probs = np.array([STATE_WEIGHTS[s] for s in states], dtype=float)
    state_probs /= state_probs.sum()

    customer_state = rng.choice(states, size=n_orders, p=state_probs)
    seller_state = rng.choice(states, size=n_orders, p=state_probs)
    # Most sellers are concentrated in the south-east in the real marketplace.
    sp_mask = rng.random(n_orders) < 0.62
    seller_state[sp_mask] = "SP"

    # Purchase timestamps: 2017-01-01 .. 2018-08-31 with a year-end peak.
    day_index = np.arange(608)
    day_weights = 0.35 + day_index / day_index.max() * 0.8
    day_weights[300:365] *= 1.35  # Nov/Dec 2017
    day_weights[560:608] *= 1.20  # Nov/Dec 2018
    day_weights /= day_weights.sum()
    purchase_day = rng.choice(day_index, size=n_orders, p=day_weights)
    hour_weights = np.array(
        [0.5, 0.3, 0.2, 0.2, 0.2, 0.3, 0.8, 1.4, 2.4, 3.2, 3.8, 4.0,
         4.1, 4.0, 3.9, 3.7, 3.5, 3.2, 2.8, 2.4, 2.0, 1.6, 1.1, 0.7]
    )
    purchase_hour = rng.choice(np.arange(24), size=n_orders, p=hour_weights / hour_weights.sum())
    purchase_minute = rng.integers(0, 60, size=n_orders)
    purchase_second = rng.integers(0, 60, size=n_orders)
    purchase = (
        pd.Timestamp("2017-01-01")
        + pd.to_timedelta(purchase_day, unit="D")
        + pd.to_timedelta(purchase_hour, unit="h")
        + pd.to_timedelta(purchase_minute, unit="m")
        + pd.to_timedelta(purchase_second, unit="s")
    )

    lead_days = (
        7.0
        + np.array([STATE_LEAD_DAYS[s] for s in customer_state])
        + rng.normal(0.0, 2.2, size=n_orders)
        + np.where(purchase.month.isin([11, 12]), 4.0, 0.0)
    )
    lead_days = np.clip(np.round(lead_days), 3, 45).astype(int)

    approval_hours = np.clip(rng.lognormal(mean=0.3, sigma=0.55, size=n_orders), 0.05, 72)
    approval_missing = rng.random(n_orders) < 0.02
    approval_hours[approval_missing] = np.nan

    price = np.clip(rng.lognormal(mean=4.05, sigma=0.75, size=n_orders), 5, 4500)
    n_items = rng.choice([1, 2, 3, 4], size=n_orders, p=[0.845, 0.11, 0.035, 0.01])
    price = price * n_items
    freight = np.clip(
        6.0
        + 0.09 * price
        + np.array([STATE_LEAD_DAYS[s] for s in customer_state]) * 1.1
        + rng.normal(0, 3.0, size=n_orders),
        0,
        None,
    )
    payment_installments = rng.choice(
        np.arange(1, 13), size=n_orders,
        p=np.array([0.45, 0.18, 0.10, 0.07, 0.05, 0.04, 0.03, 0.02, 0.02, 0.015, 0.015, 0.01]),
    )
    payment_type = rng.choice(
        ["credit_card", "boleto", "voucher", "debit_card"],
        size=n_orders,
        p=[0.74, 0.19, 0.05, 0.02],
    )
    payment_value = price + freight

    category = rng.choice(CATEGORIES, size=n_orders, p=np.array(
        [0.11, 0.10, 0.09, 0.08, 0.075, 0.07, 0.065, 0.06, 0.055, 0.05,
         0.045, 0.04, 0.035, 0.03, 0.025, 0.02, 0.02, 0.015, 0.01, 0.005]
    ))
    category = np.where(rng.random(n_orders) < 0.02, "unknown", category)

    weight_g = np.clip(rng.lognormal(mean=6.6, sigma=1.1, size=n_orders), 50, 30000)
    volume = np.clip(rng.lognormal(mean=8.5, sigma=1.2, size=n_orders), 100, 200000)
    photos = rng.integers(1, 6, size=n_orders)
    description_length = np.clip(rng.normal(650, 380, size=n_orders), 20, 3000).astype(int)

    # Lateness propensity. Only a subset of these signals is observable at
    # purchase time, which keeps the task non-trivial.
    state_effect = {s: rng.normal(0, 0.35) for s in states}
    seller_effect = {s: rng.normal(0, 0.30) for s in states}
    logits = (
        0.55 * np.where(purchase.month.isin([11, 12]), 1.0, 0.0)
        + 0.30 * (lead_days - lead_days.mean()) / lead_days.std()
        + 0.22 * (np.nan_to_num(approval_hours, nan=2.0) > 6.0)
        + 0.28 * (n_items - 1)
        + 0.20 * (freight / (price + freight) > 0.22)
        + np.array([state_effect[s] for s in customer_state])
        + np.array([seller_effect[s] for s in seller_state])
        - 0.18 * (price > np.quantile(price, 0.75))
        + rng.normal(0, 0.55, size=n_orders)
    )
    intercept = _calibrate_intercept(logits, target_late_rate)
    late_prob = _sigmoid(logits + intercept)
    is_late = rng.random(n_orders) < late_prob

    extra_delay = np.clip(rng.exponential(scale=4.5, size=n_orders), 1, 30).astype(int)
    actual_days = np.where(
        is_late,
        lead_days + extra_delay,
        np.maximum(1, lead_days - rng.integers(0, 4, size=n_orders)),
    )
    delivered = purchase + pd.to_timedelta(actual_days, unit="D") + pd.to_timedelta(
        rng.integers(0, 600, size=n_orders), unit="m"
    )
    estimated = (purchase + pd.to_timedelta(lead_days, unit="D")).normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)
    carrier = purchase + pd.to_timedelta(
        np.clip(approval_hours, 0.5, None), unit="h"
    ) + pd.to_timedelta(
        np.clip(actual_days * 0.35 + rng.normal(0, 1.0, size=n_orders), 0.5, None), unit="D"
    )

    review_score = np.clip(
        np.round(5 - 3.2 * is_late - 0.8 * (actual_days > lead_days + 3) + rng.normal(0, 0.5, size=n_orders)),
        1, 5,
    ).astype(int)

    order_ids = [f"demo_{i:08d}" for i in range(n_orders)]
    customer_ids = [f"cust_{i:08d}" for i in range(n_orders)]
    seller_ids = [f"seller_{i:05d}" for i in range(n_orders)]
    product_ids = [f"prod_{i:07d}" for i in range(n_orders)]

    df = pd.DataFrame(
        {
            "order_id": order_ids,
            "customer_id": customer_ids,
            "order_status": "delivered",
            "order_purchase_timestamp": purchase,
            "order_approved_at": purchase + pd.to_timedelta(approval_hours, unit="h"),
            "order_delivered_carrier_date": carrier,
            "order_delivered_customer_date": delivered,
            "order_estimated_delivery_date": estimated,
            "customer_state": customer_state,
            "customer_city": "demo_city",
            "customer_zip_code_prefix": rng.integers(1000, 99999, size=n_orders),
            "seller_id": seller_ids,
            "seller_state": seller_state,
            "product_id": product_ids,
            "product_category_name": category,
            "price": np.round(price, 2),
            "freight_value": np.round(freight, 2),
            "payment_type": payment_type,
            "payment_installments": payment_installments,
            "payment_value": np.round(payment_value, 2),
            "review_score": review_score,
            "product_weight_g": np.round(weight_g, 1),
            "product_length_cm": np.round(rng.uniform(10, 80, size=n_orders), 1),
            "product_height_cm": np.round(rng.uniform(2, 40, size=n_orders), 1),
            "product_width_cm": np.round(rng.uniform(10, 60, size=n_orders), 1),
            "product_photos_qty": photos,
            "product_description_lenght": description_length,
            "n_items": n_items,
            "n_sellers": np.where(rng.random(n_orders) < 0.03, 2, 1),
            "n_payments": np.where(rng.random(n_orders) < 0.05, 2, 1),
            "n_reviews": 1,
        }
    )
    df["order_approved_at"] = df["order_approved_at"].where(~approval_missing, pd.NaT)
    df.attrs["data_source"] = "DEMO"
    df.attrs["demo_seed"] = seed
    df.attrs["demo_n_orders"] = n_orders
    return df


def write_demo_csv(df: pd.DataFrame, directory) -> None:
    """Persist the demo table (used by ``scripts/make_demo_data.py``)."""
    from pathlib import Path

    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    df.to_csv(path / "demo_orders.csv", index=False)
