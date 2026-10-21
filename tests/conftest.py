from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import clean_orders  # noqa: E402
from src.demo_data import generate_demo_orders  # noqa: E402


@pytest.fixture(scope="session")
def demo_raw():
    return generate_demo_orders(n_orders=900, seed=7)


@pytest.fixture(scope="session")
def demo_df(demo_raw):
    return clean_orders(demo_raw)
