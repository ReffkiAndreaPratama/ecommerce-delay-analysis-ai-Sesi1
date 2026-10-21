"""Generate the explicitly-labelled synthetic demo dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DEMO_DATA_DIR, DEMO_N_ORDERS, RANDOM_STATE  # noqa: E402
from src.data_loader import audit_dataset, clean_orders  # noqa: E402
from src.demo_data import generate_demo_orders, write_demo_csv  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-orders", type=int, default=DEMO_N_ORDERS)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--output", type=Path, default=DEMO_DATA_DIR)
    args = parser.parse_args()

    raw = generate_demo_orders(n_orders=args.n_orders, seed=args.seed)
    cleaned = clean_orders(raw)
    write_demo_csv(cleaned, args.output)
    audit = audit_dataset(cleaned)
    print(f"Wrote {args.output / 'demo_orders.csv'}")
    print(f"Rows: {audit['n_rows']:,} | late rate: {audit['late_rate']:.2%} | source: DEMO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
