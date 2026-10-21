# Data sources and licensing

## Primary dataset: Olist Brazilian E-Commerce Public Dataset

- **Source:** Kaggle, published by Olist
- **Canonical page:** https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
- **Licence:** CC BY-NC-SA 4.0 (non-commercial, share-alike). Check the Kaggle page for the current terms.
- **Period covered:** orders from September 2016 to October 2018 (the exact range depends on the release).
- **Granularity:** one row per order item in `olist_order_items_dataset.csv`; orders, payments, reviews, customers, sellers, and products are separate tables.
- **Size:** approximately 100k orders and 112k order items across nine CSVs. The dataset is not committed to this repository.

The data is **not** downloaded automatically, because Kaggle requires an
account and acceptance of the dataset terms. Use one of these options:

1. Download the archive manually from the Kaggle page and extract every CSV
   into `data/raw/`.
2. Install the Kaggle CLI (`pip install kaggle`), configure
   `~/.kaggle/kaggle.json`, and run:

   ```bash
   python scripts/download_data.py --download
   ```

3. Verify an existing download:

   ```bash
   python scripts/download_data.py --check
   ```

Required filenames are listed in `src/config.py` (`OLIST_FILES`). They match the
official Kaggle archive.

## Target definition

`is_late` is defined as:

```
date(order_delivered_customer_date) > date(order_estimated_delivery_date)
```

The comparison is at calendar-date granularity because the marketplace
publishes an estimated **date**, not a time. Rows without a purchase timestamp,
estimate, or delivery timestamp are excluded. Chronologically impossible rows
(delivery or estimate before purchase) are also excluded and counted in the
audit.

This target measures a deviation from the marketplace's own estimate. It is
not an operational SLA and does not capture every form of customer
dissatisfaction.

## DEMO mode

`src/demo_data.py` generates a synthetic, Olist-shaped table. It is clearly
labelled `DEMO` in:

- `load_dataset()` metadata (`data_source`, `warning`),
- the Streamlit banner,
- `results/summary.json`,
- every metrics row (`data_source` column).

DEMO observations are fabricated. Any metric produced from them is a software
smoke test, not evidence about Olist or about real delivery performance.

## Leakage policy

The following are known only after delivery and are excluded from the `safe`
feature set:

- `order_delivered_customer_date`
- `order_delivered_carrier_date`
- `order_approved_at` (raw timestamp; a derived approval delay is included
  because it is observable shortly after checkout)
- `order_status`
- `review_score`

The `leaky` feature set deliberately includes post-delivery variables to
demonstrate optimistic bias. It must never be presented as a deployable model.

## Known limitations

- The dataset covers one marketplace and one historical period.
- Timestamps are self-reported by the marketplace and may contain errors.
- Product and seller attributes change over time; the latest snapshot is used.
- The Olist licence forbids commercial reuse without permission.
- The dataset may contain personal or quasi-personal information; do not
  attempt re-identification.
