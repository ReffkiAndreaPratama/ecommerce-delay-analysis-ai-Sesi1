"""Run the full leakage / imbalance / validation experiment suite.

This is the single entry point that produces every table and figure used by
the Streamlit app. It is intentionally explicit: all thresholds are selected
on the validation partition, and the test partition is touched once per
configuration.

Usage:
    python scripts/run_experiments.py --source demo
    python scripts/run_experiments.py --source auto --n-demo 15000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    DEMO_N_ORDERS,
    FEATURE_SET_LABELS,
    FIGURE_DIR,
    MODEL_LABELS,
    PREDICTION_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
    SAFE_CATEGORICAL_FEATURES,
    SAFE_NUMERIC_FEATURES,
    SPLIT_LABELS,
    STRATEGY_LABELS,
    SUPPORTED_MODELS,
    SUPPORTED_SPLITS,
    SUPPORTED_STRATEGIES,
    TARGET_COL,
    TIME_COL,
)
from src.data_loader import audit_dataset, load_dataset, save_audit  # noqa: E402
from src.evaluate import (  # noqa: E402
    augment_with_predictions,
    build_summary,
    compute_binary_metrics,
    error_breakdown,
    performance_drop,
    save_json,
    save_metrics_csv,
)
from src.features import build_supervised_frame, feature_matrix  # noqa: E402
from src.preprocessing import split_summary, splitter_for  # noqa: E402
from src.train import (  # noqa: E402
    build_model,
    class_weight_report,
    fit_model,
    predict_proba,
    select_threshold,
    threshold_curve,
)
from src import visualization as viz  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["auto", "demo", "olist"], default="auto")
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--n-demo", type=int, default=DEMO_N_ORDERS)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR)
    parser.add_argument("--models", nargs="+", default=list(SUPPORTED_MODELS), choices=SUPPORTED_MODELS)
    parser.add_argument("--splits", nargs="+", default=list(SUPPORTED_SPLITS), choices=SUPPORTED_SPLITS)
    parser.add_argument("--skip-leaky", action="store_true", help="Run only deployment-safe features")
    return parser.parse_args()


def context_columns(frame: pd.DataFrame) -> list[str]:
    candidates = [
        "order_id",
        TIME_COL,
        "customer_state",
        "seller_state",
        "product_category_name",
        "estimated_delivery_days",
        "price",
        "actual_delivery_days",
        "review_score",
        TARGET_COL,
    ]
    return [col for col in candidates if col in frame.columns]


def main() -> int:
    args = parse_args()
    output = args.output
    figure_dir = output / "figures"
    prediction_dir = output / "predictions"
    output.mkdir(parents=True, exist_ok=True)

    raw_dir = args.raw_dir if args.raw_dir is not None else ROOT / "data" / "raw"
    df, meta = load_dataset(source=args.source, raw_dir=raw_dir, n_demo=args.n_demo, seed=args.seed)
    audit = audit_dataset(df)
    save_audit(audit, output / "dataset_audit.json")
    print(f"Dataset: {meta['data_source']} | {len(df):,} labelled orders | late rate {audit['late_rate']:.3%}")

    supervised = build_supervised_frame(df)
    feature_sets = ["safe"] if args.skip_leaky else ["safe", "leaky"]

    records: list[dict] = []
    best: dict | None = None

    for split_type in args.splits:
        train, val, test = splitter_for(split_type)(supervised)
        splits = split_summary(train, val, test)
        save_json(splits, output / f"split_summary_{split_type}.json")
        if split_type == "temporal":
            viz.plot_split_timeline(train, val, test, figure_dir / "split_timeline.png")

        for feature_set in feature_sets:
            X_train, y_train = feature_matrix(train, feature_set)
            X_val, y_val = feature_matrix(val, feature_set)
            X_test, y_test = feature_matrix(test, feature_set)
            balance = class_weight_report(y_train)

            for model_name in args.models:
                for strategy in SUPPORTED_STRATEGIES:
                    config = {
                        "feature_set": feature_set,
                        "split": split_type,
                        "model": model_name,
                        "strategy": strategy,
                    }
                    print(
                        f"[{split_type}] {feature_set:5s} | {model_name:20s} | {strategy:12s}",
                        flush=True,
                    )
                    pipeline = build_model(
                        model_name, strategy=strategy, feature_set=feature_set, random_state=args.seed
                    )
                    pipeline, fit_seconds = fit_model(pipeline, X_train, y_train)

                    val_prob = predict_proba(pipeline, X_val)
                    selection = select_threshold(y_val, val_prob)
                    val_metrics = compute_binary_metrics(y_val, val_prob, selection["threshold"])

                    test_prob = predict_proba(pipeline, X_test)
                    test_metrics = compute_binary_metrics(y_test, test_prob, selection["threshold"])

                    record = {
                        **config,
                        "data_source": meta["data_source"],
                        "n_train": int(len(X_train)),
                        "n_validation": int(len(X_val)),
                        "n_test": int(len(X_test)),
                        "train_late_rate": balance["positive_rate"],
                        "imbalance_ratio": balance["imbalance_ratio"],
                        "threshold": selection["threshold"],
                        "fit_seconds": round(fit_seconds, 3),
                        "val_pr_auc": val_metrics["pr_auc"],
                        "val_f1": val_metrics["f1"],
                        "val_precision": val_metrics["precision"],
                        "val_recall": val_metrics["recall"],
                        **{k: v for k, v in test_metrics.items() if k not in {"threshold"}},
                    }
                    records.append(record)

                    if feature_set == "safe":
                        rank = (val_metrics["pr_auc"] or -1.0, val_metrics["f1"])
                        if best is None or rank > best["rank"]:
                            best = {
                                "rank": rank,
                                "record": record,
                                "pipeline": pipeline,
                                "val_prob": val_prob,
                                "test_prob": test_prob,
                                "test_frame": test,
                                "y_val": y_val,
                                "y_test": y_test,
                            }

    results = pd.DataFrame(records).sort_values(
        ["feature_set", "split", "model", "strategy"]
    ).reset_index(drop=True)
    save_metrics_csv(records, output / "metrics.csv")

    if best is not None:
        pred = augment_with_predictions(
            best["test_frame"][context_columns(best["test_frame"])],
            best["test_prob"],
            best["record"]["threshold"],
        )
        prediction_dir.mkdir(parents=True, exist_ok=True)
        pred.to_csv(prediction_dir / "best_model_test_predictions.csv", index=False)
        save_json(best["record"], output / "best_model.json")

        curve = pd.DataFrame(threshold_curve(best["y_val"], best["val_prob"]))
        viz.plot_threshold_curve(curve, figure_dir / "threshold_curve.png", best["record"]["threshold"])
        viz.plot_pr_curve(best["y_test"], best["test_prob"], figure_dir / "pr_curve_best.png")
        viz.plot_calibration(best["y_test"], best["test_prob"], figure_dir / "calibration_best.png")
        try:
            by_state = error_breakdown(pred, by="customer_state")
            viz.plot_error_by_state(by_state, figure_dir / "error_by_state.png")
            by_state.to_csv(output / "error_by_state.csv", index=False)
        except ValueError:
            pass

    viz.plot_class_distribution(supervised, figure_dir / "class_distribution.png")
    viz.plot_missingness(df, figure_dir / "missingness.png")
    viz.plot_model_comparison(results, figure_dir / "model_comparison.png")
    viz.plot_validation_comparison(results, figure_dir / "validation_comparison.png")
    if (results["feature_set"] == "leaky").any():
        viz.plot_leakage_comparison(results, figure_dir / "leakage_comparison.png")

    summary = build_summary(results)
    summary["dataset"] = audit
    summary["split_definitions"] = {
        "random": "stratified 70/10/20",
        "temporal": "chronological 70/10/20 (oldest train, newest test)",
    }
    summary["labels"] = {
        "models": MODEL_LABELS,
        "strategies": STRATEGY_LABELS,
        "splits": SPLIT_LABELS,
        "feature_sets": FEATURE_SET_LABELS,
    }
    save_json(summary, output / "summary.json")

    try:
        drops = performance_drop(results, metric="pr_auc")
        drops.to_csv(output / "temporal_degradation.csv", index=False)
    except ValueError:
        pass

    print(f"Saved {len(results)} configurations to {output / 'metrics.csv'}")
    print(f"Figures: {figure_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
