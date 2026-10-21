"""Streamlit dashboard for the e-commerce late-delivery study.

Run with:
    streamlit run app.py

The dashboard works without any experiment output: it shows the dataset audit
and the exact command needed to generate metrics. Demo data is always labelled
DEMO so synthetic findings cannot be mistaken for real research results.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import (
    FEATURE_SET_LABELS,
    FIGURE_DIR,
    MODEL_LABELS,
    PROJECT_ROOT,
    RESULTS_DIR,
    SPLIT_LABELS,
    STRATEGY_LABELS,
    TARGET_COL,
    TIME_COL,
)
from src.data_loader import audit_dataset, load_dataset

st.set_page_config(
    page_title="E-commerce Delay Analysis",
    page_icon="📦",
    layout="wide",
)

PAGES = [
    "Ringkasan dataset",
    "Kualitas data",
    "Distribusi kelas",
    "Perbandingan model",
    "Perbandingan validasi",
    "Analisis error & keterbatasan",
    "Unduh hasil",
]


@st.cache_data(show_spinner="Memuat dataset...")
def get_dataset(source: str, n_demo: int):
    df, meta = load_dataset(source=source, n_demo=n_demo)
    return df, meta, audit_dataset(df)


@st.cache_data(ttl=30, show_spinner=False)
def get_results() -> pd.DataFrame | None:
    path = RESULTS_DIR / "metrics.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data(ttl=30, show_spinner=False)
def get_json(name: str) -> dict | None:
    path = RESULTS_DIR / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(ttl=30, show_spinner=False)
def get_csv(path_str: str) -> pd.DataFrame | None:
    path = Path(path_str)
    if not path.exists():
        return None
    return pd.read_csv(path)


def figure(name: str, caption: str) -> None:
    path = FIGURE_DIR / name
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Gambar `{name}` belum tersedia. Jalankan `python scripts/run_experiments.py`.")


def metric_cards(audit: dict, meta: dict) -> None:
    cols = st.columns(4)
    cols[0].metric("Pesanan berlabel", f"{audit['n_rows']:,}")
    cols[1].metric("Tingkat keterlambatan", f"{audit['late_rate']:.2%}")
    cols[2].metric("Pelanggan unik", f"{audit['n_unique_customers']:,}")
    cols[3].metric("Sumber data", meta["data_source"])


def results_table(results: pd.DataFrame) -> pd.DataFrame:
    frame = results.copy()
    frame["model"] = frame["model"].map(lambda v: MODEL_LABELS.get(v, v))
    frame["strategy"] = frame["strategy"].map(lambda v: STRATEGY_LABELS.get(v, v))
    frame["split"] = frame["split"].map(lambda v: SPLIT_LABELS.get(v, v))
    frame["feature_set"] = frame["feature_set"].map(lambda v: FEATURE_SET_LABELS.get(v, v))
    columns = [
        "feature_set", "split", "model", "strategy", "threshold",
        "val_pr_auc", "precision", "recall", "f1", "pr_auc", "roc_auc",
        "brier", "predicted_positive_rate", "n_test", "fit_seconds",
    ]
    return frame[[c for c in columns if c in frame.columns]]


def page_overview(df: pd.DataFrame, meta: dict, audit: dict) -> None:
    st.title("📦 Prediksi Keterlambatan Pengiriman E-Commerce")
    st.caption(
        "Evaluasi pengaruh data leakage, ketidakseimbangan kelas, dan validasi temporal "
        "terhadap keandalan model."
    )
    if meta["data_source"] == "DEMO":
        st.warning(
            "**DATA DEMO SINTETIS.** Seluruh angka pada mode ini berasal dari data buatan "
            "(`src/demo_data.py`) dan tidak boleh dilaporkan sebagai temuan penelitian nyata."
        )
    else:
        st.success("Menggunakan dataset Olist asli dari `data/raw/`.")

    metric_cards(audit, meta)
    left, right = st.columns([1.4, 1])
    with left:
        st.subheader("Definisi target")
        st.code(audit["target_definition"], language="sql")
        st.write(
            "Model memprediksi apakah pesanan akan tiba setelah tanggal estimasi. "
            "Fitur yang hanya diketahui setelah pengiriman tidak digunakan pada set fitur aman."
        )
        st.subheader("Periode data")
        st.write(f"Pembelian: **{audit['date_min'][:10]}** sampai **{audit['date_max'][:10]}**")
        st.write(f"Duplikasi `order_id`: **{audit['n_duplicate_order_ids']}**")
    with right:
        st.subheader("Cakupan data")
        st.dataframe(
            pd.DataFrame(
                {
                    "Metrik": ["Baris", "Kolom", "Pelanggan", "Penjual", "Terlambat", "Tepat waktu"],
                    "Nilai": [
                        audit["n_rows"], audit["n_columns"], audit["n_unique_customers"],
                        audit["n_unique_sellers"], audit["n_late"], audit["n_on_time"],
                    ],
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

    results = get_results()
    if results is None:
        st.subheader("Mulai eksperimen")
        st.code(
            "python scripts/run_experiments.py --source demo\n"
            "streamlit run app.py",
            language="bash",
        )
        st.info(
            "Belum ada `results/metrics.csv`. Dashboard tetap dapat menampilkan audit data; "
            "jalankan perintah di atas untuk menghasilkan metrik dan gambar."
        )
    else:
        st.subheader("Hasil terbaru")
        summary = get_json("summary.json") or {}
        best = summary.get("best_safe_by_validation_pr_auc")
        if best:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Model terbaik (val)", MODEL_LABELS.get(best["model"], best["model"]))
            c2.metric("Test PR-AUC", f"{best['test_pr_auc']:.3f}")
            c3.metric("Test F1", f"{best['test_f1']:.3f}")
            c4.metric("Threshold", f"{best['threshold']:.2f}")
        st.caption("Model dipilih berdasarkan PR-AUC pada validation set, bukan test set.")


def page_quality(df: pd.DataFrame, audit: dict) -> None:
    st.title("Kualitas data")
    st.write(
        "Audit mencakup nilai kosong, duplikasi, timestamp yang tidak mungkin, "
        "dan label kategori yang tidak diketahui."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Sel kosong", f"{audit['n_missing_cells']:,}")
    c2.metric("Kategori tidak diketahui", f"{audit['n_missing_category']:,}")
    c3.metric("Estimasi sebelum pembelian", f"{audit['n_estimated_before_purchase']:,}")

    missing = pd.DataFrame(
        {"Kolom": list(audit["missing_values"].keys()), "Jumlah kosong": list(audit["missing_values"].values())}
    ).sort_values("Jumlah kosong", ascending=False)
    st.subheader("Nilai kosong per kolom")
    if missing.empty:
        st.success("Tidak ada nilai kosong pada kolom mana pun.")
    else:
        missing["Persentase"] = (missing["Jumlah kosong"] / len(df) * 100).round(2)
        st.dataframe(missing, hide_index=True, use_container_width=True)
    figure("missingness.png", "Kolom dengan nilai kosong terbanyak")

    st.subheader("Duplikasi dan konsistensi")
    st.write(f"Baris dengan `order_id` duplikat: **{audit['n_duplicate_order_ids']}**")
    st.write(f"Pengiriman sebelum pembelian: **{audit['n_delivered_before_purchase']}**")
    st.caption(
        "Baris tanpa tanggal pengiriman/estimasi atau dengan urutan waktu mustahil "
        "dibuang sebelum pemodelan; jumlahnya dicatat di `dataset_audit.json`."
    )


def page_class(df: pd.DataFrame) -> None:
    st.title("Distribusi kelas")
    counts = df[TARGET_COL].astype(int).value_counts().reindex([0, 1], fill_value=0)
    total = int(counts.sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Tepat waktu", f"{counts.loc[0]:,}", f"{counts.loc[0] / total:.1%}")
    c2.metric("Terlambat", f"{counts.loc[1]:,}", f"{counts.loc[1] / total:.1%}")
    c3.metric("Rasio imbalance", f"{counts.loc[0] / max(counts.loc[1], 1):.1f} : 1")

    figure("class_distribution.png", "Distribusi kelas target")

    by_month = (
        df.assign(month=pd.to_datetime(df[TIME_COL]).dt.to_period("M").astype(str))
        .groupby("month")[TARGET_COL]
        .agg(["mean", "size"])
        .reset_index()
        .rename(columns={"mean": "late_rate", "size": "orders"})
    )
    st.subheader("Tingkat keterlambatan per bulan")
    st.bar_chart(by_month.set_index("month")["late_rate"])
    st.caption("Garis waktu penting untuk menilai apakah random split realistis.")

    by_state = (
        df.groupby("customer_state")[TARGET_COL].agg(["mean", "size"]).sort_values("size", ascending=False).head(15)
    )
    st.subheader("Tingkat keterlambatan per negara bagian (top 15 volume)")
    st.dataframe(
        by_state.rename(columns={"mean": "late_rate", "size": "orders"}).style.format({"late_rate": "{:.2%}"}),
        use_container_width=True,
    )


def page_models(results: pd.DataFrame) -> None:
    st.title("Perbandingan model")
    feature_label = st.selectbox("Set fitur", ["safe", "leaky"], format_func=lambda v: FEATURE_SET_LABELS[v])
    subset = results[results["feature_set"] == feature_label]
    if subset.empty:
        st.warning("Tidak ada hasil untuk pilihan ini.")
        return
    st.dataframe(results_table(subset), hide_index=True, use_container_width=True)
    metric = st.selectbox("Metrik", ["pr_auc", "f1", "precision", "recall", "roc_auc"], index=0)
    pivot = subset.pivot_table(
        index=["model", "strategy"], columns="split", values=metric, aggfunc="mean"
    )
    pivot.index = [f"{MODEL_LABELS.get(m, m)} / {STRATEGY_LABELS.get(s, s)}" for m, s in pivot.index]
    st.subheader(f"{metric} berdasarkan model dan strategi")
    st.bar_chart(pivot)
    figure("model_comparison.png", "Perbandingan model pada fitur aman")
    st.caption(
        "Class weighting hanya diterapkan pada data training. Threshold dipilih pada validation set."
    )


def page_validation(results: pd.DataFrame) -> None:
    st.title("Perbandingan strategi validasi")
    st.write(
        "Random split memberi gambaran optimistis karena pesanan dari periode yang sama "
        "bisa muncul di train dan test. Temporal split menguji pengiriman di masa depan."
    )
    safe = results[results["feature_set"] == "safe"]
    pivot = safe.pivot_table(index=["model", "strategy"], columns="split", values="pr_auc", aggfunc="mean")
    if not pivot.empty:
        pivot.index = [f"{MODEL_LABELS.get(m, m)} / {STRATEGY_LABELS.get(s, s)}" for m, s in pivot.index]
        st.subheader("PR-AUC: random vs temporal")
        st.bar_chart(pivot)
        if {"random", "temporal"}.issubset(pivot.columns):
            pivot["absolute_drop"] = pivot["random"] - pivot["temporal"]
            st.dataframe(pivot.style.format("{:.3f}"), use_container_width=True)
    figure("validation_comparison.png", "Sebaran PR-AUC per strategi validasi")
    figure("leakage_comparison.png", "Perbandingan fitur aman vs fitur bocor")
    st.warning(
        "Perbedaan performa adalah deskripsi hasil eksperimen, bukan bukti kausal. "
        "Ukuran sampel dan periode test tunggal membatasi generalisasi."
    )


def page_errors() -> None:
    st.title("Analisis error & keterbatasan")
    best = get_json("best_model.json")
    pred = get_csv(str(RESULTS_DIR / "predictions" / "best_model_test_predictions.csv"))
    if best is None or pred is None:
        st.info("Jalankan eksperimen untuk menghasilkan analisis error.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Model", MODEL_LABELS.get(best["model"], best["model"]))
        c2.metric("PR-AUC test", f"{best['pr_auc']:.3f}")
        c3.metric("Recall", f"{best['recall']:.3f}")
        c4.metric("Threshold", f"{best['threshold']:.2f}")
        errors = pred[pred["error_type"] != "correct"]
        st.write(f"Jumlah kesalahan pada test set: **{len(errors):,}** dari **{len(pred):,}** pesanan.")
        by_state = get_csv(str(RESULTS_DIR / "error_by_state.csv"))
        if by_state is not None:
            st.subheader("Kesalahan per negara bagian")
            st.dataframe(
                by_state.style.format(
                    {"actual_late_rate": "{:.2%}", "predicted_late_rate": "{:.2%}",
                     "false_negative_rate": "{:.2%}", "false_positive_rate": "{:.2%}"}
                ),
                use_container_width=True,
            )
            figure("error_by_state.png", "False positive dan false negative per negara bagian")
        st.subheader("Contoh kesalahan dengan probabilitas tertinggi")
        columns = [c for c in ["order_id", "customer_state", "predicted_probability", "actual_late", "error_type"] if c in pred.columns]
        st.dataframe(errors.sort_values("predicted_probability", ascending=False).head(20)[columns], hide_index=True)

    st.subheader("Keterbatasan yang harus dinyatakan")
    st.markdown(
        "- Target didefinisikan dari tanggal pengiriman vs estimasi; keterlambatan operasional "
        "tidak otomatis berarti pelanggaran SLA.\n"
        "- Random split dan temporal split menjawab pertanyaan berbeda dan tidak boleh disamakan.\n"
        "- Satu periode test tidak cukup untuk menyimpulkan stabilitas musiman.\n"
        "- Fitur aman tetap bergantung pada kualitas estimasi dan data katalog.\n"
        "- Mode DEMO menghasilkan angka sintetis yang tidak boleh dikutip sebagai hasil nyata."
    )


def page_downloads(results: pd.DataFrame | None, audit: dict, meta: dict) -> None:
    st.title("Unduh hasil")
    st.write("Semua unduhan berasal dari artefak yang benar-benar dihasilkan di folder `results/`.")
    if results is not None:
        st.download_button(
            "metrics.csv",
            results.to_csv(index=False).encode("utf-8"),
            "metrics.csv",
            "text/csv",
            use_container_width=True,
        )
    for name in ["summary.json", "dataset_audit.json", "best_model.json", "temporal_degradation.csv"]:
        path = RESULTS_DIR / name
        if path.exists():
            st.download_button(
                name,
                path.read_bytes(),
                name,
                "application/json" if name.endswith(".json") else "text/csv",
                use_container_width=True,
            )
    pred_path = RESULTS_DIR / "predictions" / "best_model_test_predictions.csv"
    if pred_path.exists():
        st.download_button(
            "best_model_test_predictions.csv",
            pred_path.read_bytes(),
            "best_model_test_predictions.csv",
            "text/csv",
            use_container_width=True,
        )
    st.download_button(
        "dataset_audit_live.json",
        json.dumps(audit, indent=2).encode("utf-8"),
        "dataset_audit_live.json",
        "application/json",
        use_container_width=True,
    )
    st.caption(f"Data source: {meta['data_source']}. {meta['warning']}")


def main() -> None:
    with st.sidebar:
        st.header("Navigasi")
        page = st.radio("Halaman", PAGES)
        st.divider()
        source = st.selectbox(
            "Sumber data",
            ["demo", "auto", "olist"],
            help="auto memakai Olist jika file tersedia, selain itu DEMO.",
        )
        n_demo = st.number_input("Jumlah pesanan DEMO", min_value=500, max_value=50000, value=15000, step=500)
        st.caption(f"Root: `{PROJECT_ROOT.name}`")

    try:
        df, meta, audit = get_dataset(source, int(n_demo))
    except Exception as exc:  # noqa: BLE001 - surface a useful message in the UI
        st.error(f"Gagal memuat dataset: {exc}")
        st.stop()

    results = get_results()
    if page == "Ringkasan dataset":
        page_overview(df, meta, audit)
    elif page == "Kualitas data":
        page_quality(df, audit)
    elif page == "Distribusi kelas":
        page_class(df)
    elif page == "Perbandingan model":
        if results is None:
            st.info("Belum ada hasil. Jalankan `python scripts/run_experiments.py --source demo`.")
        else:
            page_models(results)
    elif page == "Perbandingan validasi":
        if results is None:
            st.info("Belum ada hasil. Jalankan `python scripts/run_experiments.py --source demo`.")
        else:
            page_validation(results)
    elif page == "Analisis error & keterbatasan":
        page_errors()
    else:
        page_downloads(results, audit, meta)


if __name__ == "__main__":
    main()
