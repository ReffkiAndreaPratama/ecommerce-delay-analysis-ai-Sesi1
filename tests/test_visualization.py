from __future__ import annotations

import numpy as np
import pandas as pd

from src import visualization as viz
from src.preprocessing import split_temporal


def test_class_distribution_figure(tmp_path, demo_df):
    path = viz.plot_class_distribution(demo_df, tmp_path / "classes.png")
    assert path.exists() and path.stat().st_size > 0


def test_missingness_figure(tmp_path, demo_df):
    frame = demo_df.copy()
    frame.loc[frame.index[:5], "review_score"] = np.nan
    path = viz.plot_missingness(frame, tmp_path / "missing.png")
    assert path.exists()


def test_split_timeline_figure(tmp_path, demo_df):
    train, val, test = split_temporal(demo_df)
    path = viz.plot_split_timeline(train, val, test, tmp_path / "timeline.png")
    assert path.exists()


def test_pr_and_calibration_figures(tmp_path):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    p = np.clip(0.2 + 0.5 * y + rng.normal(0, 0.2, size=200), 0, 1)
    assert viz.plot_pr_curve(y, p, tmp_path / "pr.png").exists()
    assert viz.plot_calibration(y, p, tmp_path / "cal.png").exists()
