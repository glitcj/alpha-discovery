import numpy as np
import pandas as pd
import pytest

from usdjpy.features import (ALL_FEATURES, CALENDAR_FEATURES, BarCleaner, BarFeatureEngineer, feature_columns,
                             make_feature_frame, make_preprocessor)


def test_output_columns(bars):
    f = make_feature_frame(bars)
    assert list(f.columns) == ALL_FEATURES and f.index.equals(bars.index)
    assert f.iloc[200:].notna().all().all()      # warm-up only at the start


def test_no_lookahead(bars):
    """Features at bar t must not change when every bar after t is removed."""
    full = make_feature_frame(bars)
    for k in (150, 400, 1000):
        part = make_feature_frame(bars.iloc[:k])
        pd.testing.assert_frame_equal(part, full.iloc[:k], check_freq=False)


def test_future_shock_does_not_leak(bars):
    shocked = bars.copy()
    shocked.iloc[500:, :] *= 1.2
    a, b = make_feature_frame(bars), make_feature_frame(shocked)
    pd.testing.assert_frame_equal(a.iloc[:500], b.iloc[:500], check_freq=False)


def test_cleaner_drops_bad_bars():
    idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    df = pd.DataFrame({"open": [1, 1, 1, 1, 1.0], "high": [1.1, 0.9, 1.1, 1.1, 1.1],
                       "low": [0.9, 1.0, 0.9, 0.9, 0.9], "close": [1, 1, -1, np.nan, 1.0]}, index=idx)
    out = BarCleaner().fit_transform(df)
    assert list(out.index) == [idx[0], idx[4]]   # high<low, negative, NaN close removed


def test_session_and_gap_features(bars):
    f = make_feature_frame(bars)
    mon = f[(f.index.day_of_week == 0) & (f.index.hour == 0)]
    assert (mon["hours_since_gap"] == 0).all()    # first bar after the weekend
    assert set(f["session_overlap"].unique()) <= {0.0, 1.0}
    assert (f.loc[f.index.hour == 13, "session_overlap"] == 1).all()


def test_preprocessors(bars):
    f = make_feature_frame(bars)
    tree = make_preprocessor("tree", drop=CALENDAR_FEATURES).fit_transform(f)
    assert not set(CALENDAR_FEATURES) & set(tree.columns)
    dense = make_preprocessor("dense").fit_transform(f)
    assert dense.notna().all().all()
    with pytest.raises(ValueError):
        feature_columns(drop=["nope"])
    with pytest.raises(ValueError):
        make_preprocessor("nope")
