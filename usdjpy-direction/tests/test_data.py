import numpy as np
import pandas as pd
import pytest

from usdjpy.data import load_bars, make_target, normalise_download, split_holdout, write_predictions


def _bars(closes, idx=None):
    idx = idx if idx is not None else pd.date_range("2024-01-01", periods=len(closes), freq="h", tz="UTC")
    c = np.asarray(closes, float)
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c}, index=idx)


def test_target_uses_next_close():
    t = make_target(_bars([100, 101, 100.5, 102]))
    assert t["y"].tolist() == [1, 0, 1]            # last bar has no next bar
    assert np.isclose(t["next_logret"].iloc[0], np.log(101 / 100))


def test_target_drops_gap_and_ties():
    idx = pd.DatetimeIndex(["2024-01-05 20:00", "2024-01-05 21:00", "2024-01-07 22:00", "2024-01-07 23:00"], tz="UTC")
    t = make_target(_bars([100, 101, 99, 99], idx))
    # 21:00 -> Sunday 22:00 is a gap (no label); Sunday 22:00 -> 23:00 is a tie (dropped)
    assert list(t.index) == [idx[0]]


def test_split_holdout_is_chronological():
    idx = pd.date_range("2024-01-01", periods=100, freq="h", tz="UTC")
    dev, hold = split_holdout(idx[::-1], 0.2)
    assert len(dev) == 80 and len(hold) == 20 and dev.max() < hold.min()
    with pytest.raises(ValueError):
        split_holdout(idx, 1.5)


def test_normalise_download_multiindex_and_tz():
    idx = pd.date_range("2024-01-01", periods=3, freq="h", tz="Europe/London")
    cols = pd.MultiIndex.from_product([["Open", "High", "Low", "Close", "Adj Close", "Volume"], ["USDJPY=X"]])
    raw = pd.DataFrame(np.ones((3, 6)), index=idx, columns=cols)
    raw = pd.concat([raw, raw.iloc[[-1]]])         # duplicate stamp
    df = normalise_download(raw)
    assert list(df.columns) == ["open", "high", "low", "close"]
    assert str(df.index.tz) == "UTC" and df.index.is_unique and len(df) == 3


def test_cache_roundtrip(bar_cache, small_bars):
    df = load_bars()
    pd.testing.assert_frame_equal(df, small_bars, check_freq=False, check_names=False)


def test_write_predictions(tmp_path):
    idx = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    out = write_predictions(idx, [0.4, 0.6, 0.5], tmp_path / "p.csv", y_true=[0, 1, 1])
    assert out["pred"].tolist() == [0, 1, 0] and (tmp_path / "p.csv").exists()
    with pytest.raises(ValueError):
        write_predictions(idx, [np.nan, 0.1, 0.2], tmp_path / "q.csv")
