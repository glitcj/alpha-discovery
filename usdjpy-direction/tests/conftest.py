import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest


def make_bars(n_days: int = 120, seed: int = 0, vol: float = 0.001, start: str = "2024-01-01") -> pd.DataFrame:
    """Synthetic hourly OHLC: a pure random walk on weekdays only (weekends are gaps)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n_days * 24, freq="h", tz="UTC")
    idx = idx[idx.day_of_week < 5]
    r = rng.normal(0, vol, len(idx))
    close = 150 * np.exp(np.cumsum(r))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0, vol / 2, len(idx))) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close},
                        index=pd.DatetimeIndex(idx, name="timestamp"))


@pytest.fixture(scope="session")
def bars():
    return make_bars()


@pytest.fixture(scope="session")
def small_bars():
    return make_bars(n_days=40, seed=1)


@pytest.fixture
def bar_cache(tmp_path, monkeypatch, small_bars):
    """Point USDJPY_DATA_DIR at a temp folder holding a cached CSV of synthetic bars."""
    from usdjpy.data import cache_path

    monkeypatch.setenv("USDJPY_DATA_DIR", str(tmp_path / "dataset"))
    path = cache_path()
    path.parent.mkdir(parents=True)
    small_bars.to_csv(path, index_label="timestamp")
    return path
