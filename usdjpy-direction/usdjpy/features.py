"""sklearn transformers for hourly bars.

``BarCleaner``           numeric OHLC, drop impossible bars (non-positive, high < low, NaN close)
``BarFeatureEngineer``   bars -> model features. **Every feature at bar t uses only bars <= t**
                         (lags, rolling windows, EWMs, calendar), so it is computed once on the
                         full contiguous series *before* any split; ``tests/test_features.py``
                         checks this by truncating the future and comparing.
``make_preprocessor``    ColumnTransformer for tree models (``"tree"``: passthrough, NaN kept)
                         or linear models (``"dense"``: median-impute + standardise)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from .data import BAR, OHLC

# ---------------------------------------------------------------------------
# feature groups
# ---------------------------------------------------------------------------
RET_LAGS = (1, 2, 3, 4, 5, 6, 12, 24)
WINDOWS = (6, 24, 120)
SMA_WINDOWS = (24, 120)
GAP_CAP = 120

RETURN_FEATURES = [f"ret_lag{k}" for k in RET_LAGS]
ROLLING_FEATURES = ([f"ret_mean_{w}" for w in WINDOWS] + [f"ret_std_{w}" for w in WINDOWS]
                    + [f"z_sma_{w}" for w in SMA_WINDOWS] + ["rsi_14", "atr_14", "vol_ratio_6_120"])
CANDLE_FEATURES = ["range_hl", "body", "upper_wick", "lower_wick"]
CALENDAR_FEATURES = ["hour", "hour_sin", "hour_cos", "weekday", "session_tokyo", "session_london",
                     "session_ny", "session_overlap", "hours_since_gap"]
ALL_FEATURES = RETURN_FEATURES + ROLLING_FEATURES + CANDLE_FEATURES + CALENDAR_FEATURES

# trading sessions in UTC hours [start, end) — fixed, ignoring daylight saving
SESSIONS = {"tokyo": (0, 9), "london": (7, 16), "ny": (12, 21)}


class _FrameTransformer(TransformerMixin, BaseEstimator):
    """Base for DataFrame-in / DataFrame-out transformers with sklearn plumbing."""

    def _check_frame(self, X) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"{type(self).__name__} expects a pandas DataFrame, got {type(X).__name__}")
        return X

    def fit(self, X, y=None):
        X = self._check_frame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.feature_names_out_ = np.asarray(self._output_columns(X), dtype=object)
        return self

    def _output_columns(self, X: pd.DataFrame) -> list[str]:  # pragma: no cover - abstract
        raise NotImplementedError

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.no_validation = True
        return tags


# ---------------------------------------------------------------------------
# stage 1: cleaning
# ---------------------------------------------------------------------------
class BarCleaner(_FrameTransformer):
    """Numeric OHLC on a sorted, unique ``DatetimeIndex``; impossible bars removed.

    Parameters
    ----------
    max_abs_logret : float
        A close-to-close log move above this (default 5%) in one hour is treated as a bad
        print and the bar is dropped. USDJPY has never moved that much in an hour.
    """

    def __init__(self, max_abs_logret: float = 0.05):
        self.max_abs_logret = max_abs_logret

    def _output_columns(self, X):
        return list(OHLC)

    def transform(self, X):
        check_is_fitted(self, "feature_names_in_")
        X = self._check_frame(X)
        if not isinstance(X.index, pd.DatetimeIndex):
            raise TypeError("BarCleaner expects a DatetimeIndex")
        df = X[OHLC].apply(pd.to_numeric, errors="coerce").astype(float)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        ok = df["close"].notna() & (df[OHLC].fillna(1.0) > 0).all(axis=1)
        ok &= ~(df["high"] < df["low"])
        df = df[ok]
        # fill missing open/high/low from close so candle features stay defined
        for c in ("open", "high", "low"):
            df[c] = df[c].fillna(df["close"])
        jump = np.log(df["close"]).diff().abs()
        return df[~(jump > self.max_abs_logret)]


# ---------------------------------------------------------------------------
# stage 2: feature engineering
# ---------------------------------------------------------------------------
def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def session_flags(index: pd.DatetimeIndex) -> pd.DataFrame:
    hour = index.hour
    out = pd.DataFrame(index=index)
    for name, (a, b) in SESSIONS.items():
        out[f"session_{name}"] = ((hour >= a) & (hour < b)).astype(float)
    out["session_overlap"] = out["session_london"] * out["session_ny"]
    return out


class BarFeatureEngineer(_FrameTransformer):
    """Backward-looking features from cleaned bars. Output columns are ``ALL_FEATURES``.

    Rolling windows count bars, not wall-clock hours, so a weekend gap is "invisible" to
    them; ``hours_since_gap`` tells the model how many bars ago the last gap was.
    """

    def _output_columns(self, X):
        return list(ALL_FEATURES)

    def transform(self, X):
        check_is_fitted(self, "feature_names_in_")
        X = self._check_frame(X)
        o, h, l, c = (X[k].astype(float) for k in OHLC)
        r = np.log(c / c.shift(1))
        f = pd.DataFrame(index=X.index)

        for k in RET_LAGS:
            f[f"ret_lag{k}"] = r.shift(k - 1)            # ret_lag1 = the return *into* bar t
        for w in WINDOWS:
            f[f"ret_mean_{w}"] = r.rolling(w, min_periods=w).mean()
        for w in WINDOWS:
            f[f"ret_std_{w}"] = r.rolling(w, min_periods=w).std()
        for w in SMA_WINDOWS:
            sma = c.rolling(w, min_periods=w).mean()
            sd = c.rolling(w, min_periods=w).std()
            f[f"z_sma_{w}"] = (c - sma) / sd.replace(0, np.nan)
        f["rsi_14"] = _rsi(c, 14)
        prev_c = c.shift(1)
        tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
        f["atr_14"] = (tr / c).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        f["vol_ratio_6_120"] = f["ret_std_6"] / f["ret_std_120"].replace(0, np.nan)

        hl = (h - l).replace(0, np.nan)
        f["range_hl"] = (h - l) / c
        f["body"] = (c - o) / hl
        f["upper_wick"] = (h - np.maximum(o, c)) / hl
        f["lower_wick"] = (np.minimum(o, c) - l) / hl

        idx = pd.DatetimeIndex(X.index)
        hour = idx.hour.to_numpy(dtype=float)
        f["hour"] = hour
        f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        f["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        f["weekday"] = idx.day_of_week.to_numpy(dtype=float)
        f = f.join(session_flags(idx))
        gap = (idx.to_series().diff() != BAR).to_numpy()   # first bar counts as a gap
        block = np.cumsum(gap)
        since = pd.Series(np.arange(len(idx)), index=X.index).groupby(block).cumcount()
        f["hours_since_gap"] = np.minimum(since.to_numpy(), GAP_CAP).astype(float)
        return f[ALL_FEATURES]


# ---------------------------------------------------------------------------
# stage 3: encoding
# ---------------------------------------------------------------------------
def feature_columns(drop=()) -> list[str]:
    drop = set(drop)
    unknown = drop - set(ALL_FEATURES)
    if unknown:
        raise ValueError(f"unknown feature(s) to drop: {sorted(unknown)}")
    return [c for c in ALL_FEATURES if c not in drop]


def make_preprocessor(kind: str = "tree", drop=()) -> ColumnTransformer:
    """Select ``ALL_FEATURES`` minus ``drop`` and encode them for a model family.

    ``"tree"``   passthrough; NaN kept (LightGBM handles missing values natively).
    ``"dense"``  median-impute + standardise (logistic regression).
    """
    cols = feature_columns(drop)
    if kind == "tree":
        step = "passthrough"
    elif kind == "dense":
        step = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    else:
        raise ValueError(f"unknown preprocessor kind {kind!r}; use 'tree' or 'dense'")
    ct = ColumnTransformer([("num", step, cols)], remainder="drop", verbose_feature_names_out=False)
    ct.set_output(transform="pandas")
    return ct


def make_feature_frame(bars: pd.DataFrame, cleaner: BarCleaner | None = None) -> pd.DataFrame:
    """bars -> cleaned -> features (convenience for notebooks and ``--predict-only``)."""
    cleaned = (cleaner or BarCleaner()).fit_transform(bars)
    return BarFeatureEngineer().fit_transform(cleaned)
