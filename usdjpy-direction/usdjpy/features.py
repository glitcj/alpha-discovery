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
    regular_grid : bool
        Reindex to every hour of the FX week (``fx_week_mask``: Sunday 17:00 → Friday 17:00
        New York time) and forward-fill missing hours (OHLC = last close), the way ml-pipe
        resamples onto ``week_range``. Positional shifts then mean fixed clock offsets within a
        week, and weekend hours are absent rather than gaps.
    """

    def __init__(self, max_abs_logret: float = 0.05, regular_grid: bool = False):
        self.max_abs_logret = max_abs_logret
        self.regular_grid = regular_grid

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
        df = df[~(jump > self.max_abs_logret)]
        return to_fx_week_grid(df) if self.regular_grid else df


FX_TZ = "America/New_York"


def fx_week_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """True for hours inside the FX week: Sunday 17:00 → Friday 16:59 New York time (DST-aware)."""
    ny = pd.DatetimeIndex(index).tz_convert(FX_TZ)
    dow, hour = ny.day_of_week, ny.hour            # Monday = 0 ... Sunday = 6
    return np.asarray(((dow >= 0) & (dow <= 3)) | ((dow == 4) & (hour < 17)) | ((dow == 6) & (hour >= 17)))


def to_fx_week_grid(bars: pd.DataFrame) -> pd.DataFrame:
    """Reindex to every hour of the FX week between the first and last bar; forward-fill gaps."""
    if bars.empty:
        return bars
    full = pd.date_range(bars.index[0], bars.index[-1], freq=BAR, tz="UTC", name=bars.index.name)
    grid = full[fx_week_mask(full)]
    out = bars[fx_week_mask(bars.index)].reindex(grid)
    close = out["close"].ffill()
    missing = out["close"].isna()
    for c in ("open", "high", "low"):
        out[c] = out[c].where(~missing, close)
    out["close"] = close
    return out.dropna(subset=["close"])


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

        f = f.join(calendar_features(pd.DatetimeIndex(X.index)))
        return f[ALL_FEATURES]


def calendar_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """``CALENDAR_FEATURES`` for a UTC index (hour, weekday, sessions, bars since the last gap)."""
    f = pd.DataFrame(index=idx)
    hour = idx.hour.to_numpy(dtype=float)
    f["hour"] = hour
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    f["weekday"] = idx.day_of_week.to_numpy(dtype=float)
    f = f.join(session_flags(idx))
    gap = (idx.to_series().diff() != BAR).to_numpy()   # first bar counts as a gap
    block = np.cumsum(gap)
    since = pd.Series(np.arange(len(idx)), index=idx).groupby(block).cumcount()
    f["hours_since_gap"] = np.minimum(since.to_numpy(), GAP_CAP).astype(float)
    return f[CALENDAR_FEATURES]


# ---------------------------------------------------------------------------
# stage 2b: ml-pipe multi-window features (``alpaca_ml`` ``time_series_features``)
# ---------------------------------------------------------------------------
# ml-pipe windows on 30-min bars [3, 12, 24, 48, 240, 1008, 3024] = 1.5h, 6h, 12h, 1d, 5d, 21d, 63d;
# the same clock spans on 1h bars (1.5h rounded to 2). Shift 23 ~ ml-pipe's 47 (1 day ago).
ML_PIPE_WINDOWS = (2, 6, 12, 24, 120, 504, 1512)
ML_PIPE_SHIFTS = (0, 23)


def _weighted_window(x: np.ndarray, window: int, centred: bool) -> np.ndarray:
    """ml-pipe's WEMA / WEVolatility kernel: weights ``(1-a)^k`` (newest = 1), ``a = 2/(1+window)``.

    ``centred=False`` -> weighted mean of x; ``True`` -> weighted variance around the *plain*
    window mean (ddof=1 scaling), exactly as ``WEVolatility._make_window_fn``.
    """
    alpha = 2 / (1 + window)
    w = (1 - alpha) ** np.arange(window)[::-1]           # oldest ... newest
    out = np.full(len(x), np.nan)
    if len(x) < window:
        return out
    k = w[::-1]                                          # np.convolve flips the kernel
    s1 = np.convolve(x, k, mode="valid")
    if not centred:
        out[window - 1:] = s1 / w.sum()
        return out
    s2 = np.convolve(x * x, k, mode="valid")
    m = np.convolve(x, np.ones(window), mode="valid") / window
    var = (s2 - 2 * m * s1 + m * m * w.sum()) / w.sum() * window / (window - 1)
    out[window - 1:] = np.sqrt(np.clip(var, 0, None))
    return out


def _fill_moments(s: pd.Series, window: int) -> pd.Series:
    """ml-pipe fills NaN skew/kurt after warm-up (flat windows) with 0."""
    s = s.copy()
    s.iloc[window - 1:] = s.iloc[window - 1:].fillna(0.0)
    return s


def _ml_pipe_feature(kind: str, c: pd.Series, w: int) -> pd.Series | None:
    """One ``tsf`` feature type on a close series; ``None`` when the window is too short for it."""
    ret = c.pct_change(fill_method=None)
    ma = lambda s, n: s.rolling(n).mean()  # noqa: E731
    if kind == "diff":
        return c.diff(w)
    if kind == "pct_change":
        return c.pct_change(w, fill_method=None)
    if kind == "ma_div":
        return (c - ma(c, w)) / ma(c, w)
    if kind == "wema_div":
        wema = pd.Series(_weighted_window(c.to_numpy(float), w, centred=False), index=c.index)
        return (c - wema) / wema
    if kind == "volatility":
        return c.rolling(w).std()
    if kind == "change_volatility":
        return ret.rolling(w).std()
    if kind in ("we_volatility", "change_we_volatility"):
        if w < 3:
            return None
        src = c if kind == "we_volatility" else ret
        return pd.Series(_weighted_window(src.to_numpy(float), w, centred=True), index=c.index)
    if kind in ("skew", "change_skew"):
        if w < 3:
            return None
        return _fill_moments((c if kind == "skew" else ret).rolling(w).skew(), w)
    if kind in ("kurt", "change_kurt"):
        if w < 4:                                      # unbiased excess kurtosis needs n >= 4
            return None
        return _fill_moments((c if kind == "kurt" else ret).rolling(w).kurt(), w)
    if kind in ("macd", "macd_diff"):
        if w < 3:
            return None
        long_ma, short_ma = ma(c, w), ma(c, int(w * 0.5))
        return (short_ma - long_ma) / long_ma if kind == "macd" else short_ma - long_ma
    if kind == "rsi":
        d = c.diff()
        gain, loss = d.clip(lower=0), (-d).clip(lower=0)
        rs = ma(gain, w) / ma(loss, w).clip(lower=1e-7)
        return 1.0 - 2.0 / (1.0 + rs)
    if kind == "drawdown":
        hi = c.rolling(w).max()
        return (hi - c) / hi
    if kind == "drawup":
        lo = c.rolling(w).min()
        return (c - lo) / lo
    if kind == "zscore":
        if w < 3:
            return None
        de_ma, sd = c - ma(c, w), c.rolling(w).std(ddof=0)
        z = de_ma / sd
        z[sd == 0] = np.sign(de_ma[sd == 0]) * 3
        return z.clip(-3, 3)
    raise ValueError(f"unknown ml-pipe feature type {kind!r}")


ML_PIPE_TYPES = ("diff", "pct_change", "ma_div", "wema_div", "volatility", "change_volatility",
                 "we_volatility", "change_we_volatility", "skew", "kurt", "change_skew", "change_kurt",
                 "macd", "macd_diff", "rsi", "drawdown", "drawup", "zscore")


ASSET_SEP = "__"          # "<asset>__<type>_w<window>_s<shift>" for basket assets; USDJPY columns are unprefixed


def asset_of(column: str) -> str:
    """``"EURUSD__ma_div_w24_s0"`` -> ``"EURUSD"``; unprefixed (USDJPY / calendar) columns -> ``"USDJPY"``."""
    return column.split(ASSET_SEP, 1)[0] if ASSET_SEP in column else "USDJPY"


def feature_type_of(column: str) -> str:
    """``"EURUSD__ma_div_w24_s0"`` -> ``"ma_div"``; calendar columns -> ``"calendar"``; ``stale_*`` -> ``"stale"``."""
    base = column.split(ASSET_SEP, 1)[-1]
    if base in CALENDAR_FEATURES:
        return "calendar"
    if base.startswith("stale"):
        return "stale"
    import re

    m = re.match(r"^(.*)_w\d+_s\d+$", base)
    return m.group(1) if m else base


class MultiWindowFeatureEngineer(_FrameTransformer):
    """ml-pipe's ``basic_feature_defs`` x ``windows`` x ``shifts`` on the close, plus calendar columns.

    Columns are ``{type}_w{window}_s{shift}`` (shift 0 = now, 23 = the same feature one day ago),
    followed by ``CALENDAR_FEATURES`` when ``calendar=True`` (candidates can exclude them).
    Every feature at bar t uses bars <= t only. Feature types follow
    ``alpaca_ml.contrib.sklearn.feature_extraction.time_series_features``; ``Volatility``,
    ``WEVolatility``, ``Skew`` and ``Kurt`` act on price *levels* as they do there, the ``Change*``
    variants on 1-bar percentage changes.

    ``assets`` (v003+): names of basket assets whose closes arrive as ``close_<name>`` columns (joined
    onto the bars by the pipeline, see ``usdjpy.data.align_basket``). Each gets the same feature set
    under the prefix ``<name>__``, plus its ``stale_<name>`` column (hours since the asset's last print)
    when present. Assets without a ``close_<name>`` column are skipped silently (e.g. dropped as too
    sparse by ``align_basket``).
    """

    def __init__(self, types=ML_PIPE_TYPES, windows=ML_PIPE_WINDOWS, shifts=ML_PIPE_SHIFTS, calendar: bool = True,
                 assets=()):
        self.types = types
        self.windows = windows
        self.shifts = shifts
        self.calendar = calendar
        self.assets = assets

    def _base(self, c: pd.Series) -> dict[str, pd.Series]:
        out = {}
        for kind in self.types:
            for w in self.windows:
                s = _ml_pipe_feature(kind, c, int(w))
                if s is not None:
                    out[f"{kind}_w{w}"] = s.replace([np.inf, -np.inf], np.nan)
        return out

    def _present_assets(self, X) -> list[str]:
        return [a for a in self.assets if f"close_{a}" in X.columns]

    def _output_columns(self, X):
        names = [k for k in self._base(pd.Series(np.linspace(1, 2, 8), dtype=float))]
        cols = [f"{n}_s{s}" for s in self.shifts for n in names]
        for a in self._present_assets(X):
            cols += [f"{a}{ASSET_SEP}{n}_s{s}" for s in self.shifts for n in names]
            if f"stale_{a}" in X.columns:
                cols.append(f"{a}{ASSET_SEP}stale")
        return cols + (list(CALENDAR_FEATURES) if self.calendar else [])

    def transform(self, X):
        check_is_fitted(self, "feature_names_in_")
        X = self._check_frame(X)
        base = self._base(X["close"].astype(float))
        parts = {f"{n}_s{s}": v.shift(s) for s in self.shifts for n, v in base.items()}
        for a in self._present_assets(X):
            ab = self._base(X[f"close_{a}"].astype(float))
            parts.update({f"{a}{ASSET_SEP}{n}_s{s}": v.shift(s) for s in self.shifts for n, v in ab.items()})
            if f"stale_{a}" in X.columns:
                parts[f"{a}{ASSET_SEP}stale"] = X[f"stale_{a}"].astype(float)
        f = pd.DataFrame(parts, index=X.index)
        if self.calendar:
            f = f.join(calendar_features(pd.DatetimeIndex(X.index)))
        return f[self._output_columns(X)]


# ---------------------------------------------------------------------------
# stage 3: encoding
# ---------------------------------------------------------------------------
def feature_columns(drop=()) -> list[str]:
    drop = set(drop)
    unknown = drop - set(ALL_FEATURES)
    if unknown:
        raise ValueError(f"unknown feature(s) to drop: {sorted(unknown)}")
    return [c for c in ALL_FEATURES if c not in drop]


class ColumnSubset:
    """Picklable column selector for ``ColumnTransformer``: every column minus ``exclude``,
    filtered by ``include_regex`` / ``exclude_regex`` (``re.search``)."""

    def __init__(self, exclude=(), include_regex: str | None = None, exclude_regex: str | None = None,
                 assets=None, exclude_assets=()):
        self.exclude = tuple(exclude)
        self.include_regex = include_regex
        self.exclude_regex = exclude_regex
        self.assets = None if assets is None else tuple(assets)       # keep only these assets' columns
        self.exclude_assets = tuple(exclude_assets)

    def __call__(self, X: pd.DataFrame) -> list[str]:
        import re

        cols = [c for c in X.columns if c not in self.exclude]
        if self.assets is not None:
            cols = [c for c in cols if asset_of(c) in self.assets]
        if self.exclude_assets:
            cols = [c for c in cols if asset_of(c) not in self.exclude_assets]
        if self.include_regex:
            cols = [c for c in cols if re.search(self.include_regex, c)]
        if self.exclude_regex:
            cols = [c for c in cols if not re.search(self.exclude_regex, c)]
        if not cols:
            raise ValueError(f"{self!r} selected no columns")
        return cols

    def __repr__(self):
        return (f"ColumnSubset(exclude={len(self.exclude)} cols, include_regex={self.include_regex!r}, "
                f"exclude_regex={self.exclude_regex!r}, assets={self.assets}, exclude_assets={self.exclude_assets})")


def make_preprocessor(kind: str = "tree", drop=(), columns=None) -> ColumnTransformer:
    """Select feature columns and encode them for a model family.

    ``columns=None`` selects ``ALL_FEATURES`` minus ``drop`` (the v001 bar features); a
    :class:`ColumnSubset` (or any list / callable) selects from whatever frame arrives.

    ``"tree"``   passthrough; NaN kept (LightGBM handles missing values natively).
    ``"dense"``  median-impute + standardise (linear models).
    """
    cols = feature_columns(drop) if columns is None else columns
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


# ---------------------------------------------------------------------------
# v004: stability-ranked feature selection (ml-pipe's FeatureSelector / MulticollinearityFilter, simplified)
# ---------------------------------------------------------------------------
class StabilityFeatureSelector(TransformerMixin, BaseEstimator):
    """Keep the ``n_features`` columns whose Spearman correlation with ``y`` is *stable* over time.

    Fit on the training rows only (it sees ``y``, so it lives inside the model pipeline and is refit per
    fold). The rows are cut into chronological blocks of ``block`` bars (ml-pipe's
    ``corr_stability_window``, 63 days = 1512 hourly bars); each feature's Spearman correlation with ``y``
    is computed inside every block and the feature's score is the weighted mean of those signed
    correlations, recent blocks weighing more (``half_life`` in blocks; ml-pipe's ``ewma_alpha``). A
    feature that flips sign between blocks cancels itself out, so ranking by ``|score|`` prefers
    consistent relationships over strong-but-episodic ones.

    Optional group caps (ml-pipe's asset / feature-type pre-selection): keep only the ``n_assets`` assets
    and ``n_types`` feature types with the highest mean ``|score|`` before the final cut. Then a greedy
    multicollinearity filter drops any candidate whose ``|Spearman|`` with an already-kept, higher-ranked
    column exceeds ``corr_threshold`` (ml-pipe's ``MulticollinearityFilter(threshold=0.8)``), among the
    top ``candidate_factor * n_features`` columns.

    Attributes after ``fit``: ``scores_`` (signed stability score per column), ``block_corr_`` (blocks x
    columns), ``selected_`` (kept columns in rank order), ``dropped_collinear_``.
    """

    def __init__(self, n_features: int = 200, block: int = 1512, half_life: float = 2.0, min_block: int = 200,
                 n_assets: int | None = None, n_types: int | None = None, corr_threshold: float | None = 0.8,
                 candidate_factor: int = 3, corr_rows: int = 3000, random_state: int = 0):
        self.n_features = n_features
        self.block = block
        self.half_life = half_life
        self.min_block = min_block
        self.n_assets = n_assets
        self.n_types = n_types
        self.corr_threshold = corr_threshold
        self.candidate_factor = candidate_factor
        self.corr_rows = corr_rows
        self.random_state = random_state

    # -- helpers
    @staticmethod
    def _rank_corr(X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        """Spearman of every column with y, NaN-aware (pairwise complete), as a 1-D array."""
        r = X.rank().to_numpy(dtype=np.float32)
        ry = pd.Series(y).rank().to_numpy(dtype=np.float32)[:, None]
        mask = ~np.isnan(r)
        n = mask.sum(axis=0)
        rx = np.where(mask, r, 0.0)
        ryy = np.where(mask, ry, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mx, my = rx.sum(0) / n, ryy.sum(0) / n
            cov = ((rx - mx) * mask * (ryy - my)).sum(0)
            sx = np.sqrt((((rx - mx) * mask) ** 2).sum(0))
            sy = np.sqrt((((ryy - my) * mask) ** 2).sum(0))
            out = cov / (sx * sy)
        out[n < 3] = np.nan
        return out.astype(float)

    def _blocks(self, n: int) -> list[slice]:
        starts = list(range(0, n, self.block))
        blocks = [slice(s, min(s + self.block, n)) for s in starts]
        if len(blocks) > 1 and (blocks[-1].stop - blocks[-1].start) < self.min_block:
            blocks[-2] = slice(blocks[-2].start, n)
            blocks.pop()
        return blocks

    # -- sklearn API
    def fit(self, X: pd.DataFrame, y=None):
        if y is None:
            raise ValueError("StabilityFeatureSelector needs y")
        X = pd.DataFrame(X)
        y = np.asarray(y, dtype=float).ravel()
        blocks = self._blocks(len(X))
        corr = np.vstack([self._rank_corr(X.iloc[b], y[b]) for b in blocks])        # blocks x columns
        ages = np.arange(len(blocks))[::-1].astype(float)                            # 0 = most recent block
        w = 0.5 ** (ages / self.half_life)
        valid = ~np.isnan(corr)
        wsum = (w[:, None] * valid).sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            score = np.where(wsum > 0, (w[:, None] * np.nan_to_num(corr)).sum(0) / wsum, np.nan)
        self.block_corr_ = pd.DataFrame(corr, columns=X.columns, index=[f"block{i}" for i in range(len(blocks))])
        self.scores_ = pd.Series(score, index=X.columns, name="stability")
        strength = self.scores_.abs().fillna(0.0)

        cols = list(X.columns)
        if self.n_assets:
            by_asset = strength.groupby([asset_of(c) for c in cols]).mean().sort_values(ascending=False)
            keep = set(by_asset.index[: self.n_assets]) | {"USDJPY"}
            cols = [c for c in cols if asset_of(c) in keep]
        if self.n_types:
            by_type = strength[cols].groupby([feature_type_of(c) for c in cols]).mean().sort_values(ascending=False)
            keep_t = set(by_type.index[: self.n_types])
            cols = [c for c in cols if feature_type_of(c) in keep_t]
        ranked = strength[cols].sort_values(ascending=False)
        ranked = ranked[ranked > 0]
        candidates = list(ranked.index[: max(self.n_features, 1) * max(self.candidate_factor, 1)])

        dropped: list[str] = []
        if self.corr_threshold is not None and len(candidates) > self.n_features:
            rng = np.random.default_rng(self.random_state)
            rows = np.sort(rng.choice(len(X), size=min(self.corr_rows, len(X)), replace=False))
            C = X.iloc[rows][candidates].rank().corr().abs().to_numpy()
            kept_idx: list[int] = []
            for i in range(len(candidates)):
                if kept_idx and np.nanmax(C[i, kept_idx]) > self.corr_threshold:
                    dropped.append(candidates[i])
                    continue
                kept_idx.append(i)
                if len(kept_idx) >= self.n_features:
                    break
            selected = [candidates[i] for i in kept_idx]
        else:
            selected = candidates[: self.n_features]
        if not selected:
            raise ValueError("StabilityFeatureSelector selected no columns (all scores NaN?)")
        self.selected_ = selected
        self.dropped_collinear_ = dropped
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "selected_")
        return pd.DataFrame(X)[self.selected_]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "selected_")
        return np.asarray(self.selected_, dtype=object)
