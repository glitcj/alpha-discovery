"""Hourly bars: download from Yahoo Finance, cache as CSV, build the target, split off the holdout.

Only :func:`fetch_bars` touches the network (``make fetch``). Every experiment reads the
cached CSV, so a run is reproducible until the cache is refreshed.

    python -m usdjpy.data --ticker USDJPY=X --period 729d     # what `make fetch` runs
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TICKER = "USDJPY=X"
INTERVAL = "1h"
OHLC = ["open", "high", "low", "close"]
BAR = pd.Timedelta(hours=1)

_HERE = Path(__file__).resolve().parent


def data_dir() -> Path:
    """``$USDJPY_DATA_DIR`` if set, else ``<project>/dataset``."""
    env = os.environ.get("USDJPY_DATA_DIR")
    return Path(env) if env else _HERE.parent / "dataset"


def cache_path(ticker: str = TICKER, interval: str = INTERVAL) -> Path:
    slug = ticker.lower().replace("=x", "").replace("^", "").replace("=", "")
    return data_dir() / f"{slug}_{interval}.csv"


# ---------------------------------------------------------------------------
# download + cache
# ---------------------------------------------------------------------------
def normalise_download(raw: pd.DataFrame) -> pd.DataFrame:
    """yfinance frame -> lower-case OHLC indexed by a sorted, unique UTC ``timestamp``."""
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):         # (field, ticker) for single-ticker downloads
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    missing = set(OHLC) - set(df.columns)
    if missing:
        raise ValueError(f"download is missing columns {sorted(missing)}: {list(df.columns)}")
    idx = pd.DatetimeIndex(df.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    df.index = idx.rename("timestamp")
    df = df[OHLC].astype(float)                        # FX volume on Yahoo is always 0: dropped
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.dropna(subset=["close"])


def fetch_bars(ticker: str = TICKER, interval: str = INTERVAL, period: str = "729d",
               path: str | os.PathLike | None = None) -> pd.DataFrame:
    """Download bars with yfinance and write them to the CSV cache."""
    import yfinance as yf

    raw = yf.download(ticker, interval=interval, period=period, auto_adjust=False,
                      progress=False, threads=False)
    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker} {interval} {period}")
    bars = normalise_download(raw)
    # the current hour is still forming: its "close" is just the latest tick, so drop it
    bars = bars[bars.index + BAR <= pd.Timestamp.now(tz="UTC")]
    path =Path(path) if path else cache_path(ticker, interval)
    path.parent.mkdir(parents=True, exist_ok=True)
    bars.to_csv(path, index_label="timestamp")
    return bars


def load_bars(path: str | os.PathLike | None = None) -> pd.DataFrame:
    """Read the cached CSV (``make fetch`` writes it)."""
    path = Path(path) if path else cache_path()
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `make fetch` (or set USDJPY_DATA_DIR)")
    df = pd.read_csv(path, index_col="timestamp")
    df.index = pd.to_datetime(df.index, utc=True)
    return df[OHLC].astype(float).sort_index()


# ---------------------------------------------------------------------------
# target
# ---------------------------------------------------------------------------
def make_target(bars: pd.DataFrame) -> pd.DataFrame:
    """Label each bar with the direction of the *next* hourly close.

    Returns a frame indexed like the labelled bars with ``y`` (1 = next close higher) and
    ``next_logret`` (log return to the next close, used by the trading metrics). A bar is
    labelled only when the next bar is exactly one hour later (so the last bar before a
    weekend or a data gap gets no label) and the next close differs from this one.
    """
    close = bars["close"].astype(float)
    nxt = close.shift(-1)
    step = bars.index.to_series().shift(-1) - bars.index.to_series()
    ret = np.log(nxt / close)
    ok = (step == BAR) & ret.notna() & (ret != 0)
    out = pd.DataFrame({"y": (ret > 0).astype(int), "next_logret": ret}, index=bars.index)
    return out[ok.to_numpy()]


def make_forward_return_target(bars: pd.DataFrame, horizon: int = 24, offset: int = 1) -> pd.DataFrame:
    """ml-pipe's ``FWDLogReturn(lookahead_window=horizon, lookahead_offset=offset)``.

    ``y_t = log(close[t + offset + horizon] / close[t + offset])``: the log return over the next
    ``horizon`` bars, starting ``offset`` bars after t (a buffer for acting on the signal). Shifts
    are positional, so on the FX-week grid a label can span a weekend, as in ml-pipe. Returns
    ``y`` and ``next_logret`` (the same values, used by the trading read-out).
    """
    if horizon < 1 or offset < 0:
        raise ValueError("horizon must be >= 1 and offset >= 0")
    close = bars["close"].astype(float)
    y = np.log(close.shift(-(offset + horizon)) / close.shift(-offset))
    out = pd.DataFrame({"y": y, "next_logret": y}, index=bars.index)
    return out.dropna()


def split_holdout(index: pd.Index, frac: float = 0.15, purge: int = 0) -> tuple[pd.Index, pd.Index]:
    """Chronological split: the last ``frac`` of the rows is the untouched holdout.

    ``purge`` drops that many rows just before the holdout, so no development label (which looks
    ``offset + horizon`` bars ahead) is computed from holdout prices.
    """
    if not 0 < frac < 1:
        raise ValueError("frac must be in (0, 1)")
    if purge < 0:
        raise ValueError("purge must be >= 0")
    index = pd.Index(index).sort_values()
    n_hold = max(1, int(round(len(index) * frac)))
    return index[:-(n_hold + purge)], index[-n_hold:]


def write_predictions(index, scores, path, y_true=None, next_logret=None, task: str = "direction") -> pd.DataFrame:
    """Holdout (or any) predictions.

    ``task="direction"``: ``timestamp, p_up, pred[, y_true, next_logret]``;
    ``task="return"``:    ``timestamp, y_pred, pred_sign[, y_true]`` (predicted forward log return).
    """
    p = np.asarray(scores, dtype=float).ravel()
    if len(p) != len(index):
        raise ValueError(f"{len(index)} timestamps but {len(p)} predictions")
    if not np.isfinite(p).all():
        raise ValueError("predictions contain NaN/inf")
    idx = pd.Index(index, name="timestamp")
    if task == "return":
        out = pd.DataFrame({"y_pred": p, "pred_sign": np.sign(p).astype(int)}, index=idx)
        if y_true is not None:
            out["y_true"] = np.asarray(y_true, dtype=float)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(path)
        return out
    out = pd.DataFrame({"p_up": p, "pred": (p > 0.5).astype(int)}, index=idx)
    if y_true is not None:
        out["y_true"] = np.asarray(y_true, dtype=int)
    if next_logret is not None:
        out["next_logret"] = np.asarray(next_logret, dtype=float)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m usdjpy.data", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ticker", default=TICKER)
    p.add_argument("--interval", default=INTERVAL)
    p.add_argument("--period", default="729d")
    args = p.parse_args(argv)
    bars = fetch_bars(args.ticker, args.interval, args.period)
    path = cache_path(args.ticker, args.interval)
    gaps = (bars.index.to_series().diff() > BAR).sum()
    print(f"[fetch] {args.ticker} {args.interval}: {len(bars):,} bars "
          f"{bars.index[0]} -> {bars.index[-1]} ({gaps} gaps) -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
