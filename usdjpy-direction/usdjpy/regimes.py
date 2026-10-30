"""Market-regime labels shared by the regime slice and the regime figures.

Two definitions, both computed from the cleaned USDJPY bars (so they are available before any
feature engineering and for any arm):

* :func:`joint_regimes`: the 6-cell grid used by the ``regime`` slice in :mod:`usdjpy.evaluation`:
  sign of the past ``window``-bar (21 trading days) move × tercile of the past ``window``-bar
  volatility of hourly log returns;
* :func:`decile_regimes`: ml-pipe's ``InferenceReporter`` definition: trend (``pct_change(window)``)
  or volatility (rolling std of log returns over ``window``) bucketed separately into deciles
  ``Q0 … Q90`` over the whole sample, either for the *past* window or the *future* one
  (``LeakyShift(-(window + offset))`` of the past feature).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluation import REGIME_LABELS

DECILE_LABELS = [f"Q{q}" for q in range(0, 100, 10)]
VOL_LABELS = ("low vol", "mid vol", "high vol")
DEFAULT_WINDOW = 504                                # bars ≈ 21 FX-week days


def _close(bars: pd.DataFrame | pd.Series) -> pd.Series:
    return (bars["close"] if isinstance(bars, pd.DataFrame) else bars).astype(float)


def trend_feature(bars, window: int = DEFAULT_WINDOW) -> pd.Series:
    """Past ``window``-bar percentage change of the close (ml-pipe ``PCTChange``)."""
    return _close(bars).pct_change(window)


def vol_feature(bars, window: int = DEFAULT_WINDOW) -> pd.Series:
    """Past ``window``-bar standard deviation of hourly log returns (ml-pipe ``LogReturnVolatility``)."""
    return np.log(_close(bars)).diff().rolling(window).std()


def joint_regimes(bars, window: int = DEFAULT_WINDOW) -> pd.Series:
    """One of ``REGIME_LABELS`` per bar (``"unknown"`` during the warm-up)."""
    trend, vol = trend_feature(bars, window), vol_feature(bars, window)
    try:
        tercile = pd.qcut(vol, 3, labels=list(VOL_LABELS)).astype(object)
    except ValueError:
        tercile = pd.Series(np.nan, index=vol.index, dtype=object)
    sign = np.where(trend >= 0, "up-trend", "down-trend")
    lab = pd.Series(sign, index=vol.index) + " / " + tercile.astype(str)
    lab = lab.where(trend.notna() & tercile.notna(), "unknown")
    return lab.rename("regime")


def decile_regimes(bars, window: int = DEFAULT_WINDOW, offset: int = 1, *, kind: str = "trend",
                   when: str = "past") -> pd.Series:
    """ml-pipe regime deciles. ``kind`` ``"trend"`` | ``"vol"``; ``when`` ``"past"`` | ``"future"``.

    The future variant looks ``window + offset`` bars ahead (a *diagnostic* grouping, never a feature).
    """
    if kind not in ("trend", "vol"):
        raise ValueError("kind must be 'trend' or 'vol'")
    if when not in ("past", "future"):
        raise ValueError("when must be 'past' or 'future'")
    feat = trend_feature(bars, window) if kind == "trend" else vol_feature(bars, window)
    if when == "future":
        feat = feat.shift(-(window + offset))
    try:
        lab = pd.qcut(feat, 10, labels=DECILE_LABELS, duplicates="drop").astype(object)
    except ValueError:
        lab = pd.Series(np.nan, index=feat.index, dtype=object)
    return lab.where(lab.notna(), "unknown").astype(str).rename(f"{when}_{kind}_decile")


def regime_stats(labels: pd.Series, fwd_ret: pd.Series, order=None) -> pd.DataFrame:
    """Per-regime: share of bars, mean / std of the forward return (bp), up share, mean spell (bars);
    ``"unknown"`` rows (warm-up) are excluded."""
    df = pd.DataFrame({"g": labels.astype(str), "r": fwd_ret.astype(float)}).dropna()
    df = df[df["g"] != "unknown"]                       # warm-up rows carry no regime
    spells = (df["g"] != df["g"].shift()).cumsum()
    spell_len = df.groupby(spells)["g"].agg(["first", "size"]).groupby("first")["size"].mean()
    rows = df.groupby("g")["r"].agg(n="size", mean_bp="mean", std_bp="std", up_share=lambda s: (s > 0).mean())
    rows[["mean_bp", "std_bp"]] *= 1e4
    rows["share"] = rows["n"] / rows["n"].sum()
    rows["mean_spell_bars"] = spell_len
    rows = rows[["n", "share", "mean_bp", "std_bp", "up_share", "mean_spell_bars"]]
    order = list(order) if order else [o for o in REGIME_LABELS if o in rows.index]
    keep = [o for o in order if o in rows.index] + [i for i in rows.index if i not in order]
    rows.index.name = "regime"
    return rows.loc[keep]


def fold_spans(folds: pd.DataFrame | None) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """``(test_start, test_end)`` per fold from an ``EvaluationReport.folds`` frame."""
    if folds is None or "test_start" not in folds:
        return []
    return [(pd.Timestamp(s), pd.Timestamp(e)) for s, e in zip(folds["test_start"], folds["test_end"])]
