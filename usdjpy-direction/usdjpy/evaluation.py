"""Walk-forward evaluation for direction classifiers and forward-return regressors.

Two tasks share one evaluator design (``EvaluationReport.task``):

``"direction"`` (v001)  ``DirectionEvaluator``: ``predict_proba`` models, primary metric ``auc``.
``"return"`` (v002+)    ``ReturnEvaluator``: ``predict`` models of ml-pipe's forward log return,
                        primary metric ``spearman`` (plus ``pearson``, ``r2``, ``mse`` and sign
                        metrics ``mcc`` / ``auc`` / ``hit_rate``). Labels overlap by ``horizon`` bars,
                        so the trading read-out uses every ``horizon``-th row (one decision a day).

Direction task details:

* metrics: probability quality (``auc``, ``log_loss``, ``brier``), hit rates (``accuracy``,
  ``balanced_accuracy``) and a naive trading read-out: go long when ``P(up) > 0.5``, short
  when below, pay ``cost_bps`` per unit of position change (``strategy_bps`` = mean net
  return per bar in basis points, ``sharpe`` annualised);
* ``DirectionSlicer``: feature rows -> report slices (session, time of day, weekday, volatility);
* ``DirectionEvaluator`` (a ``BaseEstimator`` so its settings are ``get_params``-able): an
  expanding-window ``TimeSeriesSplit`` with a ``gap`` of bars between train and test so no
  rolling feature or label straddles the boundary; collects out-of-fold probabilities into an
  ``EvaluationReport`` with overall, per-fold and per-slice metrics;
* ``*Display`` classes following sklearn's ``from_predictions`` / ``plot(ax=)`` convention.
"""

from __future__ import annotations

import json
import re
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibrationDisplay
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (RocCurveDisplay, accuracy_score, balanced_accuracy_score, brier_score_loss,
                             log_loss, matthews_corrcoef, mean_squared_error, r2_score, roc_auc_score)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

BARS_PER_YEAR = 24 * 5 * 52        # FX trades ~24h x 5 days
PRIMARY_METRIC = "auc"
DEFAULT_COST_BPS = 0.5


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def positions(p_up, threshold: float = 0.0) -> np.ndarray:
    """+1 long / -1 short / 0 flat from ``P(up)``; flat when ``|p - 0.5| <= threshold``."""
    edge = np.asarray(p_up, dtype=float).ravel() - 0.5
    pos = np.sign(edge)
    pos[np.abs(edge) <= threshold] = 0.0
    return pos


def strategy_returns(p_up, next_logret, cost_bps: float = DEFAULT_COST_BPS, threshold: float = 0.0) -> np.ndarray:
    """Per-bar net log return of trading the sign of ``P(up) - 0.5``.

    Cost is ``cost_bps`` per unit of position change (entering from flat = 1, flipping = 2).
    """
    pos = positions(p_up, threshold)
    ret = np.asarray(next_logret, dtype=float).ravel()
    turnover = np.abs(np.diff(pos, prepend=0.0))
    return pos * ret - turnover * cost_bps * 1e-4


def _safe(fn, *a, **k) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            out = float(fn(*a, **k))
        except ValueError:        # e.g. AUC with a single class in the slice
            return float("nan")
    return out


def compute_metrics(y_true, p_up, next_logret=None, cost_bps: float = DEFAULT_COST_BPS) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int).ravel()
    p = np.clip(np.asarray(p_up, dtype=float).ravel(), 1e-6, 1 - 1e-6)
    pred = (p > 0.5).astype(int)
    out = {
        "auc": _safe(roc_auc_score, y, p),
        "log_loss": _safe(log_loss, y, p, labels=[0, 1]),
        "brier": _safe(brier_score_loss, y, p),
        "accuracy": _safe(accuracy_score, y, pred),
        "balanced_accuracy": _safe(balanced_accuracy_score, y, pred),
        "pred_up_share": float(pred.mean()),
    }
    if next_logret is not None:
        net = strategy_returns(p, next_logret, cost_bps)
        sd = net.std(ddof=0)
        out["strategy_bps"] = float(net.mean() * 1e4)
        out["sharpe"] = float(net.mean() / sd * np.sqrt(BARS_PER_YEAR)) if sd > 0 else float("nan")
        out["turnover"] = float(np.abs(np.diff(positions(p), prepend=0.0)).mean())
    return out


DAYS_PER_YEAR = 260


def daily_strategy(y_pred, fwd_ret, horizon: int = 24, cost_bps: float = DEFAULT_COST_BPS,
                   start: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Trade ``sign(y_pred)`` on every ``horizon``-th row only (non-overlapping forward returns).

    Returns ``(row positions used, net log return per trade)``; cost is ``cost_bps`` per unit of
    position change between consecutive trades.
    """
    rows = np.arange(start, len(np.asarray(fwd_ret)), max(1, int(horizon)))
    pos = np.sign(np.asarray(y_pred, dtype=float).ravel()[rows])
    ret = np.asarray(fwd_ret, dtype=float).ravel()[rows]
    turnover = np.abs(np.diff(pos, prepend=0.0))
    return rows, pos * ret - turnover * cost_bps * 1e-4


def compute_return_metrics(y_true, y_pred, next_logret=None, cost_bps: float = DEFAULT_COST_BPS,
                           horizon: int = 24) -> dict[str, float]:
    """Regression + sign metrics for forward-return predictions (ml-pipe's leaderboard set)."""
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(y_pred, dtype=float).ravel()
    up, pred_up = (y > 0).astype(int), (p > 0).astype(int)
    flat = np.nanstd(p) == 0
    out = {
        "spearman": float("nan") if flat else _safe(lambda a, b: spearmanr(a, b).statistic, y, p),
        "pearson": float("nan") if flat else _safe(lambda a, b: pearsonr(a, b).statistic, y, p),
        "r2": _safe(r2_score, y, p),
        "mse": _safe(mean_squared_error, y, p),
        "mcc": _safe(matthews_corrcoef, up, pred_up),
        "auc": _safe(roc_auc_score, up, p),
        "hit_rate": float(np.mean(np.sign(p) == np.sign(y))),
        "pred_up_share": float(pred_up.mean()),
    }
    if next_logret is not None:
        _, net = daily_strategy(p, next_logret, horizon, cost_bps)
        sd = net.std(ddof=0)
        out["strategy_bps"] = float(net.mean() * 1e4) if len(net) else float("nan")
        out["sharpe"] = float(net.mean() / sd * np.sqrt(DAYS_PER_YEAR)) if len(net) and sd > 0 else float("nan")
    return out


# name -> greater_is_better (None = descriptive only)
METRICS: dict[str, bool | None] = {
    "auc": True, "log_loss": False, "brier": False, "accuracy": True, "balanced_accuracy": True,
    "pred_up_share": None, "strategy_bps": True, "sharpe": True, "turnover": None,
    "spearman": True, "pearson": True, "r2": True, "mse": False, "mcc": True, "hit_rate": True,
}

# task -> primary metric, tie-breaker (name, ascending)
TASKS = {
    "direction": {"primary": "auc", "tiebreak": ("log_loss", True)},
    "return": {"primary": "spearman", "tiebreak": ("mse", True)},
}


# ---------------------------------------------------------------------------
# slices
# ---------------------------------------------------------------------------
TOD_BINS = [-1, 3, 7, 11, 15, 19, 23]
TOD_LABELS = ["00-03", "04-07", "08-11", "12-15", "16-19", "20-23"]
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
SESSION_ORDER = ["tokyo", "tokyo+london", "london", "london+ny", "ny", "off-hours"]
DEFAULT_SLICES = ("session", "time_of_day", "weekday", "volatility")
SLICE_ORDER = {
    "session": SESSION_ORDER,
    "time_of_day": TOD_LABELS,
    "weekday": WEEKDAYS,
    "volatility": ["low vol", "mid vol", "high vol"],
    "confidence": ["<1pp", "1-2pp", "2-5pp", "5pp+"],
}
CONF_BINS = [-0.001, 0.01, 0.02, 0.05, 1.0]


def _session_label(row) -> str:
    t, l, n = row["session_tokyo"], row["session_london"], row["session_ny"]
    if t and l:
        return "tokyo+london"
    if l and n:
        return "london+ny"
    if t:
        return "tokyo"
    if l:
        return "london"
    if n:
        return "ny"
    return "off-hours"


class DirectionSlicer:
    """Engineered feature rows -> one label column per slice (``DEFAULT_SLICES``)."""

    def __init__(self, slices=DEFAULT_SLICES):
        self.slices = tuple(slices)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        need = {"session_tokyo", "session_london", "session_ny", "hour", "weekday"}
        if not need <= set(X.columns):          # derive calendar columns from the timestamps
            from .features import calendar_features

            cal = calendar_features(pd.DatetimeIndex(X.index))
            X = pd.concat([X.drop(columns=[c for c in cal if c in X]), cal], axis=1)
        out = pd.DataFrame(index=X.index)
        for name in self.slices:
            if name == "session":
                out[name] = X[["session_tokyo", "session_london", "session_ny"]].apply(_session_label, axis=1)
            elif name == "time_of_day":
                out[name] = pd.cut(X["hour"], TOD_BINS, labels=TOD_LABELS).astype(str)
            elif name == "weekday":
                out[name] = X["weekday"].map(lambda d: WEEKDAYS[int(d)] if pd.notna(d) else "unknown")
            elif name == "volatility":
                col = next((c for c in ("ret_std_24", "change_volatility_w24_s0") if c in X), None)
                v = X[col] if col else pd.Series(np.nan, index=X.index)
                try:
                    lab = pd.qcut(v, 3, labels=SLICE_ORDER["volatility"]).astype(object)
                except ValueError:
                    lab = pd.Series(np.nan, index=X.index, dtype=object)
                out[name] = lab.where(lab.notna(), "unknown")
            else:
                raise ValueError(f"unknown slice {name!r}; choose from {DEFAULT_SLICES}")
        return out.astype(str)


RETURN_CONF_LABELS = ["q1 (smallest |pred|)", "q2", "q3", "q4 (largest |pred|)"]


def confidence_labels(scores, task: str = "direction") -> np.ndarray:
    """Direction: ``|P(up) - 0.5|`` buckets; return: quartiles of ``|predicted return|``."""
    s = np.asarray(scores, dtype=float)
    if task == "return":
        try:
            return pd.qcut(np.abs(s), 4, labels=RETURN_CONF_LABELS).astype(str)
        except ValueError:                     # constant predictions
            return np.full(len(s), RETURN_CONF_LABELS[0])
    edge = np.abs(s - 0.5)
    return pd.cut(edge, CONF_BINS, labels=SLICE_ORDER["confidence"]).astype(str)


def slice_table(y_true, y_pred, next_logret, labels, *, cost_bps: float = DEFAULT_COST_BPS,
                order: list | None = None, task: str = "direction", horizon: int = 24) -> pd.DataFrame:
    """Per-group metrics in ``order`` (else by n).

    Direction: ``n, share, up_share, auc, accuracy, strategy_bps``. Return: ``n, share, up_share,
    mean_y_bp, spearman, hit_rate`` (no per-slice trading: slices break the daily sampling).
    """
    df = pd.DataFrame({"y": np.asarray(y_true, float), "p": np.asarray(y_pred, float),
                       "r": np.asarray(next_logret, float), "g": np.asarray(labels).astype(str)})
    rows = []
    for g, part in df.groupby("g", sort=False):
        if task == "return":
            m = compute_return_metrics(part.y, part.p)
            rows.append({"slice": g, "n": len(part), "share": len(part) / len(df), "up_share": (part.y > 0).mean(),
                         "mean_y_bp": part.y.mean() * 1e4, "spearman": m["spearman"], "hit_rate": m["hit_rate"]})
            continue
        m = compute_metrics(part.y.astype(int), part.p, part.r, cost_bps)
        rows.append({"slice": g, "n": len(part), "share": len(part) / len(df), "up_share": part.y.mean(),
                     "auc": m["auc"], "accuracy": m["accuracy"], "strategy_bps": m["strategy_bps"]})
    tab = pd.DataFrame(rows).set_index("slice")
    if order:
        tab = tab.loc[[o for o in order if o in tab.index] + [i for i in tab.index if i not in order]]
    else:
        tab = tab.sort_values("n", ascending=False)
    return tab


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
@dataclass
class EvaluationReport:
    """Everything produced by one evaluation of one estimator (CV or holdout)."""

    name: str
    overall: dict[str, float]
    y_true: pd.Series
    y_pred: np.ndarray                         # P(up) for direction, predicted log return for return
    next_logret: np.ndarray
    slices: pd.DataFrame
    folds: pd.DataFrame | None = None
    fit_time: float = float("nan")
    method: str = "cv"
    cost_bps: float = DEFAULT_COST_BPS
    wall_time: float = float("nan")
    estimator_: object = None
    task: str = "direction"
    horizon: int = 24
    per_slice: dict[str, pd.DataFrame] = field(default_factory=dict)

    @property
    def p_up(self) -> np.ndarray:
        return self.y_pred

    @property
    def primary_metric(self) -> str:
        return TASKS[self.task]["primary"]

    @property
    def primary(self) -> float:
        return self.overall[self.primary_metric]

    @property
    def auc(self) -> float:
        return self.overall["auc"]

    @property
    def index(self) -> pd.Index:
        return self.y_true.index

    def slice_table(self, name: str) -> pd.DataFrame:
        if name not in self.per_slice:
            labels = confidence_labels(self.y_pred, self.task) if name == "confidence" else self.slices[name]
            order = RETURN_CONF_LABELS if (name == "confidence" and self.task == "return") else SLICE_ORDER.get(name)
            self.per_slice[name] = slice_table(self.y_true, self.y_pred, self.next_logret, labels,
                                               cost_bps=self.cost_bps, order=order, task=self.task,
                                               horizon=self.horizon)
        return self.per_slice[name]

    def summary_frame(self) -> pd.DataFrame:
        row = dict(self.overall)
        pm = self.primary_metric
        if self.folds is not None and f"test_{pm}" in self.folds:
            row[f"{pm}_fold_std"] = float(self.folds[f"test_{pm}"].std(ddof=0))
        row["fit_time_s"] = self.fit_time
        row["n"] = len(self.y_true)
        return pd.DataFrame([row], index=pd.Index([self.name], name="model"))


def leaderboard(reports: dict[str, EvaluationReport]) -> pd.DataFrame:
    """Models sorted by the task's primary metric (``auc`` / ``spearman``, desc), then the tie-breaker."""
    board = pd.concat([r.summary_frame() for r in reports.values()])
    task = TASKS[next(iter(reports.values())).task]
    tb, tb_asc = task["tiebreak"]
    return board.sort_values([task["primary"], tb], ascending=[False, tb_asc])


class DirectionEvaluator(BaseEstimator):
    """Walk-forward cross-validation of ``predict_proba`` models.

    Parameters
    ----------
    cv : int or splitter
        Number of ``TimeSeriesSplit`` folds, or any splitter with ``split(X)``.
    gap : int
        Bars dropped between each train block and its test block (>= the longest feature
        window you worry about leaking; labels look one bar ahead, so ``gap >= 1`` matters).
    max_train_size : int or None
        Rolling window instead of expanding when set.
    cost_bps : float
        Cost per unit position change for the trading metrics.
    slices : tuple of str
    """

    task = "direction"

    def __init__(self, cv=5, gap: int = 24, max_train_size: int | None = None,
                 cost_bps: float = DEFAULT_COST_BPS, slices=DEFAULT_SLICES):
        self.cv = cv
        self.gap = gap
        self.max_train_size = max_train_size
        self.cost_bps = cost_bps
        self.slices = slices

    def _cv(self):
        if isinstance(self.cv, int):
            return TimeSeriesSplit(n_splits=self.cv, gap=self.gap, max_train_size=self.max_train_size)
        return self.cv

    def _slicer(self) -> DirectionSlicer:
        return DirectionSlicer(self.slices)

    @property
    def primary_metric(self) -> str:
        return TASKS[self.task]["primary"]

    def predict_scores(self, estimator, X) -> np.ndarray:
        """What a fitted model contributes to the metrics: ``P(up)`` here."""
        return estimator.predict_proba(X)[:, 1]

    def _target(self, y) -> np.ndarray:
        return np.asarray(y, int)

    def _metrics(self, y, scores, next_logret) -> dict[str, float]:
        return compute_metrics(y, scores, next_logret, self.cost_bps)

    def _report_kwargs(self) -> dict:
        return {"task": self.task}

    def score(self, name, y, scores, next_logret, X, *, method="cv", folds=None, fit_time=float("nan"),
              estimator=None) -> EvaluationReport:
        y = pd.Series(self._target(y), index=X.index, name="y")
        p = np.asarray(scores, float).ravel()
        r = np.asarray(next_logret, float).ravel()
        return EvaluationReport(name=name, overall=self._metrics(y, p, r), y_true=y, y_pred=p, next_logret=r,
                                slices=self._slicer().transform(X), folds=folds, fit_time=fit_time,
                                method=method, cost_bps=self.cost_bps, estimator_=estimator,
                                **self._report_kwargs())

    def evaluate(self, estimator, X: pd.DataFrame, y, next_logret, name: str = "model",
                 verbose: bool = False) -> EvaluationReport:
        """Out-of-fold scores over the walk-forward test blocks (the first train block has none)."""
        t0 = time.perf_counter()
        y = self._target(y)
        r = np.asarray(next_logret, float)
        oof = np.full(len(X), np.nan)
        rows = []
        for k, (tr, te) in enumerate(self._cv().split(X)):
            est = clone(estimator)
            t1 = time.perf_counter()
            est.fit(X.iloc[tr], y[tr])
            fit_s = time.perf_counter() - t1
            p = self.predict_scores(est, X.iloc[te])
            oof[te] = p
            m = self._metrics(y[te], p, r[te])
            rows.append({"fold": k, "train_start": X.index[tr[0]], "train_end": X.index[tr[-1]],
                         "test_start": X.index[te[0]], "test_end": X.index[te[-1]], "n_train": len(tr),
                         "n_test": len(te), "fit_time": fit_s, **{f"test_{n}": v for n, v in m.items()}})
            if verbose:
                pm = self.primary_metric
                hit = m.get("accuracy", m.get("hit_rate"))
                print(f"  [{name}] fold {k}: {pm}={m[pm]:.4f} hit={hit:.4f} "
                      f"strat={m['strategy_bps']:+.3f}bp ({fit_s:.1f}s)", flush=True)
        mask = ~np.isnan(oof)
        folds = pd.DataFrame(rows).set_index("fold")
        rep = self.score(name, y[mask], oof[mask], r[mask], X[mask], method="cv", folds=folds,
                         fit_time=float(folds["fit_time"].mean()))
        rep.wall_time = time.perf_counter() - t0
        return rep

    def holdout(self, estimator, X_tr, y_tr, X_val, y_val, ret_val, name: str = "model") -> EvaluationReport:
        est = clone(estimator)
        t0 = time.perf_counter()
        est.fit(X_tr, self._target(y_tr))
        fit_s = time.perf_counter() - t0
        p = self.predict_scores(est, X_val)
        return self.score(name, y_val, p, ret_val, X_val, method="holdout", fit_time=fit_s, estimator=est)

    def compare(self, candidates: dict, X, y, next_logret, verbose: bool = False):
        reports = {}
        for name, est in candidates.items():
            if verbose:
                print(f"[cv] {name}", flush=True)
            reports[name] = self.evaluate(est, X, y, next_logret, name=name, verbose=verbose)
        return leaderboard(reports), reports


class ReturnEvaluator(DirectionEvaluator):
    """Walk-forward cross-validation of forward-return regressors (``predict``), primary ``spearman``.

    Parameters (on top of :class:`DirectionEvaluator`)
    ----------
    gap : int
        Default 274 bars = 16 calendar days of FX-week hours, ml-pipe's
        ``skip_days_between_train_test``. Must be at least ``offset + horizon`` so no training label
        overlaps a test bar.
    horizon : int
        Label horizon in bars; the trading read-out trades every ``horizon``-th row.
    """

    task = "return"

    def __init__(self, cv=5, gap: int = 274, max_train_size: int | None = None,
                 cost_bps: float = DEFAULT_COST_BPS, slices=DEFAULT_SLICES, horizon: int = 24):
        super().__init__(cv=cv, gap=gap, max_train_size=max_train_size, cost_bps=cost_bps, slices=slices)
        self.horizon = horizon

    def predict_scores(self, estimator, X) -> np.ndarray:
        return np.asarray(estimator.predict(X), dtype=float).ravel()

    def _target(self, y) -> np.ndarray:
        return np.asarray(y, float)

    def _metrics(self, y, scores, next_logret) -> dict[str, float]:
        return compute_return_metrics(y, scores, next_logret, self.cost_bps, self.horizon)

    def _report_kwargs(self) -> dict:
        return {"task": self.task, "horizon": self.horizon}


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def md_table(df: pd.DataFrame, digits: int = 4) -> str:
    """Plain markdown table (no ``tabulate`` dependency)."""
    df = df.copy()
    cols = [str(df.index.name or "")] + [str(c) for c in df.columns]

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "nan" if np.isnan(v) else f"{v:.{digits}f}"
        if isinstance(v, pd.Timestamp):
            return v.strftime("%Y-%m-%d %H:%M")
        return str(v)

    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for idx, *row in df.itertuples(name=None):
        lines.append("| " + " | ".join([fmt(idx)] + [fmt(v) for v in row]) + " |")
    return "\n".join(lines)


def style_table(df: pd.DataFrame, digits: int = 4, caption: str | None = None, highlight_best: bool = True):
    """Notebook Styler: numeric formatting, green best per known metric column."""
    num = df.select_dtypes("number").columns
    sty = df.style.format({c: f"{{:.{digits}f}}" for c in num})
    if highlight_best and len(df) > 1:
        for c in num:
            gib = METRICS.get(c)
            if gib is None:
                continue
            sty = (sty.highlight_max if gib else sty.highlight_min)(subset=[c], color="#c9ecd4")
    if "auc" in df:
        sty = sty.bar(subset=["auc"], color="#9ec5fe", vmin=0.45, vmax=0.6)
    if "spearman" in df:
        sty = sty.bar(subset=["spearman"], color="#9ec5fe", align="zero", vmin=-0.1, vmax=0.1)
    if caption:
        sty = sty.set_caption(caption)
    return sty


def style_leaderboard(board: pd.DataFrame, caption: str | None = None):
    if "spearman" in board:
        cols = ("spearman", "spearman_fold_std", "pearson", "r2", "mse", "mcc", "auc", "hit_rate",
                "strategy_bps", "sharpe", "n", "fit_time_s")
    else:
        cols = ("auc", "auc_fold_std", "log_loss", "brier", "accuracy", "balanced_accuracy",
                "strategy_bps", "sharpe", "turnover", "n", "fit_time_s")
    keep = [c for c in cols if c in board]
    return style_table(board[keep], caption=caption)


# ---------------------------------------------------------------------------
# displays
# ---------------------------------------------------------------------------
class EquityCurveDisplay:
    """Cumulative net log return of the sign strategy vs. buy-and-hold USDJPY."""

    def __init__(self, index, strategy, buy_hold, name=None, cost_bps=DEFAULT_COST_BPS):
        self.index, self.strategy, self.buy_hold = index, strategy, buy_hold
        self.name, self.cost_bps = name, cost_bps

    @classmethod
    def from_predictions(cls, p_up, next_logret, index, *, name=None, cost_bps=DEFAULT_COST_BPS, ax=None):
        net = strategy_returns(p_up, next_logret, cost_bps)
        disp = cls(pd.Index(index), np.cumsum(net), np.cumsum(np.asarray(next_logret, float)), name, cost_bps)
        return disp.plot(ax=ax)

    @classmethod
    def from_report(cls, report: "EvaluationReport", ax=None):
        """Direction: hourly sign strategy. Return: one ``sign(y_pred)`` trade every ``horizon`` rows."""
        if report.task != "return":
            return cls.from_predictions(report.y_pred, report.next_logret, report.index, name=report.name,
                                        cost_bps=report.cost_bps, ax=ax)
        rows, net = daily_strategy(report.y_pred, report.next_logret, report.horizon, report.cost_bps)
        bh = np.asarray(report.next_logret, float)[rows]
        return cls(report.index[rows], np.cumsum(net), np.cumsum(bh), report.name, report.cost_bps).plot(ax=ax)

    def plot(self, ax=None):
        import matplotlib.pyplot as plt

        ax = ax or plt.gca()
        ax.plot(self.index, self.strategy * 1e2, label=f"{self.name or 'model'} ({self.cost_bps}bp cost)", lw=1.2)
        ax.plot(self.index, self.buy_hold * 1e2, label="buy & hold", lw=1, color="#888888", alpha=0.8)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_ylabel("cumulative log return (%)")
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", labelrotation=30, labelsize=7)
        self.ax_, self.figure_ = ax, ax.figure
        return self


class ConfidenceDisplay:
    """Hit rate per ``|P(up) - 0.5|`` bucket: does the model know when it knows?"""

    def __init__(self, table: pd.DataFrame, name=None):
        self.table, self.name = table, name

    @classmethod
    def from_report(cls, report: EvaluationReport, ax=None):
        return cls(report.slice_table("confidence"), report.name).plot(ax=ax)

    def plot(self, ax=None):
        import matplotlib.pyplot as plt

        ax = ax or plt.gca()
        t = self.table
        col = "accuracy" if "accuracy" in t else "hit_rate"
        ax.bar(range(len(t)), t[col], color="#2f5597")
        for i, (acc, n) in enumerate(zip(t[col], t["n"])):
            ax.text(i, acc + 0.003, f"n={n}", ha="center", fontsize=7)
        ax.axhline(0.5, color="black", lw=0.6, ls="--")
        ax.set_xticks(range(len(t)), [str(x).split(" ")[0] for x in t.index])
        lo = min(0.45, float(t[col].min()) - 0.01)
        ax.set_ylim(lo, max(0.56, float(t[col].max()) + 0.02))
        ax.set_xlabel("|P(up) - 0.5|" if col == "accuracy" else "|predicted return| quartile")
        ax.set_ylabel("hit rate")
        self.ax_, self.figure_ = ax, ax.figure
        return self


class FeatureImportanceDisplay:
    """Gain importance of a fitted LightGBM pipeline (top ``k``)."""

    def __init__(self, importances: pd.Series):
        self.importances = importances

    @classmethod
    def from_estimator(cls, estimator, k: int = 20, ax=None):
        return cls(feature_importance(estimator).head(k)).plot(ax=ax)

    def plot(self, ax=None):
        import matplotlib.pyplot as plt

        ax = ax or plt.gca()
        s = self.importances.iloc[::-1]
        ax.barh(s.index, s.to_numpy(), color="#2f5597")
        ax.set_xlabel("gain (share)")
        ax.tick_params(axis="y", labelsize=7)
        self.ax_, self.figure_ = ax, ax.figure
        return self


def feature_importance(estimator) -> pd.Series:
    """Normalised gain importance from a fitted ``encode -> LightGBM`` pipeline (target wrapper ok)."""
    model = estimator[-1] if isinstance(estimator, Pipeline) else estimator
    model = getattr(model, "regressor_", model)          # TransformedTargetRegressor
    booster = getattr(model, "booster_", None)
    if booster is None:
        raise ValueError("feature importance needs a fitted LightGBM model")
    gain = pd.Series(booster.feature_importance("gain"), index=booster.feature_name(), dtype=float)
    return (gain / gain.sum()).sort_values(ascending=False)


class DecileDisplay:
    """Mean realised forward return per decile of the predicted return (monotone = useful ranking)."""

    def __init__(self, table: pd.DataFrame, name=None):
        self.table, self.name = table, name

    @classmethod
    def from_predictions(cls, y_true, y_pred, *, name=None, n_bins: int = 10, ax=None):
        df = pd.DataFrame({"y": np.asarray(y_true, float), "p": np.asarray(y_pred, float)})
        try:
            df["bin"] = pd.qcut(df["p"].rank(method="first"), n_bins, labels=False)
        except ValueError:
            df["bin"] = 0
        tab = df.groupby("bin").agg(pred_bp=("p", "mean"), real_bp=("y", "mean"), n=("y", "size"))
        tab[["pred_bp", "real_bp"]] *= 1e4
        return cls(tab, name).plot(ax=ax)

    def plot(self, ax=None):
        import matplotlib.pyplot as plt

        ax = ax or plt.gca()
        t = self.table
        ax.bar(t.index + 1, t["real_bp"], color=np.where(t["real_bp"] >= 0, "#2f5597", "#b35c00"))
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xlabel("predicted-return decile (1 = most bearish)")
        ax.set_ylabel("mean realised fwd return (bp)")
        self.ax_, self.figure_ = ax, ax.figure
        return self


class RollingCorrelationDisplay:
    """Rolling Spearman between predicted and realised forward returns (ml-pipe's inference reporter)."""

    def __init__(self, series: pd.Series, name=None):
        self.series, self.name = series, name

    @classmethod
    def from_report(cls, report: "EvaluationReport", window: int = 24 * 21, ax=None):
        df = pd.DataFrame({"y": np.asarray(report.y_true, float), "p": report.y_pred}, index=report.index)
        r = df.rank()                    # global ranks, rolling Pearson of ranks ~ rolling Spearman
        rho = r["y"].rolling(min(window, max(len(df) // 4, 10))).corr(r["p"])
        return cls(rho, report.name).plot(ax=ax)

    def plot(self, ax=None):
        import matplotlib.pyplot as plt

        ax = ax or plt.gca()
        ax.plot(self.series.index, self.series.to_numpy(), lw=1, color="#2f5597")
        ax.axhline(0, color="black", lw=0.6)
        ax.set_ylabel("rolling rank correlation")
        ax.tick_params(axis="x", labelrotation=30, labelsize=7)
        self.ax_, self.figure_ = ax, ax.figure
        return self


def plot_report(report: EvaluationReport):
    """2x2 panel. Direction: ROC, calibration, hit rate by confidence, equity curve.
    Return: realised return by predicted decile, rolling 21-day rank correlation, hit rate by
    |prediction| quartile, daily non-overlapping equity curve."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    if report.task == "return":
        o = report.overall
        DecileDisplay.from_predictions(report.y_true, report.y_pred, name=report.name, ax=axes[0, 0])
        axes[0, 0].set_title(f"realised vs predicted (Spearman {o['spearman']:+.4f})")
        RollingCorrelationDisplay.from_report(report, ax=axes[0, 1])
        axes[0, 1].set_title("rolling 21-day Spearman (hourly rows)")
        ConfidenceDisplay.from_report(report, ax=axes[1, 0])
        axes[1, 0].set_title(f"sign hit rate by |prediction| (overall {o['hit_rate']:.3f})")
        EquityCurveDisplay.from_report(report, ax=axes[1, 1])
        axes[1, 1].set_title(f"daily sign trade: {o['strategy_bps']:+.2f} bp/day, Sharpe {o['sharpe']:.2f}")
        fig.suptitle(f"{report.name} — {report.method}", fontsize=11)
        fig.tight_layout()
        return fig
    RocCurveDisplay.from_predictions(report.y_true, report.p_up, ax=axes[0, 0], name=report.name,
                                     plot_chance_level=True)
    axes[0, 0].set_title(f"ROC (AUC {report.auc:.4f})")
    CalibrationDisplay.from_predictions(report.y_true, report.p_up, n_bins=10, strategy="quantile",
                                        ax=axes[0, 1], name=report.name)
    axes[0, 1].set_title("calibration (quantile bins)")
    ConfidenceDisplay.from_report(report, ax=axes[1, 0])
    axes[1, 0].set_title("hit rate by confidence")
    EquityCurveDisplay.from_predictions(report.p_up, report.next_logret, report.index, name=report.name,
                                        cost_bps=report.cost_bps, ax=axes[1, 1])
    axes[1, 1].set_title(f"sign strategy: {report.overall['strategy_bps']:+.3f} bp/bar, "
                         f"Sharpe {report.overall['sharpe']:.2f}")
    fig.suptitle(f"{report.name} — {report.method}", fontsize=11)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------
def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def _jsonable(v):
    if isinstance(v, (pd.Timestamp, np.datetime64)):
        return str(v)
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, np.integer):
        return int(v)
    return v


def save_report(reports: dict[str, EvaluationReport], board: pd.DataFrame, output_dir, *, best: str,
                figures: bool = True, extra: dict | None = None, holdout: EvaluationReport | None = None) -> Path:
    """Write ``metrics.json`` (+ ``figures/<model>.png``) and return the JSON path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        **(extra or {}),
        "best": best,
        "leaderboard": {i: {k: _jsonable(v) for k, v in row.items()} for i, row in board.iterrows()},
        "folds": {n: [{k: _jsonable(v) for k, v in r.items()} for r in rep.folds.reset_index().to_dict("records")]
                  for n, rep in reports.items() if rep.folds is not None},
        "slices": {n: {s: rep.slice_table(s).reset_index().to_dict("records")
                       for s in [*rep.slices.columns, "confidence"]} for n, rep in reports.items()},
    }
    if holdout is not None:
        payload["holdout"] = {"name": holdout.name, "n": len(holdout.y_true),
                              "start": str(holdout.index[0]), "end": str(holdout.index[-1]),
                              **{k: _jsonable(v) for k, v in holdout.overall.items()}}
    path = output_dir / "metrics.json"
    path.write_text(json.dumps(payload, indent=2, default=_jsonable))
    if figures:
        import matplotlib.pyplot as plt

        fig_dir = output_dir / "figures"
        fig_dir.mkdir(exist_ok=True)
        for old in fig_dir.glob("*.png"):        # figures always describe the latest run only
            old.unlink()
        for name, rep in {**reports, **({f"holdout_{holdout.name}": holdout} if holdout else {})}.items():
            fig = plot_report(rep)
            fig.savefig(fig_dir / f"{_slug(name)}.png", dpi=110, bbox_inches="tight")
            plt.close(fig)
    return path
