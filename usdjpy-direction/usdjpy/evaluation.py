"""Walk-forward evaluation for direction classifiers.

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
from sklearn.metrics import (RocCurveDisplay, accuracy_score, balanced_accuracy_score, brier_score_loss,
                             log_loss, roc_auc_score)
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


# name -> greater_is_better (None = descriptive only)
METRICS: dict[str, bool | None] = {
    "auc": True, "log_loss": False, "brier": False, "accuracy": True, "balanced_accuracy": True,
    "pred_up_share": None, "strategy_bps": True, "sharpe": True, "turnover": None,
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
        out = pd.DataFrame(index=X.index)
        for name in self.slices:
            if name == "session":
                out[name] = X[["session_tokyo", "session_london", "session_ny"]].apply(_session_label, axis=1)
            elif name == "time_of_day":
                out[name] = pd.cut(X["hour"], TOD_BINS, labels=TOD_LABELS).astype(str)
            elif name == "weekday":
                out[name] = X["weekday"].map(lambda d: WEEKDAYS[int(d)] if pd.notna(d) else "unknown")
            elif name == "volatility":
                v = X["ret_std_24"]
                try:
                    lab = pd.qcut(v, 3, labels=SLICE_ORDER["volatility"]).astype(object)
                except ValueError:
                    lab = pd.Series(np.nan, index=X.index, dtype=object)
                out[name] = lab.where(lab.notna(), "unknown")
            else:
                raise ValueError(f"unknown slice {name!r}; choose from {DEFAULT_SLICES}")
        return out.astype(str)


def confidence_labels(p_up) -> np.ndarray:
    edge = np.abs(np.asarray(p_up, dtype=float) - 0.5)
    return pd.cut(edge, CONF_BINS, labels=SLICE_ORDER["confidence"]).astype(str)


def slice_table(y_true, p_up, next_logret, labels, *, cost_bps: float = DEFAULT_COST_BPS,
                order: list | None = None) -> pd.DataFrame:
    """Per-group ``n, share, up_share, auc, accuracy, strategy_bps`` in ``order`` (else by n)."""
    df = pd.DataFrame({"y": np.asarray(y_true, int), "p": np.asarray(p_up, float),
                       "r": np.asarray(next_logret, float), "g": np.asarray(labels).astype(str)})
    rows = []
    for g, part in df.groupby("g", sort=False):
        m = compute_metrics(part.y, part.p, part.r, cost_bps)
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
    p_up: np.ndarray
    next_logret: np.ndarray
    slices: pd.DataFrame
    folds: pd.DataFrame | None = None
    fit_time: float = float("nan")
    method: str = "cv"
    cost_bps: float = DEFAULT_COST_BPS
    wall_time: float = float("nan")
    estimator_: object = None
    per_slice: dict[str, pd.DataFrame] = field(default_factory=dict)

    @property
    def auc(self) -> float:
        return self.overall[PRIMARY_METRIC]

    @property
    def index(self) -> pd.Index:
        return self.y_true.index

    def slice_table(self, name: str) -> pd.DataFrame:
        if name not in self.per_slice:
            labels = confidence_labels(self.p_up) if name == "confidence" else self.slices[name]
            self.per_slice[name] = slice_table(self.y_true, self.p_up, self.next_logret, labels,
                                               cost_bps=self.cost_bps, order=SLICE_ORDER.get(name))
        return self.per_slice[name]

    def summary_frame(self) -> pd.DataFrame:
        row = dict(self.overall)
        if self.folds is not None and "test_auc" in self.folds:
            row["auc_fold_std"] = float(self.folds["test_auc"].std(ddof=0))
        row["fit_time_s"] = self.fit_time
        row["n"] = len(self.y_true)
        return pd.DataFrame([row], index=pd.Index([self.name], name="model"))


def leaderboard(reports: dict[str, EvaluationReport]) -> pd.DataFrame:
    """Models sorted by ``auc`` (desc), ties broken by ``log_loss``."""
    board = pd.concat([r.summary_frame() for r in reports.values()])
    return board.sort_values([PRIMARY_METRIC, "log_loss"], ascending=[False, True])


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

    def score(self, name, y, p_up, next_logret, X, *, method="cv", folds=None, fit_time=float("nan"),
              estimator=None) -> EvaluationReport:
        y = pd.Series(np.asarray(y, int), index=X.index, name="y")
        p = np.asarray(p_up, float).ravel()
        r = np.asarray(next_logret, float).ravel()
        return EvaluationReport(name=name, overall=compute_metrics(y, p, r, self.cost_bps), y_true=y,
                                p_up=p, next_logret=r, slices=self._slicer().transform(X), folds=folds,
                                fit_time=fit_time, method=method, cost_bps=self.cost_bps, estimator_=estimator)

    def evaluate(self, estimator, X: pd.DataFrame, y, next_logret, name: str = "model",
                 verbose: bool = False) -> EvaluationReport:
        """Out-of-fold ``P(up)`` over the walk-forward test blocks (the first train block has none)."""
        t0 = time.perf_counter()
        y = np.asarray(y, int)
        r = np.asarray(next_logret, float)
        oof = np.full(len(X), np.nan)
        rows = []
        for k, (tr, te) in enumerate(self._cv().split(X)):
            est = clone(estimator)
            t1 = time.perf_counter()
            est.fit(X.iloc[tr], y[tr])
            fit_s = time.perf_counter() - t1
            p = est.predict_proba(X.iloc[te])[:, 1]
            oof[te] = p
            m = compute_metrics(y[te], p, r[te], self.cost_bps)
            rows.append({"fold": k, "train_start": X.index[tr[0]], "train_end": X.index[tr[-1]],
                         "test_start": X.index[te[0]], "test_end": X.index[te[-1]], "n_train": len(tr),
                         "n_test": len(te), "fit_time": fit_s, **{f"test_{n}": v for n, v in m.items()}})
            if verbose:
                print(f"  [{name}] fold {k}: auc={m['auc']:.4f} acc={m['accuracy']:.4f} "
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
        est.fit(X_tr, np.asarray(y_tr, int))
        fit_s = time.perf_counter() - t0
        p = est.predict_proba(X_val)[:, 1]
        return self.score(name, y_val, p, ret_val, X_val, method="holdout", fit_time=fit_s, estimator=est)

    def compare(self, candidates: dict, X, y, next_logret, verbose: bool = False):
        reports = {}
        for name, est in candidates.items():
            if verbose:
                print(f"[cv] {name}", flush=True)
            reports[name] = self.evaluate(est, X, y, next_logret, name=name, verbose=verbose)
        return leaderboard(reports), reports


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
    if caption:
        sty = sty.set_caption(caption)
    return sty


def style_leaderboard(board: pd.DataFrame, caption: str | None = None):
    keep = [c for c in ("auc", "auc_fold_std", "log_loss", "brier", "accuracy", "balanced_accuracy",
                        "strategy_bps", "sharpe", "turnover", "n", "fit_time_s") if c in board]
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
        ax.bar(range(len(t)), t["accuracy"], color="#2f5597")
        for i, (acc, n) in enumerate(zip(t["accuracy"], t["n"])):
            ax.text(i, acc + 0.003, f"n={n}", ha="center", fontsize=7)
        ax.axhline(0.5, color="black", lw=0.6, ls="--")
        ax.set_xticks(range(len(t)), t.index)
        lo = min(0.45, float(t["accuracy"].min()) - 0.01)
        ax.set_ylim(lo, max(0.56, float(t["accuracy"].max()) + 0.02))
        ax.set_xlabel("|P(up) - 0.5|")
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
    """Normalised gain importance from a fitted ``encode -> LGBMClassifier`` pipeline."""
    model = estimator[-1] if isinstance(estimator, Pipeline) else estimator
    booster = getattr(model, "booster_", None)
    if booster is None:
        raise ValueError("feature importance needs a fitted LightGBM model")
    gain = pd.Series(booster.feature_importance("gain"), index=booster.feature_name(), dtype=float)
    return (gain / gain.sum()).sort_values(ascending=False)


def plot_report(report: EvaluationReport):
    """2x2 panel: ROC, calibration, hit rate by confidence, equity curve."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
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
        for name, rep in {**reports, **({f"holdout_{holdout.name}": holdout} if holdout else {})}.items():
            fig = plot_report(rep)
            fig.savefig(fig_dir / f"{_slug(name)}.png", dpi=110, bbox_inches="tight")
            plt.close(fig)
    return path
