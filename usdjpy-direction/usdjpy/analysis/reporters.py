"""Pre- and post-analysis reporters.

Each reporter is configuration-only in its constructor and exposes one ``display(...)``
method that renders a section through :mod:`usdjpy.analysis.display`. ``PreAnalysisReporter``
and ``PostAnalysisReporter`` compose them in order under ``##``-level headers.

``result`` is a :class:`usdjpy.pipeline.classification.ExecutionResult` (duck-typed here to
avoid a circular import).
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import pandas as pd

from ..data import ASSET_GROUP, BAR
from ..evaluation import (feature_importance, leaderboard, plot_report, style_leaderboard, style_table)
from ..features import asset_of, feature_type_of
from .display import display_dataframe, display_figure, display_info_box, display_markdown, display_title


class Reporter(Protocol):
    """Interface for components that display one part of a run."""

    def display(self, *args: Any, **context: Any) -> None: ...


def _plt():
    import matplotlib.pyplot as plt

    return plt


# ===========================================================================
# pre-analysis (development rows only, before any model is fitted)
# ===========================================================================
class CoverageReporter:
    """Bar count, date range, gaps (weekends, holidays, missing hours)."""

    def display(self, *, bars: pd.DataFrame, **_):
        step = bars.index.to_series().diff()
        gaps = step[step > BAR]
        hours = (gaps / BAR).round().astype(int)
        weekend = int((hours >= 40).sum())
        display_info_box(
            f"<b>{len(bars):,}</b> hourly bars {bars.index[0]:%Y-%m-%d %H:%M} → {bars.index[-1]:%Y-%m-%d %H:%M} UTC · "
            f"<b>{len(gaps)}</b> gaps ({weekend} weekend-sized ≥ 40h, {len(gaps) - weekend} shorter)")
        if len(gaps):
            tab = hours.value_counts().sort_index().rename("count").to_frame()
            tab.index = tab.index.map(lambda h: f"{h}h")
            tab.index.name = "gap length"
            display_dataframe(tab.head(15), "gap lengths (bars missing + 1)", highlight_best=False, bars=False)
        by_wd = pd.Series(bars.index.day_of_week).value_counts().sort_index()
        by_wd.index = [["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][i] for i in by_wd.index]
        display_dataframe(by_wd.rename("bars").to_frame().T, "bars per weekday", highlight_best=False, bars=False)


class TargetReporter:
    """Target distribution and up share (overall, by hour of day); for the forward-return task also
    the label autocorrelation (overlapping labels are strongly autocorrelated up to the horizon)."""

    def __init__(self, figsize=(11, 3.4), acf_lags: int = 120):
        self.figsize = figsize
        self.acf_lags = acf_lags

    def display(self, *, y: pd.Series, ret: pd.Series, X: pd.DataFrame, task: str = "direction", **_):
        what = "forward log return" if task == "return" else "next-hour return"
        r = ret * 1e4
        up = (ret > 0).astype(float)
        stats = pd.DataFrame({"mean_bp": [r.mean()], "std_bp": [r.std()], "skew": [r.skew()],
                              "kurtosis": [r.kurt()], "p01_bp": [r.quantile(0.01)], "p99_bp": [r.quantile(0.99)],
                              "up_share": [up.mean()], "n": [len(y)]}, index=pd.Index([what]))
        display_dataframe(stats, f"target: {what} (basis points) and up share", highlight_best=False, bars=False)
        if task == "return":
            lags = list(range(1, self.acf_lags + 1))
            acf = pd.Series([y.autocorr(k) for k in lags], index=lags)
            plt = _plt()
            fig, ax = plt.subplots(figsize=(self.figsize[0], 2.8))
            ax.bar(acf.index, acf.to_numpy(), color="#2f5597", width=1.0)
            ax.axhline(0, color="black", lw=0.6)
            ax.set_xlabel("lag (bars)")
            ax.set_title("label autocorrelation (overlapping windows → ACF decays to 0 at the horizon)")
            fig.tight_layout()
            display_figure(fig)
        y = up
        by_hour = pd.DataFrame({"up_share": y.groupby(X["hour"]).mean(),
                                "mean_bp": r.groupby(X["hour"]).mean(), "n": y.groupby(X["hour"]).size()})
        plt = _plt()
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        axes[0].hist(r.clip(r.quantile(0.005), r.quantile(0.995)), bins=80, color="#2f5597")
        axes[0].set_xlabel(f"{what} (bp, clipped 0.5–99.5%)")
        axes[0].set_title("return distribution")
        axes[1].bar(by_hour.index, by_hour["up_share"] - 0.5, color="#2f5597")
        axes[1].axhline(0, color="black", lw=0.6)
        axes[1].set_xlabel("hour (UTC)")
        axes[1].set_ylabel("up share − 0.5")
        axes[1].set_title("class balance by hour")
        fig.tight_layout()
        display_figure(fig)
        display_markdown(f"Up share {y.mean():.4f}; by hour it ranges "
                         f"{by_hour['up_share'].min():.3f}–{by_hour['up_share'].max():.3f} "
                         f"(±1σ sampling noise per hour ≈ {0.5 / np.sqrt(by_hour['n'].mean()):.3f}).")


class AutocorrelationReporter:
    """ACF of hourly returns and of absolute returns (direction vs. volatility predictability)."""

    def __init__(self, max_lag: int = 24, figsize=(11, 3.2)):
        self.max_lag = max_lag
        self.figsize = figsize

    def display(self, *, bars: pd.DataFrame, **_):
        r = np.log(bars["close"]).diff().dropna()
        lags = range(1, self.max_lag + 1)
        acf = pd.DataFrame({"ret": [r.autocorr(k) for k in lags], "abs_ret": [r.abs().autocorr(k) for k in lags]},
                           index=pd.Index(list(lags), name="lag"))
        band = 1.96 / np.sqrt(len(r))
        plt = _plt()
        fig, axes = plt.subplots(1, 2, figsize=self.figsize, sharex=True)
        for ax, col, title in zip(axes, acf.columns, ("returns", "|returns|")):
            ax.bar(acf.index, acf[col], color="#2f5597")
            ax.axhspan(-band, band, color="#cccccc", alpha=0.5)
            ax.set_title(f"ACF of hourly {title}")
            ax.set_xlabel("lag (bars)")
        fig.tight_layout()
        display_figure(fig)
        sig = acf.index[acf["ret"].abs() > band].tolist()
        display_markdown(f"Return ACF outside the ±{band:.4f} band at lags {sig or 'none'}; "
                         f"|return| ACF at lag 1 = {acf['abs_ret'].iloc[0]:.3f} (volatility clusters, "
                         f"direction mostly does not).")


class FeatureSummaryReporter:
    """Feature summary + Spearman correlation of each feature with the next-hour return."""

    def __init__(self, top: int = 15):
        self.top = top

    def display(self, *, X: pd.DataFrame, ret: pd.Series, task: str = "direction", **_):
        what = "forward return" if task == "return" else "next-hour return"
        desc = X.describe().T[["mean", "std", "min", "max"]]
        desc["missing"] = X.isna().mean()
        corr = X.corrwith(ret, method="spearman")
        desc["spearman_target"] = corr
        desc = desc.reindex(corr.abs().sort_values(ascending=False).index)
        display_dataframe(desc.head(self.top), f"top {self.top} of {X.shape[1]:,} features by |Spearman ρ| with the {what}",
                          highlight_best=False, bars=False)
        display_markdown(f"Largest |ρ| = {corr.abs().max():.4f} (n = {len(X):,}; ±2σ noise ≈ "
                         f"{2 / np.sqrt(len(X)):.4f}). Single features carry very little linear signal.")
        # ml-pipe Steps 8-9: importance grouped by asset and by feature type (mean |ρ| over the group)
        groups = pd.DataFrame({"abs_rho": corr.abs(), "asset": [asset_of(c) for c in corr.index],
                               "type": [feature_type_of(c) for c in corr.index]})
        if groups["asset"].nunique() > 1:
            by_asset = groups.groupby("asset")["abs_rho"].agg(["mean", "max", "size"]).sort_values("mean", ascending=False)
            by_asset["group"] = [ASSET_GROUP.get(a, "target") for a in by_asset.index]
            display_dataframe(by_asset, "mean |Spearman ρ| by asset (ml-pipe 'contemporary asset importance')",
                              highlight_best=False, bars=False)
        by_type = groups.groupby("type")["abs_rho"].agg(["mean", "max", "size"]).sort_values("mean", ascending=False)
        display_dataframe(by_type, "mean |Spearman ρ| by feature type", highlight_best=False, bars=False)


class PreAnalysisReporter:
    """Compose the pre-analysis sections."""

    def __init__(self, sections: dict[str, Reporter] | None = None):
        self.sections = sections if sections is not None else {
            "Coverage": CoverageReporter(),
            "Target": TargetReporter(),
            "Autocorrelation": AutocorrelationReporter(),
            "Features": FeatureSummaryReporter(),
        }

    def display(self, *, exp_name: str, **context):
        display_title("#", f"Pre-analysis — {exp_name}")
        for title, rep in self.sections.items():
            display_title("##", title)
            rep.display(**context)


# ===========================================================================
# post-analysis
# ===========================================================================
class DataReporter:
    """One-line run summary."""

    def display(self, result, **_):
        m = result.meta
        tgt = m.get("target", {})
        display_info_box(
            f"task <b>{m.get('task', 'direction')}</b> ({tgt.get('kind', 'next-bar direction')}) · {m['ticker']} · bars {m['bars_start'][:16]} → {m['bars_end'][:16]} · dev rows <b>{m['n_dev']:,}</b> "
            f"(up {m['dev_up_share']:.3f}) · holdout rows <b>{m['n_holdout']:,}</b> from {m['holdout_start'][:16]} "
            f"(up {m['holdout_up_share']:.3f}) · cv {m['cv']} · cost {m['cost_bps']} bp · seed {m['seed']} · "
            f"candidates {', '.join(m['candidates'])}")


class LeaderboardReporter:
    """Walk-forward leaderboard, per-fold scores of the winner, selected-model banner."""

    def __init__(self, show_folds: bool = True):
        self.show_folds = show_folds

    def display(self, result, **_):
        display_dataframe(style_leaderboard(result.board, caption=f"{result.meta['cv']} — out-of-fold"))
        best = result.best
        o = best.overall
        if best.task == "return":
            msg = (f"Selected model: <b>{result.best_name}</b> — CV Spearman <b>{o['spearman']:+.4f}</b>, "
                   f"Pearson {o['pearson']:+.4f}, R² {o['r2']:+.4f}, sign hit rate {o['hit_rate']:.4f} "
                   f"(MCC {o['mcc']:+.4f}), daily sign trade {o['strategy_bps']:+.2f} bp/day (Sharpe {o['sharpe']:.2f}).")
        else:
            msg = (f"Selected model: <b>{result.best_name}</b> — CV AUC <b>{o['auc']:.4f}</b>, accuracy "
                   f"{o['accuracy']:.4f}, log loss {o['log_loss']:.4f}, sign strategy {o['strategy_bps']:+.3f} bp/bar "
                   f"(Sharpe {o['sharpe']:.2f}).")
        display_info_box(msg, kind="success")
        if self.show_folds and best.folds is not None:
            cols = ["test_start", "test_end", "n_train", "n_test"] + [c for c in best.folds if c.startswith("test_")
                                                                      and c not in ("test_start", "test_end")]
            folds = best.folds[cols].copy()
            folds.columns = [c[5:] if c.startswith("test_") and c not in ("test_start", "test_end") else c
                             for c in folds.columns]
            display_dataframe(folds, f"{result.best_name}: per-fold scores", highlight_best=False, bars=False)


class TrainingReporter:
    """Chronological 80/20 fit of the non-baseline candidates + LightGBM validation curves."""

    def __init__(self, figsize=(11, 3.4)):
        self.figsize = figsize

    def display(self, result, **_):
        diag = result.diagnostics
        if not diag:
            display_info_box("diagnostics disabled for this run.", kind="warning")
            return
        display_dataframe(style_leaderboard(diag["holdout"], caption=diag["split"]))
        gen = diag.get("generalization")
        if gen is not None:
            keep = [c for c in gen.columns if c in diag["holdout"].columns and c != "fit_time_s"] + \
                   [c for c in gen.columns if c.startswith("in_sample_")]
            display_dataframe(gen[keep], "generalization gap: in-sample − out-of-sample on the same fit "
                                         "(ml-pipe Step 9; large positive = overfit)", highlight_best=False, bars=False)
        c = diag.get("lgbm")
        if not c:
            display_info_box("no LightGBM candidate — no training curves.", kind="warning")
            return
        plt = _plt()
        curves = c["curves"]
        fig, axes = plt.subplots(1, len(curves), figsize=self.figsize)
        for ax, (metric, cv) in zip(np.atleast_1d(axes), curves.items()):
            it = np.arange(1, len(cv["train"]) + 1)
            ax.plot(it, cv["train"], label="train")
            ax.plot(it, cv["valid"], label="valid")
            if metric == c["best_metric"]:
                ax.axvline(c["best_iter"], color="#888888", ls="--", lw=0.8)
            ax.set_title(metric)
            ax.set_xlabel("boosting iteration")
            ax.legend(fontsize=8)
        fig.tight_layout()
        display_figure(fig)
        display_markdown(f"`{c['name']}`: best validation {c['best_metric']} {c['best_valid']:.4f} at iteration "
                         f"{c['best_iter']} of {c['n_estimators']}. A widening train/valid gap means the extra "
                         "trees fit noise.")


class ExperimentsReporter:
    """Variants of the winner next to it (same walk-forward rows unless the evaluator differs)."""

    def display(self, result, **_):
        if not result.experiments:
            display_info_box("no experiments in this run.", kind="warning")
            return
        board = leaderboard({result.best_name: result.best, **result.experiments})
        display_dataframe(style_leaderboard(board, caption="experiments vs. the selected model"))


class SliceReporter:
    """Where does the winner do better or worse? Out-of-fold metrics per slice."""

    def __init__(self, slices=("session", "time_of_day", "weekday", "volatility", "regime", "confidence")):
        self.slices = slices

    def display(self, result, **_):
        best = result.best
        for s in self.slices:
            display_dataframe(style_table(best.slice_table(s), highlight_best=False),
                              f"{result.best_name} by {s}")
        conf = best.slice_table("confidence")
        col = "accuracy" if "accuracy" in conf else "hit_rate"
        what = "|P(up) − 0.5|" if col == "accuracy" else "the size of the predicted return"
        display_markdown("Read the confidence table first: a usable model's hit rate should rise with "
                         f"{what}. Here the top bucket hits {conf[col].iloc[-1]:.3f} "
                         f"on n = {int(conf['n'].iloc[-1])}.")


class ResidualReporter:
    """Residual diagnostics of the winner's out-of-fold predictions (return task; ml-pipe ResidualReporter).

    Residual ACF to ``acf_lags`` bars, rolling means (trend), rolling std (heteroskedasticity),
    Jarque-Bera normality and the ARCH-LM test via ``statsmodels`` when installed.
    """

    def __init__(self, acf_lags: int = 120, trend_windows=(240, 504, 1512), std_window: int = 504, figsize=(11, 3.2)):
        self.acf_lags = acf_lags
        self.trend_windows = trend_windows
        self.std_window = std_window
        self.figsize = figsize

    def display(self, result, **_):
        best = result.best
        if best.task != "return":
            display_info_box("residual diagnostics apply to the return task only.", kind="warning")
            return
        res = pd.Series(np.asarray(best.y_true, float) - best.y_pred, index=best.index, name="residual")
        n = len(res)
        lags = list(range(1, self.acf_lags + 1))
        acf = pd.Series([res.autocorr(k) for k in lags], index=lags)
        band = 1.96 / np.sqrt(n)
        plt = _plt()
        fig, axes = plt.subplots(1, 3, figsize=(self.figsize[0] * 1.2, self.figsize[1]))
        axes[0].bar(acf.index, acf.to_numpy(), color="#2f5597", width=1.0)
        axes[0].axhspan(-band, band, color="#cccccc", alpha=0.5)
        axes[0].set_title("residual ACF (overlapping labels ⇒ high to the horizon)")
        axes[0].set_xlabel("lag (bars)")
        for w in self.trend_windows:
            axes[1].plot(res.index, res.rolling(w).mean() * 1e4, lw=1, label=f"{w} bars")
        axes[1].axhline(0, color="black", lw=0.6)
        axes[1].set_title("rolling mean residual (bp): trend / bias")
        axes[1].legend(fontsize=7)
        axes[2].plot(res.index, res.rolling(self.std_window).std() * 1e4, lw=1, color="#2f5597")
        axes[2].set_title(f"rolling {self.std_window}-bar residual std (bp): heteroskedasticity")
        for ax in axes[1:]:
            ax.tick_params(axis="x", labelrotation=30, labelsize=7)
        fig.tight_layout()
        display_figure(fig)
        notes = [f"mean residual {res.mean() * 1e4:+.2f} bp (bias), std {res.std() * 1e4:.1f} bp, "
                 f"|ACF| beyond ±{band:.3f} at {int((acf.abs() > band).sum())}/{self.acf_lags} lags"]
        try:
            from scipy.stats import jarque_bera
            from statsmodels.stats.diagnostic import het_arch

            jb = jarque_bera(res.to_numpy())
            arch = het_arch(res.to_numpy(), nlags=24)
            notes.append(f"Jarque-Bera p = {jb.pvalue:.3g} (normality), ARCH-LM(24) p = {arch[1]:.3g} "
                         "(volatility clustering); p < 0.05 rejects.")
        except ImportError:
            notes.append("statsmodels not installed: Jarque-Bera / ARCH-LM skipped.")
        display_markdown(" ".join(notes))


class ImportanceReporter:
    """Gain importance of the refit LightGBM model."""

    def __init__(self, k: int = 20, figsize=(7, 5)):
        self.k = k
        self.figsize = figsize

    def display(self, result, **_):
        try:
            imp = feature_importance(result.model)
        except (ValueError, TypeError):
            display_info_box("the refit model is not LightGBM — no feature importance.", kind="warning")
            return
        plt = _plt()
        fig, ax = plt.subplots(figsize=self.figsize)
        s = imp.head(self.k).iloc[::-1]
        ax.barh(s.index, s.to_numpy(), color="#2f5597")
        ax.set_xlabel("gain share")
        ax.set_title(f"{result.best_name}: top {self.k} features")
        fig.tight_layout()
        display_figure(fig)


class HoldoutReporter:
    """The untouched final period: one score for the refit winner, compared with its CV score."""

    def display(self, result, **_):
        hold = result.holdout
        if hold is None:
            display_info_box("no holdout scoring in this run (--evaluate-only).", kind="warning")
            return
        cv = result.best.summary_frame().rename(index={result.best_name: f"{result.best_name} (CV)"})
        ho = hold.summary_frame().rename(index={hold.name: f"{hold.name} (holdout)"})
        display_dataframe(style_leaderboard(pd.concat([cv, ho]), caption=f"holdout {hold.index[0]:%Y-%m-%d} → "
                                                                         f"{hold.index[-1]:%Y-%m-%d}"))
        display_figure(plot_report(hold))
        if hold.task == "return":
            o = hold.overall
            rho = o["spearman"]
            kind = "success" if rho > 0.03 else ("warning" if rho > 0 else "error")
            display_info_box(f"Holdout Spearman <b>{rho:+.4f}</b> on {len(hold.y_true):,} hourly rows "
                             f"(≈ {len(hold.y_true) // hold.horizon} independent days); sign hit rate "
                             f"{o['hit_rate']:.3f}; daily sign trade {o['strategy_bps']:+.2f} bp/day after "
                             f"{hold.cost_bps} bp costs. Spearman near 0 means no usable return signal.", kind=kind)
            return
        auc = hold.auc
        kind = "success" if auc > 0.52 else ("warning" if auc > 0.5 else "error")
        display_info_box(f"Holdout AUC <b>{auc:.4f}</b> on {len(hold.y_true):,} bars; sign strategy "
                         f"{hold.overall['strategy_bps']:+.3f} bp/bar after {hold.cost_bps} bp costs. "
                         "AUC near 0.5 means no usable direction signal.", kind=kind)


class PostAnalysisReporter:
    """Compose the post-analysis sections."""

    def __init__(self, sections: dict[str, Reporter] | None = None):
        self.sections = sections if sections is not None else {
            "Run": DataReporter(),
            "Leaderboard": LeaderboardReporter(),
            "Diagnostics": TrainingReporter(),
            "Experiments": ExperimentsReporter(),
            "Slices": SliceReporter(),
            "Residuals": ResidualReporter(),
            "Feature importance": ImportanceReporter(),
            "Holdout": HoldoutReporter(),
        }

    def display(self, result, **context):
        display_title("#", f"Post-analysis — {result.exp_name}")
        for title, rep in self.sections.items():
            display_title("##", title)
            rep.display(result, **context)
