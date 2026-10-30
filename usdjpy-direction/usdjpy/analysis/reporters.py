"""Pre- and post-analysis reporters.

Each reporter is configuration-only in its constructor, carries a one-line ``subtitle`` (what the
section answers) and exposes one ``display(...)`` method that renders a section through
:mod:`usdjpy.analysis.display`. ``PreAnalysisReporter`` and ``PostAnalysisReporter`` compose them in
order under a banner and numbered ``##`` headers. Every figure is built with the helpers in
:mod:`usdjpy.style`, carries a title / subtitle / caption, and is saved under
``outputs/figures/<name>.png`` when the pipeline runs with ``figures=True``.

``result`` is a :class:`usdjpy.pipeline.classification.ExecutionResult` (duck-typed here to
avoid a circular import).
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import pandas as pd

from .. import style
from ..data import ASSET_GROUP, BAR
from ..evaluation import (BARS_PER_YEAR, REGIME_LABELS, FeatureImportanceDisplay, compute_metrics,
                          compute_return_metrics, daily_strategy, feature_importance, leaderboard, plot_report,
                          strategy_returns, style_leaderboard, style_table)
from ..features import asset_of, feature_type_of
from ..regimes import (DECILE_LABELS, DEFAULT_WINDOW, decile_regimes, fold_spans, joint_regimes, regime_stats)
from .display import (display_banner, display_caption, display_dataframe, display_figure, display_info_box,
                      display_info_grid, display_markdown, display_status, display_title)

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_REGIME_FILL = {k: v for k, v in style.REGIME_COLORS.items() if k != "unknown"}


class Reporter(Protocol):
    """Interface for components that display one part of a run."""

    subtitle: str

    def display(self, *args: Any, **context: Any) -> None: ...


# --------------------------------------------------------------------------- shared helpers
def _fwd(y: pd.Series, ret: pd.Series, task: str) -> pd.Series:
    """The return the task is about: the forward log return (return task) or the next-bar return."""
    return y.astype(float) if task == "return" else ret.astype(float)


def _hours(X: pd.DataFrame) -> pd.Series:
    if "hour" in X:
        return X["hour"].astype(int)
    return pd.Series(pd.DatetimeIndex(X.index).hour, index=X.index)


def _weekdays(X: pd.DataFrame) -> pd.Series:
    if "weekday" in X:
        return X["weekday"].astype(int)
    return pd.Series(pd.DatetimeIndex(X.index).day_of_week, index=X.index)


def _short(label: str) -> str:
    """Tick-friendly slice / regime label: ``"down-trend / high vol"`` -> ``"down\nhigh vol"``."""
    label = label.split(" (")[0]
    label = label.replace("down-trend", "down").replace("up-trend", "up").replace(" / ", "\n")
    return label.replace("+", "+\n") if len(label) > 9 and "+" in label else label


def _primary(report) -> tuple[str, float, float]:
    """(metric name, chance level, value) for the task of ``report``."""
    pm = report.primary_metric
    return pm, (0.5 if pm == "auc" else 0.0), float(report.overall[pm])


def _rolling_metric(report, window: int) -> tuple[pd.Series, pd.Series]:
    """Rolling rank correlation (return) / rolling corr(P(up), outcome) (direction), and rolling hit rate."""
    y = pd.Series(np.asarray(report.y_true, float), index=report.index)
    p = pd.Series(np.asarray(report.y_pred, float), index=report.index)
    w = min(window, max(len(y) // 4, 10))
    if report.task == "return":
        r = pd.DataFrame({"y": y, "p": p}).rank()
        corr = r["y"].rolling(w).corr(r["p"])
        hit = (np.sign(y) == np.sign(p)).astype(float).rolling(w).mean()
    else:
        corr = y.rolling(w).corr(p)
        hit = ((p > 0.5).astype(float) == y).astype(float).rolling(w).mean()
    return corr, hit


def _violin(ax, groups: list[np.ndarray], positions, color: str, width: float = 0.8, alpha: float = 0.75):
    data = [g[np.isfinite(g)] for g in groups]
    pos = [p for p, d in zip(positions, data) if len(d) > 1]
    data = [d for d in data if len(d) > 1]
    if not data:
        return None
    parts = ax.violinplot(data, positions=pos, widths=width, showmedians=True, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(alpha)
    parts["cmedians"].set_color(style.INK["primary"])
    parts["cmedians"].set_linewidth(1)
    return parts


# ===========================================================================
# pre-analysis (development rows only, before any model is fitted)
# ===========================================================================
class CoverageReporter:
    """Bar count, date range, gaps (weekends, holidays, missing hours), bars per week."""

    subtitle = "How much data there is, where the gaps are, and whether every week is complete."

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
        per_week = bars["close"].resample("W-SUN").size()
        by_wd = pd.Series(bars.index.day_of_week).value_counts().sort_index()
        by_wd.index = [WEEKDAYS[i] for i in by_wd.index]
        with style.plot_style():
            fig, axes = style.new_figure(1, 2, title="Coverage", height=3.0, width=11,
                                         subtitle="bars per calendar week (left) and per weekday (right); a full FX week has 120 hourly bars",
                                         gridspec_kw={"width_ratios": [3, 1]})
            ax = axes[0, 0]
            ax.fill_between(per_week.index, per_week.to_numpy(), step="mid", color=style.SERIES[0], alpha=0.85, lw=0)
            style.zero_line(ax, 120, ls=":")
            ax.set_ylabel("bars per week")
            style.date_axis(ax)
            style.vbar(axes[0, 1], by_wd, labels=[f"{v:,}" for v in by_wd])
            axes[0, 1].set_ylabel("bars")
            style.finish(fig)
        short = per_week[per_week < 100]
        display_figure(fig, name="pre_coverage", caption=f"{len(short)} week(s) below 100 bars (holidays or missing "
                                                          f"hours); forward-filled grid weeks all show 120.")


class MarketReporter:
    """USDJPY close over the whole history with the dev / holdout split, rolling volatility, drawdown."""

    subtitle = "What the market did over the sample: level, volatility clusters and drawdowns, with the holdout marked."

    def __init__(self, window: int = DEFAULT_WINDOW):
        self.window = window

    def display(self, *, bars: pd.DataFrame, all_bars: pd.DataFrame | None = None, holdout_start=None, **_):
        px = (all_bars if all_bars is not None else bars)["close"].astype(float)
        lr = np.log(px).diff()
        vol = lr.rolling(self.window).std() * np.sqrt(BARS_PER_YEAR) * 100
        dd = (px / px.cummax() - 1) * 100
        with style.plot_style():
            fig, axes = style.new_figure(3, 1, sharex=True, height=8.2, title="Market context",
                                         subtitle=f"close (top), rolling {self.window}-bar annualised volatility "
                                                  f"(middle), drawdown from the running high (bottom); shaded = holdout")
            axes[0, 0].plot(px.index, px.to_numpy(), color=style.SERIES[0], lw=1.2)
            axes[0, 0].set_ylabel("USDJPY close")
            axes[1, 0].plot(vol.index, vol.to_numpy(), color=style.SERIES[1], lw=1.2)
            style.zero_line(axes[1, 0], float(vol.median()), ls=":")
            axes[1, 0].set_ylabel("ann. vol (%)")
            axes[2, 0].fill_between(dd.index, dd.to_numpy(), 0, color=style.NEG, alpha=0.25, lw=0)
            axes[2, 0].plot(dd.index, dd.to_numpy(), color=style.NEG, lw=0.9)
            axes[2, 0].set_ylabel("drawdown (%)")
            if holdout_start is not None:
                for ax in axes[:, 0]:
                    ax.axvspan(pd.Timestamp(holdout_start), px.index[-1], color=style.INK["neutral"], alpha=0.7,
                               lw=0, zorder=0)
                axes[0, 0].text(pd.Timestamp(holdout_start), float(px.max()), " holdout", va="top", fontsize=8,
                                color=style.INK["muted"])
            style.date_axis(axes[2, 0])
            style.finish(fig)
        hi, lo = px.idxmax(), px.idxmin()
        display_figure(fig, name="pre_market",
                       caption=f"Range {px.min():.1f} ({lo:%Y-%m-%d}) → {px.max():.1f} ({hi:%Y-%m-%d}); worst drawdown "
                               f"{dd.min():.1f}% on {dd.idxmin():%Y-%m-%d}; median annualised vol {vol.median():.1f}% "
                               f"(max {vol.max():.1f}%).")


class TargetReporter:
    """Target distribution and up share (overall, by hour of day); for the forward-return task also
    the label autocorrelation (overlapping labels are strongly autocorrelated up to the horizon)."""

    subtitle = "What we are predicting: its distribution, how balanced it is by hour, and how overlapping labels are."

    def __init__(self, acf_lags: int = 120):
        self.acf_lags = acf_lags

    def display(self, *, y: pd.Series, ret: pd.Series, X: pd.DataFrame, task: str = "direction", **_):
        what = "forward log return" if task == "return" else "next-hour return"
        r = ret * 1e4
        up = (ret > 0).astype(float)
        stats = pd.DataFrame({"mean_bp": [r.mean()], "std_bp": [r.std()], "skew": [r.skew()],
                              "kurtosis": [r.kurt()], "p01_bp": [r.quantile(0.01)], "p99_bp": [r.quantile(0.99)],
                              "up_share": [up.mean()], "n": [len(y)]}, index=pd.Index([what]))
        display_dataframe(stats, f"target: {what} (basis points) and up share", highlight_best=False, bars=False)
        hours = _hours(X)
        by_hour = pd.DataFrame({"up_share": up.groupby(hours).mean(), "mean_bp": r.groupby(hours).mean(),
                                "n": up.groupby(hours).size()})
        with style.plot_style():
            fig, axes = style.new_figure(1, 2, title="Target", height=3.4,
                                         subtitle=f"{what} in basis points (left, clipped to 0.5–99.5%) and the up "
                                                  "share by hour of day relative to 50% (right)")
            ax = axes[0, 0]
            clipped = r.clip(r.quantile(0.005), r.quantile(0.995))
            ax.hist(clipped, bins=80, color=style.SERIES[0], alpha=0.85, density=True, lw=0)
            xs = np.linspace(float(clipped.min()), float(clipped.max()), 200)
            ax.plot(xs, np.exp(-0.5 * ((xs - r.mean()) / r.std()) ** 2) / (r.std() * np.sqrt(2 * np.pi)),
                    color=style.INK["secondary"], lw=1, ls=":", label="normal with the same mean / std")
            ax.legend(loc="upper left")
            ax.set_xlabel(f"{what} (bp)")
            ax.set_ylabel("density")
            ax = axes[0, 1]
            style.vbar(ax, (by_hour["up_share"] - 0.5), colors=style.sign_colors(by_hour["up_share"] - 0.5))
            style.band(ax, -0.5 / np.sqrt(by_hour["n"].mean()), 0.5 / np.sqrt(by_hour["n"].mean()))
            style.zero_line(ax)
            ax.set_xlabel("hour (UTC)")
            ax.set_ylabel("up share − 0.5")
            style.finish(fig)
        display_figure(fig, name="pre_target",
                       caption=f"Up share {up.mean():.4f}; by hour it ranges {by_hour['up_share'].min():.3f}–"
                               f"{by_hour['up_share'].max():.3f} (grey band = ±1σ sampling noise per hour ≈ "
                               f"{0.5 / np.sqrt(by_hour['n'].mean()):.3f}). Fat tails vs the normal overlay are the "
                               "usual FX picture.")
        if task == "return":
            lags = list(range(1, self.acf_lags + 1))
            acf = pd.Series([y.autocorr(k) for k in lags], index=lags)
            band = 1.96 / np.sqrt(len(y))
            with style.plot_style():
                fig, axes = style.new_figure(1, 1, title="Label autocorrelation", height=3.0,
                                             subtitle="ACF of the forward return label by lag in bars; overlapping windows "
                                                      "make neighbouring labels nearly identical")
                ax = axes[0, 0]
                ax.bar(acf.index, acf.to_numpy(), color=style.SERIES[0], width=1.0, lw=0)
                style.band(ax, -band, band)
                style.zero_line(ax)
                ax.set_xlabel("lag (bars)")
                ax.set_ylabel("ACF")
                style.finish(fig)
            display_figure(fig, name="pre_target_acf",
                           caption="The ACF decays to zero at the horizon: this is why the CV gap and the purge "
                                   "exist, and why only one trade per day is counted.")


class AutocorrelationReporter:
    """ACF of hourly returns and of absolute returns (direction vs. volatility predictability)."""

    subtitle = "Is there any linear memory in returns (direction) versus in their size (volatility)?"

    def __init__(self, max_lag: int = 24):
        self.max_lag = max_lag

    def display(self, *, bars: pd.DataFrame, **_):
        r = np.log(bars["close"]).diff().dropna()
        lags = range(1, self.max_lag + 1)
        acf = pd.DataFrame({"ret": [r.autocorr(k) for k in lags], "abs_ret": [r.abs().autocorr(k) for k in lags]},
                           index=pd.Index(list(lags), name="lag"))
        band = 1.96 / np.sqrt(len(r))
        with style.plot_style():
            fig, axes = style.new_figure(1, 2, sharex=True, height=3.2, title="Return autocorrelation",
                                         subtitle=f"hourly log returns (left) and their absolute values (right), "
                                                  f"grey band = ±1.96/√n = ±{band:.4f}")
            for ax, col, title in zip(axes[0], acf.columns, ("Returns", "|Returns|")):
                ax.bar(acf.index, acf[col], color=style.sign_colors(acf[col]), width=0.8, lw=0)
                style.band(ax, -band, band)
                style.zero_line(ax)
                ax.set_title(title)
                ax.set_xlabel("lag (bars)")
            style.finish(fig)
        sig = acf.index[acf["ret"].abs() > band].tolist()
        display_figure(fig, name="pre_acf",
                       caption=f"Return ACF outside the band at lags {sig or 'none'}; |return| ACF at lag 1 = "
                               f"{acf['abs_ret'].iloc[0]:.3f}: volatility clusters, direction mostly does not.")


class RegimeReporter:
    """Market regimes of the development sample: the joint trend × volatility grid (as used by the
    ``regime`` slice), ml-pipe's past / future trend and volatility deciles, and intraday seasonality."""

    subtitle = ("Which regimes the sample contains, how the target behaves in each, and whether the hour × weekday "
                "grid carries structure.")

    def __init__(self, window: int = DEFAULT_WINDOW, offset: int = 1):
        self.window = window
        self.offset = offset

    def display(self, *, bars: pd.DataFrame, X: pd.DataFrame, y: pd.Series, ret: pd.Series,
                task: str = "direction", **_):
        fwd = _fwd(y, ret, task)
        what = "forward return" if task == "return" else "next-hour return"
        reg = joint_regimes(bars, self.window)
        reg_x = reg.reindex(fwd.index).fillna("unknown")
        px = bars["close"].astype(float)
        # 1. timeline
        with style.plot_style():
            fig, axes = style.new_figure(1, 1, height=4.0, title="Regime timeline",
                                         subtitle=f"USDJPY close with the past-{self.window}-bar trend sign × volatility "
                                                  "tercile underneath (blues = up-trend, reds = down-trend, darker = more volatile)")
            ax = axes[0, 0]
            style.shade(ax, px.index, reg.to_numpy(), _REGIME_FILL, alpha=0.3)
            ax.plot(px.index, px.to_numpy(), color=style.INK["primary"], lw=1.0)
            ax.set_ylabel("USDJPY close")
            style.legend_patches(ax, _REGIME_FILL, order=REGIME_LABELS, ncol=6, loc="upper center",
                                 bbox_to_anchor=(0.5, -0.14), fontsize=7)
            style.date_axis(ax)
            style.finish(fig)
        stats = regime_stats(reg_x, fwd)
        display_figure(fig, name="pre_regimes_timeline",
                       caption="Spells are the same 21-day definitions the regime slice uses later, so model "
                               "performance by regime can be read against this picture.")
        # 2. distribution per joint regime
        order = [r for r in REGIME_LABELS if r in stats.index]
        with style.plot_style():
            fig, axes = style.new_figure(1, 1, height=3.6, title=f"{what.capitalize()} by regime",
                                         subtitle="distribution of the target (bp) in each joint regime; the line in each "
                                                  "violin is the median")
            ax = axes[0, 0]
            groups = [(fwd[reg_x == r] * 1e4).to_numpy() for r in order]
            for i, (r, g) in enumerate(zip(order, groups)):
                _violin(ax, [g], [i], style.REGIME_COLORS[r])
            ax.set_xticks(range(len(order)), [_short(r) for r in order], fontsize=7.5)
            style.zero_line(ax)
            ax.set_ylabel(f"{what} (bp)")
            lim = float(np.nanquantile(np.abs(fwd.to_numpy()) * 1e4, 0.995)) if len(fwd) else 1.0
            ax.set_ylim(-lim, lim)
            style.finish(fig)
        display_figure(fig, name="pre_regimes_dist",
                       caption="A regime whose median sits off zero is where a naive trend / mean-reversion bet "
                               "would have paid; the table gives the numbers.")
        display_dataframe(stats, "joint regimes: share of bars, target mean / std (bp), up share, mean spell",
                          highlight_best=False, bars=False)
        # 3. ml-pipe deciles
        panels = [("past", "trend"), ("past", "vol"), ("future", "trend"), ("future", "vol")]
        tables = {}
        with style.plot_style():
            fig, axes = style.new_figure(2, 2, height=6.4, title="ml-pipe regime deciles",
                                         subtitle=f"mean {what} (bp) by decile of the past (top) and future (bottom) "
                                                  f"{self.window}-bar trend (left) and volatility (right); whiskers = ±2 s.e.")
            for ax, (when, kind) in zip(axes.ravel(), panels):
                lab = decile_regimes(bars, self.window, self.offset, kind=kind, when=when).reindex(fwd.index)
                df = pd.DataFrame({"g": lab, "r": fwd * 1e4}).dropna()
                df = df[df["g"] != "unknown"]
                agg = df.groupby("g")["r"].agg(["mean", "std", "size", lambda s: (s > 0).mean()])
                agg.columns = ["mean_bp", "std_bp", "n", "up_share"]
                agg = agg.reindex([d for d in DECILE_LABELS if d in agg.index])
                tables[f"{when} {kind}"] = agg
                se = 2 * agg["std_bp"] / np.sqrt(agg["n"].clip(lower=1))
                style.vbar(ax, agg["mean_bp"], colors=style.sign_colors(agg["mean_bp"]), fmt="{:.1f}")
                ax.errorbar(np.arange(len(agg)), agg["mean_bp"], yerr=se, fmt="none", ecolor=style.INK["secondary"],
                            elinewidth=0.8, capsize=2)
                style.zero_line(ax)
                ax.set_title(f"{when} {kind} decile")
                ax.set_ylabel(f"mean {what} (bp)")
            style.finish(fig)
        display_figure(fig, name="pre_regimes_deciles",
                       caption="Past panels are what a model could condition on (a monotone pattern = exploitable "
                               "momentum / reversal); future panels only describe the label (a U-shape in future "
                               "volatility is mechanical: big moves either way).")
        wide = pd.concat({k: v[["mean_bp", "up_share", "n"]] for k, v in tables.items()}, axis=1)
        display_dataframe(wide, "regime deciles: mean target (bp), up share, n", highlight_best=False, bars=False)
        # 4. intraday seasonality
        hr = np.log(px).diff().reindex(fwd.index) * 1e4
        hours, wds = _hours(X).reindex(fwd.index), _weekdays(X).reindex(fwd.index)
        act = hr.abs().groupby([wds, hours]).mean().unstack()
        mean_fwd = (fwd * 1e4).groupby([wds, hours]).mean().unstack()
        for t in (act, mean_fwd):
            t.index = [WEEKDAYS[int(i)] for i in t.index]
            t.columns = [f"{int(c):02d}" for c in t.columns]
        with style.plot_style():
            fig, axes = style.new_figure(2, 1, height=6.0, title="Intraday seasonality",
                                         subtitle="hour (UTC) × weekday: mean |hourly return| in bp (top, activity) and "
                                                  f"mean {what} in bp (bottom)")
            style.heatmap(axes[0, 0], act, fmt="{:.0f}", fontsize=6.5, label="|ret| (bp)")
            axes[0, 0].set_title("Activity")
            style.heatmap(axes[1, 0], mean_fwd, cmap=style.cmap_diverging(), center=0.0, fmt="{:.0f}", fontsize=6.5,
                          label=f"{what} (bp)")
            axes[1, 0].set_title(f"Mean {what}")
            axes[1, 0].set_xlabel("hour (UTC)")
            style.finish(fig)
        display_figure(fig, name="pre_seasonality",
                       caption="Activity peaks in the London / New York overlap and dies on Friday evening; a "
                               "structured bottom panel would be a calendar effect the model can learn, a random one is noise.")


class CrossAssetReporter:
    """Correlation structure of the basket on daily returns, what USDJPY co-moves with over time, and how
    fresh each asset's prints are on the FX-week grid (v003+; skipped without basket columns)."""

    subtitle = "Which assets move with USDJPY, whether that changes over time, and how live each feed is."

    def __init__(self, window_days: int = 63, top: int = 6, horizon: int = 24):
        self.window_days = window_days
        self.top = top
        self.horizon = horizon

    def display(self, *, bars: pd.DataFrame, **_):
        closes = [c for c in bars.columns if c.startswith("close_")]
        if not closes:
            display_info_box("no basket columns in this arm — cross-asset section skipped.", kind="info")
            return
        names = ["USDJPY"] + [c[len("close_"):] for c in closes]
        px = pd.concat([bars["close"].rename("USDJPY")] + [bars[c].rename(c[len("close_"):]) for c in closes], axis=1)
        daily = np.log(px.astype(float)).diff(self.horizon).iloc[::self.horizon]          # non-overlapping
        groups = {n: ("target" if n == "USDJPY" else ASSET_GROUP.get(n, "other")) for n in names}
        order = sorted(names, key=lambda n: (list(style.GROUP_COLORS).index(groups[n]), n))
        corr = daily[order].corr()
        with style.plot_style():
            fig, axes = style.new_figure(1, 1, height=max(5.5, 0.28 * len(order) + 1.5), width=max(8, 0.32 * len(order) + 2),
                                         title="Basket correlation",
                                         subtitle=f"Pearson correlation of non-overlapping {self.horizon}-bar log returns; "
                                                  "tick colour = asset group")
            ax = axes[0, 0]
            style.heatmap(ax, corr, cmap=style.cmap_diverging(), center=0.0, fmt="{:.1f}", fontsize=5.5,
                          annotate=len(order) <= 32, label="ρ")
            ax.tick_params(axis="x", labelrotation=90)
            for lab in ax.get_xticklabels() + ax.get_yticklabels():
                lab.set_color(style.GROUP_COLORS[groups[lab.get_text()]])
            style.legend_patches(ax, {g: style.GROUP_COLORS[g] for g in dict.fromkeys(groups.values())},
                                 loc="upper left", bbox_to_anchor=(1.12, 1.0), fontsize=7)
            style.finish(fig)
        display_figure(fig, name="pre_crossasset_corr",
                       caption="JPY crosses and USD rates sit closest to USDJPY by construction; off-diagonal blocks "
                               "show which groups are redundant with one another.")
        rel = corr["USDJPY"].drop("USDJPY").abs().sort_values(ascending=False)
        top = [n for n in rel.index[:self.top]]
        with style.plot_style():
            fig, axes = style.new_figure(1, 1, height=3.6, title="What USDJPY moves with over time",
                                         subtitle=f"rolling {self.window_days}-day correlation of daily returns with the "
                                                  f"{len(top)} most correlated assets (by |ρ| over the whole sample)")
            ax = axes[0, 0]
            for i, n in enumerate(top):
                rc = daily["USDJPY"].rolling(self.window_days).corr(daily[n])
                ax.plot(rc.index, rc.to_numpy(), color=style.SERIES[i], lw=1.2, label=n)
            style.zero_line(ax)
            ax.set_ylabel("rolling correlation")
            ax.legend(ncol=len(top), loc="lower left", fontsize=7)
            style.date_axis(ax)
            style.finish(fig)
        display_figure(fig, name="pre_crossasset_rolling",
                       caption="Correlations that flip sign are the ones a tree model has to learn conditionally; "
                               "stable ones are the easy part of the cross-asset signal.")
        stale = [c for c in bars.columns if c.startswith("stale_")]
        if stale:
            fresh = pd.Series({c[len("stale_"):]: float((bars[c] <= 1.0).mean()) for c in stale}).sort_values()
            cols = [style.GROUP_COLORS[groups.get(n, "other")] for n in fresh.index]
            with style.plot_style():
                fig, axes = style.new_figure(1, 1, height=max(3.0, 0.22 * len(fresh) + 1), width=8,
                                             title="Basket freshness", subtitle="share of USDJPY grid hours on which the asset "
                                                                                 "printed within the last hour (the rest is forward-filled)")
                style.hbar(axes[0, 0], fresh, colors=cols, fmt="{:.0%}")
                axes[0, 0].set_xlim(0, 1.15)
                axes[0, 0].set_xlabel("fresh share")
                style.legend_patches(axes[0, 0], {g: style.GROUP_COLORS[g] for g in dict.fromkeys(groups.values())
                                                  if g != "target"}, loc="upper right", fontsize=7)
                style.finish(fig)
            display_figure(fig, name="pre_basket_freshness",
                           caption="Exchange-traded assets are live for roughly a third of the FX week; their "
                                   "features are stale the rest of the time, which the <asset>__stale column tells the model.")


class FeatureSummaryReporter:
    """Feature summary + Spearman correlation of each feature with the target."""

    subtitle = "Which single features and which asset / feature-type groups correlate with the target at all."

    def __init__(self, top: int = 15):
        self.top = top

    def display(self, *, X: pd.DataFrame, ret: pd.Series, y: pd.Series | None = None, task: str = "direction", **_):
        what = "forward return" if task == "return" else "next-hour return"
        target = _fwd(y, ret, task) if y is not None else ret
        desc = X.describe().T[["mean", "std", "min", "max"]]
        desc["missing"] = X.isna().mean()
        corr = X.corrwith(target, method="spearman")
        desc["spearman_target"] = corr
        desc = desc.reindex(corr.abs().sort_values(ascending=False).index)
        display_dataframe(desc.head(self.top), f"top {self.top} of {X.shape[1]:,} features by |Spearman ρ| with the {what}",
                          highlight_best=False, bars=False)
        noise = 2 / np.sqrt(len(X))
        display_markdown(f"Largest |ρ| = {corr.abs().max():.4f} (n = {len(X):,}; ±2σ noise ≈ {noise:.4f}). "
                         "Single features carry very little linear signal.")
        # ml-pipe Steps 8-9: importance grouped by asset and by feature type (mean |ρ| over the group)
        groups = pd.DataFrame({"abs_rho": corr.abs(), "rho": corr, "asset": [asset_of(c) for c in corr.index],
                               "type": [feature_type_of(c) for c in corr.index]})
        groups["group"] = [("target" if a == "USDJPY" else ASSET_GROUP.get(a, "other")) for a in groups["asset"]]
        multi = groups["asset"].nunique() > 1
        if multi:
            by_asset = groups.groupby("asset")["abs_rho"].agg(["mean", "max", "size"]).sort_values("mean", ascending=False)
            by_asset["group"] = [ASSET_GROUP.get(a, "target") for a in by_asset.index]
            display_dataframe(by_asset, "mean |Spearman ρ| by asset (ml-pipe 'contemporary asset importance')",
                              highlight_best=False, bars=False)
        by_type = groups.groupby("type")["abs_rho"].agg(["mean", "max", "size"]).sort_values("mean", ascending=False)
        display_dataframe(by_type, "mean |Spearman ρ| by feature type", highlight_best=False, bars=False)
        top = groups.sort_values("abs_rho", ascending=False).head(20)
        with style.plot_style():
            fig, axes = style.new_figure(1, 2, height=5.2, width=12, gridspec_kw={"width_ratios": [1.1, 1]},
                                         title="Feature–target correlation",
                                         subtitle=f"top 20 features by |Spearman ρ| with the {what} (left, colour = asset group) "
                                                  "and mean |ρ| by asset group × feature type (right)")
            ax = axes[0, 0]
            style.hbar(ax, top["rho"], colors=[style.GROUP_COLORS[g] for g in top["group"]], fmt="{:+.3f}")
            style.band(ax, -noise, noise)
            style.zero_line(ax)
            ax.tick_params(axis="y", labelsize=6.5)
            ax.set_xlabel("Spearman ρ")
            if multi:
                style.legend_patches(ax, {g: style.GROUP_COLORS[g] for g in dict.fromkeys(top["group"])},
                                     loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=6, fontsize=7)
            grid = groups.pivot_table(index="group", columns="type", values="abs_rho", aggfunc="mean")
            grid = grid.reindex([g for g in style.GROUP_COLORS if g in grid.index])
            grid = grid[grid.mean().sort_values(ascending=False).index]
            style.heatmap(axes[0, 1], grid, fmt="{:.3f}", fontsize=6, label="mean |ρ|")
            axes[0, 1].tick_params(axis="x", labelrotation=90, labelsize=7)
            style.finish(fig)
        display_figure(fig, name="pre_feature_rho",
                       caption="Everything inside the grey band is indistinguishable from noise on its own; the heatmap "
                               "shows which (group, type) cells are worth a selection step.")


class PreAnalysisReporter:
    """Compose the pre-analysis sections under a banner with numbered headers."""

    def __init__(self, sections: dict[str, Reporter] | None = None):
        self.sections = sections if sections is not None else {
            "Coverage": CoverageReporter(),
            "Market": MarketReporter(),
            "Target": TargetReporter(),
            "Autocorrelation": AutocorrelationReporter(),
            "Regimes": RegimeReporter(),
            "Cross-asset": CrossAssetReporter(),
            "Features": FeatureSummaryReporter(),
        }

    def display(self, *, exp_name: str, bars: pd.DataFrame, X: pd.DataFrame, task: str = "direction", **context):
        chips = {"task": task, "bars": f"{bars.index[0]:%Y-%m-%d} → {bars.index[-1]:%Y-%m-%d}",
                 "dev rows": f"{len(X):,}", "features": f"{X.shape[1]:,}"}
        display_banner(f"Pre-analysis — {exp_name}", chips,
                       subtitle="development rows only; nothing here has seen a model or the holdout")
        for i, (title, rep) in enumerate(self.sections.items(), 1):
            display_title("##", title, subtitle=getattr(rep, "subtitle", None), number=f"{i}.")
            rep.display(exp_name=exp_name, bars=bars, X=X, task=task, **context)


# ===========================================================================
# post-analysis
# ===========================================================================
class DataReporter:
    """Run record as an information grid."""

    subtitle = "What was run, on what, with which settings."

    def display(self, result, **_):
        m = result.meta
        tgt = m.get("target", {})
        items = {
            "experiment": m.get("exp_name", result.exp_name), "task": m.get("task", "direction"),
            "target": tgt.get("kind", "next-bar direction"), "ticker": m["ticker"],
            "bars": f"{m['bars_start'][:10]} → {m['bars_end'][:10]}",
            "dev rows": f"{m['n_dev']:,} (up {m['dev_up_share']:.3f})",
            "holdout rows": f"{m['n_holdout']:,} from {m['holdout_start'][:10]} (up {m['holdout_up_share']:.3f})",
            "features": f"{m.get('n_features', '?'):,}" if isinstance(m.get("n_features"), int) else "?",
            "basket": f"{len(m.get('basket_used', []))} assets" if m.get("basket_used") else "none",
            "validation": m["cv"], "purge": m.get("purge", 0), "cost": f"{m['cost_bps']} bp", "seed": m["seed"],
            "candidates": ", ".join(m["candidates"]), "wall time": f"{m.get('wall_time_s', '?')} s",
        }
        display_info_grid(items)


class LeaderboardReporter:
    """Walk-forward leaderboard, fold spread per model, selected-model banner."""

    subtitle = "Which candidate wins out-of-fold, and how much of that is one lucky fold."

    def __init__(self, show_folds: bool = True):
        self.show_folds = show_folds

    def _figure(self, reports: dict, best_name: str, name: str, title: str):
        first = next(iter(reports.values()))
        pm, chance, _ = _primary(first)
        board = leaderboard(reports)
        point = board[pm]
        folds = {n: r.folds[f"test_{pm}"].to_numpy() for n, r in reports.items()
                 if r.folds is not None and f"test_{pm}" in r.folds}
        lo = pd.Series({n: (folds[n].min() if n in folds else np.nan) for n in point.index})
        hi = pd.Series({n: (folds[n].max() if n in folds else np.nan) for n in point.index})
        colors = [style.color_for(n) for n in point.index]
        grid = pd.DataFrame({n: folds[n] for n in point.index if n in folds}).T
        grid.columns = [f"fold {i}" for i in range(grid.shape[1])]
        with style.plot_style():
            fig, axes = style.new_figure(1, 2, height=max(3.2, 0.32 * len(point) + 1.6), width=12,
                                         gridspec_kw={"width_ratios": [1.3, 1]}, title=title,
                                         subtitle=f"out-of-fold {pm} per model (dot) with the per-fold range (whisker), "
                                                  "and the same metric per fold (heatmap)")
            ax = axes[0, 0]
            style.dot_range(ax, point, lo, hi, colors=colors, highlight=best_name,
                            fmt="{:+.3f}" if chance == 0 else "{:.3f}")
            style.zero_line(ax)
            ax.axvline(chance, color=style.INK["axis"], lw=0.8)
            ax.set_xlabel(pm)
            if len(grid):
                style.heatmap(axes[0, 1], grid, cmap=style.cmap_diverging(), center=chance,
                              fmt="{:+.2f}" if chance == 0 else "{:.2f}", cbar=False)
                axes[0, 1].tick_params(axis="y", labelsize=7)
            else:
                axes[0, 1].set_axis_off()
            style.finish(fig)
        return fig

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
        fig = self._figure(result.reports, result.best_name, "post_leaderboard", "Leaderboard")
        pm = best.primary_metric
        spread = ""
        if best.folds is not None and f"test_{pm}" in best.folds:
            f = best.folds[f"test_{pm}"]
            spread = f" {result.best_name}: folds {', '.join(f'{v:+.3f}' for v in f)} (std {f.std(ddof=0):.3f})."
        display_figure(fig, name="post_leaderboard",
                       caption=f"A model whose whisker straddles chance in most folds has no stable edge.{spread}")
        if self.show_folds and best.folds is not None:
            cols = ["test_start", "test_end", "n_train", "n_test"] + [c for c in best.folds if c.startswith("test_")
                                                                      and c not in ("test_start", "test_end")]
            folds = best.folds[cols].copy()
            folds.columns = [c[5:] if c.startswith("test_") and c not in ("test_start", "test_end") else c
                             for c in folds.columns]
            display_dataframe(folds, f"{result.best_name}: per-fold scores", highlight_best=False, bars=False)


class TrainingReporter:
    """Chronological 80/20 fit of the non-baseline candidates, generalisation gap, LightGBM curves."""

    subtitle = "Does the model memorise? In-sample versus out-of-sample on one fit, and the boosting curves."

    def display(self, result, **_):
        diag = result.diagnostics
        if not diag:
            display_info_box("diagnostics disabled for this run.", kind="warning")
            return
        display_dataframe(style_leaderboard(diag["holdout"], caption=diag["split"]))
        gen = diag.get("generalization")
        pm = result.best.primary_metric
        if gen is not None:
            keep = [c for c in gen.columns if c in diag["holdout"].columns and c != "fit_time_s"] + \
                   [c for c in gen.columns if c.startswith("in_sample_")]
            display_dataframe(gen[keep], "generalization gap: in-sample − out-of-sample on the same fit "
                                         "(ml-pipe Step 9; large positive = overfit)", highlight_best=False, bars=False)
        c = diag.get("lgbm")
        curves = c["curves"] if c else {}
        n_panels = len(curves) + (1 if gen is not None else 0)
        if n_panels == 0:
            display_info_box("no LightGBM candidate — no training curves.", kind="warning")
            return
        with style.plot_style():
            fig, axes = style.new_figure(1, n_panels, height=3.4, width=min(12, 4.2 * n_panels),
                                         title="Training diagnostics",
                                         subtitle=f"{diag['split']}: LightGBM train / valid curves per boosting iteration"
                                                  + (f" and the in-sample vs out-of-sample {pm} per candidate" if gen is not None else ""))
            k = 0
            for metric, cv in curves.items():
                ax = axes[0, k]
                it = np.arange(1, len(cv["train"]) + 1)
                ax.plot(it, cv["train"], label="train", color=style.SERIES[0])
                ax.plot(it, cv["valid"], label="valid", color=style.SERIES[1])
                if metric == c["best_metric"]:
                    ax.axvline(c["best_iter"], color=style.INK["axis"], lw=0.8)
                    ax.text(c["best_iter"], ax.get_ylim()[1], f" best {c['best_iter']}", va="top", fontsize=7,
                            color=style.INK["muted"])
                ax.set_title(metric)
                ax.set_xlabel("boosting iteration")
                ax.legend()
                k += 1
            if gen is not None:
                ax = axes[0, k]
                ins = gen[f"in_sample_{pm}"] if f"in_sample_{pm}" in gen else None
                oos = diag["holdout"][pm].reindex(gen.index)
                y = np.arange(len(gen))[::-1]
                if ins is not None:
                    ax.hlines(y, oos.to_numpy(), ins.to_numpy(), color=style.INK["axis"], lw=2)
                    ax.scatter(ins.to_numpy(), y, color=style.SERIES[1], s=36, zorder=3, label="in-sample")
                ax.scatter(oos.to_numpy(), y, color=style.SERIES[0], s=36, zorder=3, label="out-of-sample")
                ax.set_yticks(y, list(gen.index))
                ax.grid(axis="y", visible=False)
                ax.axvline(0.5 if pm == "auc" else 0.0, color=style.INK["axis"], lw=0.8)
                ax.set_xlabel(pm)
                ax.set_title("Generalisation gap")
                ax.legend(loc="lower right")
            style.finish(fig)
        note = ""
        if c:
            note = (f" {c['name']}: best validation {c['best_metric']} {c['best_valid']:.4f} at iteration "
                    f"{c['best_iter']} of {c['n_estimators']}.")
        display_figure(fig, name="post_training",
                       caption=f"A widening train / valid gap means the extra trees fit noise; a long dumbbell means "
                               f"the fit memorises the training rows.{note}")


class ExperimentsReporter:
    """Variants of the winner next to it (same walk-forward rows unless the evaluator differs)."""

    subtitle = "Which single change helps or hurts the selected model, on identical folds."

    def display(self, result, **_):
        if not result.experiments:
            display_info_box("no experiments in this run.", kind="warning")
            return
        reports = {result.best_name: result.best, **result.experiments}
        board = leaderboard(reports)
        display_dataframe(style_leaderboard(board, caption="experiments vs. the selected model"))
        fig = LeaderboardReporter()._figure(reports, result.best_name, "post_experiments", "Experiments")
        display_figure(fig, name="post_experiments",
                       caption="Variants are ranked by the primary metric; a variant inside the winner's fold range "
                               "is not a real difference.")


class PerformanceTimelineReporter:
    """When does the selected model work? Rolling metrics, equity and the prediction band over time,
    with fold boundaries and the market regimes underneath, plus a year × month calendar."""

    subtitle = "When the edge appears and disappears, and whether it lines up with regimes or folds."

    def __init__(self, window: int = DEFAULT_WINDOW, regime_window: int = DEFAULT_WINDOW):
        self.window = window
        self.regime_window = regime_window

    def display(self, result, **_):
        best = result.best
        pm, chance, _ = _primary(best)
        idx = best.index
        corr, hit = _rolling_metric(best, self.window)
        y = pd.Series(np.asarray(best.y_true, float), index=idx)
        p = pd.Series(np.asarray(best.y_pred, float), index=idx)
        reg = joint_regimes(result.data.bars, self.regime_window).reindex(idx).fillna("unknown")
        spans = fold_spans(best.folds)
        with style.plot_style():
            fig, axes = style.new_figure(4, 1, sharex=True, height=10.5, title=f"{result.best_name}: performance over time",
                                         subtitle=f"out-of-fold rows; rolling {self.window}-bar windows; hairlines = fold "
                                                  "starts; regime bands under the top panel (blues up-trend, reds down-trend)")
            ax = axes[0, 0]
            style.shade(ax, idx, reg.to_numpy(), _REGIME_FILL, alpha=0.3)
            ax.fill_between(corr.index, 0, corr.to_numpy(), where=corr.to_numpy() >= 0, color=style.POS, alpha=0.2, lw=0)
            ax.fill_between(corr.index, 0, corr.to_numpy(), where=corr.to_numpy() < 0, color=style.NEG, alpha=0.2, lw=0)
            ax.plot(corr.index, corr.to_numpy(), color=style.INK["primary"], lw=1.0)
            style.zero_line(ax)
            ax.set_ylabel("rolling Spearman" if best.task == "return" else "rolling corr(P(up), y)")
            ax = axes[1, 0]
            ax.plot(hit.index, hit.to_numpy(), color=style.color_for(result.best_name), lw=1.1)
            style.zero_line(ax, 0.5)
            ax.set_ylabel("rolling hit rate")
            ax = axes[2, 0]
            if best.task == "return":
                rows, net = daily_strategy(best.y_pred, best.next_logret, best.horizon, best.cost_bps)
                eq_idx, eq = idx[rows], np.cumsum(net) * 1e2
                bh = np.cumsum(np.asarray(best.next_logret, float)[rows]) * 1e2
            else:
                net = strategy_returns(best.y_pred, best.next_logret, best.cost_bps)
                eq_idx, eq = idx, np.cumsum(net) * 1e2
                bh = np.cumsum(np.asarray(best.next_logret, float)) * 1e2
            peak = np.maximum.accumulate(eq)
            ax.fill_between(eq_idx, eq, peak, color=style.NEG, alpha=0.18, lw=0, label="drawdown")
            ax.plot(eq_idx, eq, color=style.color_for(result.best_name), label=f"sign trade ({best.cost_bps} bp cost)")
            ax.plot(eq_idx, bh, color=style.BENCHMARK, lw=1.2, label="buy & hold")
            style.zero_line(ax)
            ax.legend(loc="upper left", ncol=3)
            ax.set_ylabel("cumulative log return (%)")
            ax = axes[3, 0]
            scale = 1e4 if best.task == "return" else 1.0
            w = min(self.window, max(len(p) // 4, 10))
            q = pd.DataFrame({k: (p * scale).rolling(w).quantile(v) for k, v in (("p10", 0.1), ("p50", 0.5), ("p90", 0.9))})
            ax.fill_between(q.index, q["p10"], q["p90"], color=style.SERIES[0], alpha=0.2, lw=0, label="predicted p10–p90")
            ax.plot(q.index, q["p50"], color=style.SERIES[0], lw=1.0, label="predicted median")
            if best.task == "return":
                ax.plot(idx, (y * scale).rolling(w).mean(), color=style.SERIES[1], lw=1.0, label="realised rolling mean")
                style.zero_line(ax)
                ax.set_ylabel("forward return (bp)")
            else:
                ax.plot(idx, y.rolling(w).mean(), color=style.SERIES[1], lw=1.0, label="realised up share")
                style.zero_line(ax, 0.5)
                ax.set_ylabel("P(up)")
            ax.legend(loc="upper left", ncol=3)
            for a in axes[:, 0]:
                style.vlines(a, [s for s, _ in spans])
            style.date_axis(axes[3, 0])
            style.finish(fig)
        display_figure(fig, name="post_timeline",
                       caption="Read the top two panels against the regime bands: an edge that only exists in one colour "
                               "is a regime bet, not a model. The bottom panel shows whether predictions track the realised "
                               "level or are just scaled noise.")
        # calendar heatmaps
        if best.task == "return":
            rows, net = daily_strategy(best.y_pred, best.next_logret, best.horizon, best.cost_bps)
            daily = pd.Series(net * 1e4, index=idx[rows])
        else:
            daily = pd.Series(strategy_returns(best.y_pred, best.next_logret, best.cost_bps) * 1e4, index=idx)
        cal = daily.groupby([daily.index.year, daily.index.month]).mean().unstack()
        frame = pd.DataFrame({"y": y, "p": p})
        mcorr = frame.groupby([frame.index.year, frame.index.month]).apply(
            lambda d: d["y"].corr(d["p"], method="spearman") if len(d) > 10 else np.nan).unstack()
        for t in (cal, mcorr):
            t.columns = [f"{int(c):02d}" for c in t.columns]
            t.index = [str(i) for i in t.index]
        with style.plot_style():
            fig, axes = style.new_figure(1, 2, height=max(2.6, 0.5 * len(cal) + 1.6), width=12,
                                         title=f"{result.best_name}: calendar",
                                         subtitle=("mean net return per trade (bp) by year × month (left) and the monthly "
                                                   "Spearman between predicted and realised (right)"))
            style.heatmap(axes[0, 0], cal, cmap=style.cmap_diverging(), center=0.0, fmt="{:+.0f}", fontsize=7,
                          label="bp / trade")
            axes[0, 0].set_title("Net return per trade")
            style.heatmap(axes[0, 1], mcorr, cmap=style.cmap_diverging(), center=0.0, fmt="{:+.2f}", fontsize=7,
                          label="Spearman")
            axes[0, 1].set_title("Monthly rank correlation")
            for a in axes[0]:
                a.set_xlabel("month")
            style.finish(fig)
        pos_months = int((mcorr > 0).sum().sum())
        n_months = int(mcorr.notna().sum().sum())
        display_figure(fig, name="post_calendar",
                       caption=f"Positive rank correlation in {pos_months} of {n_months} months. A usable signal is "
                               "mostly blue on the right; a few deep cells carrying the whole average is the warning sign.")


class SliceReporter:
    """Where does the winner do better or worse? Out-of-fold metrics per slice."""

    subtitle = "Where the edge lives: by session, hour, weekday, volatility, regime and prediction confidence."

    def __init__(self, slices=("session", "time_of_day", "weekday", "volatility", "regime", "confidence")):
        self.slices = slices

    def display(self, result, **_):
        best = result.best
        pm, chance, _ = _primary(best)
        tables = {}
        for s in self.slices:
            tables[s] = best.slice_table(s)
            display_dataframe(style_table(tables[s], highlight_best=False), f"{result.best_name} by {s}")
        conf = tables.get("confidence", best.slice_table("confidence"))
        col = "accuracy" if "accuracy" in conf else "hit_rate"
        what = "|P(up) − 0.5|" if col == "accuracy" else "the size of the predicted return"
        metric = pm if pm in conf else col
        n = len(self.slices)
        ncols = 3 if n > 4 else n
        nrows = int(np.ceil(n / ncols))
        with style.plot_style():
            fig, axes = style.new_figure(nrows, ncols, height=3.0 * nrows + 0.6, width=12,
                                         title=f"{result.best_name}: {metric} by slice",
                                         subtitle=f"out-of-fold {metric} per group; grey band = ±1.96/√n around chance; "
                                                  "small text = rows in the group")
            for ax, (s, t) in zip(axes.ravel(), tables.items()):
                vals = (t[metric] - chance).rename(index=lambda i: _short(str(i)))
                style.vbar(ax, vals, colors=style.sign_colors(vals), labels=[f"n={int(v):,}" for v in t["n"]])
                se = 1.96 / np.sqrt(t["n"].clip(lower=1).to_numpy())
                ax.fill_between(np.arange(len(t)), -se, se, step="mid", color=style.INK["neutral"], alpha=0.8, lw=0)
                style.zero_line(ax)
                ax.set_title(s.replace("_", " "))
                ax.set_ylabel(f"{metric} − {chance:g}" if chance else metric)
                ax.tick_params(axis="x", labelsize=6.5)
            for ax in axes.ravel()[len(tables):]:
                ax.set_axis_off()
            style.finish(fig)
        display_figure(fig, name="post_slices",
                       caption="Read the confidence panel first: a usable model's hit rate should rise with "
                               f"{what}. Here the top bucket hits {conf[col].iloc[-1]:.3f} on n = {int(conf['n'].iloc[-1])}. "
                               "Bars inside the grey band are noise.")


class RegimePerformanceReporter:
    """Performance of the winner by market regime: the joint grid (metric, daily P&L, hit rate), ml-pipe's
    predicted-vs-realised distributions by past trend / volatility decile, and predicted vs realised."""

    subtitle = "In which market conditions the model's predictions are worth anything."

    def __init__(self, window: int = DEFAULT_WINDOW, offset: int = 1):
        self.window = window
        self.offset = offset

    def display(self, result, **_):
        best = result.best
        pm, chance, _ = _primary(best)
        idx = best.index
        y = np.asarray(best.y_true, float)
        p = np.asarray(best.y_pred, float)
        bars = result.data.bars
        reg = joint_regimes(bars, self.window).reindex(idx).fillna("unknown")
        order = [r for r in REGIME_LABELS if (reg == r).any()]
        rows = []
        if best.task == "return":
            trade_rows, net = daily_strategy(p, best.next_logret, best.horizon, best.cost_bps)
            trades = pd.Series(net * 1e4, index=idx[trade_rows])
        else:
            trades = pd.Series(strategy_returns(p, best.next_logret, best.cost_bps) * 1e4, index=idx)
        for r in order:
            m = reg.to_numpy() == r
            if best.task == "return":
                met = compute_return_metrics(y[m], p[m])
                hit = met["hit_rate"]
            else:
                met = compute_metrics(y[m].astype(int), p[m], np.asarray(best.next_logret, float)[m], best.cost_bps)
                hit = met["accuracy"]
            rows.append({"regime": r, "n": int(m.sum()), pm: met[pm], "hit_rate": hit,
                         "strategy_bps": float(trades[reg.reindex(trades.index).to_numpy() == r].mean())})
        tab = pd.DataFrame(rows).set_index("regime")
        display_dataframe(tab, f"{result.best_name} by joint regime (trend sign × volatility tercile, past "
                               f"{self.window} bars)", highlight_best=False, bars=False)
        unit = "bp/day" if best.task == "return" else "bp/bar"
        with style.plot_style():
            fig, axes = style.new_figure(1, 3, height=3.6, width=12, title=f"{result.best_name}: by market regime",
                                         subtitle=f"out-of-fold {pm} (left), net sign-trade return in {unit} (middle) and "
                                                  "hit rate (right) per joint regime; colour = regime")
            cols = [style.REGIME_COLORS[r] for r in tab.index]
            short = tab.rename(index=_short)
            style.vbar(axes[0, 0], short[pm] - chance, colors=cols, labels=[f"n={n:,}" for n in tab["n"]])
            se = 1.96 / np.sqrt(tab["n"].clip(lower=1).to_numpy())
            axes[0, 0].fill_between(np.arange(len(tab)), -se, se, step="mid", color=style.INK["neutral"], alpha=0.8, lw=0)
            axes[0, 0].set_ylabel(f"{pm} − {chance:g}" if chance else pm)
            style.vbar(axes[0, 1], short["strategy_bps"], colors=cols, fmt="{:+.1f}")
            axes[0, 1].set_ylabel(unit)
            style.vbar(axes[0, 2], short["hit_rate"] - 0.5, colors=cols)
            axes[0, 2].set_ylabel("hit rate − 0.5")
            for ax in axes[0]:
                style.zero_line(ax)
                ax.tick_params(axis="x", labelsize=7)
            style.finish(fig)
        display_figure(fig, name="post_regime_perf",
                       caption="A model that only pays in one regime is a regime detector; the useful case is a metric "
                               "above the band in most cells. Low-volatility cells have the smallest returns to capture.")
        if best.task != "return":
            return
        # ml-pipe: predicted vs realised distributions by past trend / vol decile
        with style.plot_style():
            fig, axes = style.new_figure(1, 2, height=3.8, width=12, title=f"{result.best_name}: ml-pipe regime deciles",
                                         subtitle=f"predicted (blue) vs realised (orange) forward return in bp by decile of the "
                                                  f"past {self.window}-bar trend (left) and volatility (right)")
            lim = float(np.nanquantile(np.abs(y) * 1e4, 0.99))
            dec_tables = {}
            for ax, kind in zip(axes[0], ("trend", "vol")):
                lab = decile_regimes(bars, self.window, self.offset, kind=kind, when="past").reindex(idx).fillna("unknown")
                decs = [d for d in DECILE_LABELS if (lab == d).any()]
                stats = []
                for i, d in enumerate(decs):
                    m = lab.to_numpy() == d
                    _violin(ax, [p[m] * 1e4], [i - 0.2], style.SERIES[0], width=0.38)
                    _violin(ax, [y[m] * 1e4], [i + 0.2], style.SERIES[1], width=0.38)
                    met = compute_return_metrics(y[m], p[m])
                    stats.append({"decile": d, "n": int(m.sum()), "spearman": met["spearman"], "hit_rate": met["hit_rate"],
                                  "mean_pred_bp": float(p[m].mean() * 1e4), "mean_real_bp": float(y[m].mean() * 1e4)})
                dec_tables[f"past {kind}"] = pd.DataFrame(stats).set_index("decile")
                ax.set_xticks(range(len(decs)), decs, fontsize=7)
                ax.set_ylim(-lim, lim)
                style.zero_line(ax)
                ax.set_title(f"past {kind} decile")
                ax.set_ylabel("forward return (bp)")
                style.legend_patches(ax, {"predicted": style.SERIES[0], "realised": style.SERIES[1]}, ncol=2,
                                     loc="upper center", bbox_to_anchor=(0.5, -0.12))
            style.finish(fig)
        display_figure(fig, name="post_regime_deciles",
                       caption="Predictions are shrunk towards zero by design; what matters is whether their median moves "
                               "with the realised median across deciles (ml-pipe's 'performance across regime' view).")
        display_dataframe(pd.concat(dec_tables, axis=1), "per decile: Spearman, hit rate, mean predicted / realised (bp)",
                          highlight_best=False, bars=False)
        # predicted vs realised density
        with style.plot_style():
            fig, axes = style.new_figure(1, 1, height=4.2, width=6.5, title=f"{result.best_name}: predicted vs realised",
                                         subtitle="density of out-of-fold rows (bp); dots = mean realised per predicted decile; "
                                                  "line = least-squares fit")
            ax = axes[0, 0]
            pb, yb = p * 1e4, y * 1e4
            ax.hexbin(pb, yb, gridsize=45, cmap=style.cmap_sequential(), mincnt=1, linewidths=0)
            df = pd.DataFrame({"p": pb, "y": yb})
            df["bin"] = pd.qcut(df["p"].rank(method="first"), 10, labels=False)
            g = df.groupby("bin").mean()
            ax.scatter(g["p"], g["y"], color=style.SERIES[1], s=36, zorder=3, edgecolor=style.INK["surface"])
            slope, inter = np.polyfit(pb, yb, 1)
            xs = np.array([np.nanquantile(pb, 0.005), np.nanquantile(pb, 0.995)])
            ax.plot(xs, inter + slope * xs, color=style.INK["primary"], lw=1, ls=":")
            ax.set_xlim(*xs)
            ax.set_ylim(-lim, lim)
            style.zero_line(ax)
            ax.axvline(0, color=style.INK["axis"], lw=0.8)
            ax.set_xlabel("predicted (bp)")
            ax.set_ylabel("realised (bp)")
            style.finish(fig)
        display_figure(fig, name="post_pred_vs_real",
                       caption=f"Slope {slope:.2f}: realised bp per predicted bp (1 = calibrated, 0 = no information). "
                               "The decile dots should climb from left to right.")


class ResidualReporter:
    """Residual diagnostics of the winner's out-of-fold predictions (return task; ml-pipe ResidualReporter).

    Residual ACF, rolling means (trend), rolling std (heteroskedasticity), Q–Q plot, residual vs fitted,
    CUSUM drift, Jarque-Bera normality and the ARCH-LM test via ``statsmodels`` when installed.
    """

    subtitle = "What the model leaves behind: bias, drift, volatility clustering and fat tails in the residuals."

    def __init__(self, acf_lags: int = 120, trend_windows=(240, 504, 1512), std_window: int = 504):
        self.acf_lags = acf_lags
        self.trend_windows = trend_windows
        self.std_window = std_window

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
        with style.plot_style():
            fig, axes = style.new_figure(2, 3, height=6.6, width=12, title=f"{result.best_name}: residuals",
                                         subtitle="residual = realised − predicted forward return, out-of-fold rows; "
                                                  "basis points where shown")
            ax = axes[0, 0]
            ax.bar(acf.index, acf.to_numpy(), color=style.SERIES[0], width=1.0, lw=0)
            style.band(ax, -band, band)
            style.zero_line(ax)
            ax.set_title("ACF (overlapping labels: high to the horizon)")
            ax.set_xlabel("lag (bars)")
            ax = axes[0, 1]
            for i, w in enumerate(self.trend_windows):
                ax.plot(res.index, res.rolling(w).mean() * 1e4, lw=1, label=f"{w} bars", color=style.SERIES[i])
            style.zero_line(ax)
            ax.set_title("Rolling mean (bp): trend / bias")
            ax.legend()
            style.date_axis(ax)
            ax = axes[0, 2]
            ax.plot(res.index, res.rolling(self.std_window).std() * 1e4, lw=1, color=style.SERIES[0])
            ax.set_title(f"Rolling {self.std_window}-bar std (bp): heteroskedasticity")
            style.date_axis(ax)
            ax = axes[1, 0]
            from scipy import stats as sps

            (osm, osr), (slope, intercept, _) = sps.probplot(res.to_numpy() * 1e4, dist="norm")
            ax.scatter(osm, osr, s=6, color=style.SERIES[0], alpha=0.6, lw=0)
            ax.plot(osm, intercept + slope * osm, color=style.INK["primary"], lw=1, ls=":")
            ax.set_title("Q–Q vs normal")
            ax.set_xlabel("theoretical quantile")
            ax.set_ylabel("residual (bp)")
            ax = axes[1, 1]
            ax.hexbin(best.y_pred * 1e4, res.to_numpy() * 1e4, gridsize=40, cmap=style.cmap_sequential(), mincnt=1,
                      linewidths=0)
            style.zero_line(ax)
            ax.set_title("Residual vs fitted")
            ax.set_xlabel("predicted (bp)")
            ax.set_ylabel("residual (bp)")
            ax = axes[1, 2]
            cusum = (res - res.mean()).cumsum() * 1e4
            ax.fill_between(cusum.index, 0, cusum.to_numpy(), color=style.SERIES[0], alpha=0.2, lw=0)
            ax.plot(cusum.index, cusum.to_numpy(), color=style.SERIES[0], lw=1)
            style.zero_line(ax)
            ax.set_title("CUSUM of residuals (bp): drift")
            style.date_axis(ax)
            style.finish(fig)
        notes = [f"mean residual {res.mean() * 1e4:+.2f} bp (bias), std {res.std() * 1e4:.1f} bp, "
                 f"|ACF| beyond ±{band:.3f} at {int((acf.abs() > band).sum())}/{self.acf_lags} lags."]
        try:
            from scipy.stats import jarque_bera
            from statsmodels.stats.diagnostic import het_arch

            jb = jarque_bera(res.to_numpy())
            arch = het_arch(res.to_numpy(), nlags=24)
            notes.append(f"Jarque-Bera p = {jb.pvalue:.3g} (normality), ARCH-LM(24) p = {arch[1]:.3g} "
                         "(volatility clustering); p < 0.05 rejects.")
        except ImportError:
            notes.append("statsmodels not installed: Jarque-Bera / ARCH-LM skipped.")
        display_figure(fig, name="post_residuals",
                       caption="A CUSUM that wanders far from zero is a slowly drifting bias; a fanning residual-vs-fitted "
                               "cloud means the model's confidence does not scale with realised size.")
        display_markdown(" ".join(notes))


class ImportanceReporter:
    """Gain importance of the refit LightGBM model: top features, by asset, by feature type."""

    subtitle = "What the refit model actually uses, and whether that is one asset or many."

    def __init__(self, k: int = 20):
        self.k = k

    def display(self, result, **_):
        try:
            imp = feature_importance(result.model)
        except (ValueError, TypeError):
            display_info_box("the refit model is not LightGBM — no feature importance.", kind="warning")
            return
        assets = pd.Series([asset_of(c) for c in imp.index], index=imp.index)
        types = pd.Series([feature_type_of(c) for c in imp.index], index=imp.index)
        by_asset = imp.groupby(assets).sum().sort_values(ascending=False)
        if len(by_asset) > 12:
            by_asset = pd.concat([by_asset.iloc[:12], pd.Series({"Other": by_asset.iloc[12:].sum()})])
        by_type = imp.groupby(types).sum().sort_values(ascending=False)
        with style.plot_style():
            fig, axes = style.new_figure(1, 3, height=5.4, width=13, gridspec_kw={"width_ratios": [1.4, 1, 1]},
                                         title=f"{result.best_name}: feature importance",
                                         subtitle=f"share of LightGBM gain: top {self.k} features (colour = asset group), "
                                                  "by asset, by feature type")
            FeatureImportanceDisplay(imp.head(self.k)).plot(ax=axes[0, 0])
            axes[0, 0].set_title(f"Top {self.k}")
            cols = [style.GROUP_COLORS["target" if a == "USDJPY" else ASSET_GROUP.get(a, "other")] for a in by_asset.index]
            style.hbar(axes[0, 1], by_asset, colors=cols, fmt="{:.1%}")
            axes[0, 1].set_title("By asset")
            style.hbar(axes[0, 2], by_type, fmt="{:.1%}")
            axes[0, 2].set_title("By feature type")
            for ax in axes[0, 1:]:
                ax.set_xlabel("gain share")
            groups_used = dict.fromkeys("target" if a == "USDJPY" else ASSET_GROUP.get(a, "other") for a in by_asset.index)
            style.legend_patches(axes[0, 1], {g: style.GROUP_COLORS[g] for g in groups_used}, loc="center right", fontsize=7)
            style.finish(fig)
        display_figure(fig, name="post_importance",
                       caption=f"Top feature carries {imp.iloc[0]:.1%} of the gain; the top {self.k} together "
                               f"{imp.head(self.k).sum():.1%}. Diffuse importance over thousands of columns is the "
                               "fingerprint of memorisation rather than of a few real drivers.")


class HoldoutReporter:
    """The untouched final period: one score for the refit winner, compared with its CV score, with a
    timeline of daily predictions against what happened."""

    subtitle = "The one number that counts: the refit model on rows it has never seen, against its CV score."

    def display(self, result, **_):
        hold = result.holdout
        if hold is None:
            display_info_box("no holdout scoring in this run (--evaluate-only).", kind="warning")
            return
        cv = result.best.summary_frame().rename(index={result.best_name: f"{result.best_name} (CV)"})
        ho = hold.summary_frame().rename(index={hold.name: f"{hold.name} (holdout)"})
        both = pd.concat([cv, ho])
        display_dataframe(style_leaderboard(both, caption=f"holdout {hold.index[0]:%Y-%m-%d} → {hold.index[-1]:%Y-%m-%d}"))
        display_figure(plot_report(hold), caption="The same four panels as for every CV model, on the holdout rows.")
        pm, chance, _ = _primary(hold)
        idx = hold.index
        p = np.asarray(hold.y_pred, float)
        y = np.asarray(hold.y_true, float)
        with style.plot_style():
            fig, axes = style.new_figure(2, 1, sharex=True, height=6.0, title=f"{hold.name}: holdout timeline",
                                         subtitle=("one trade per day: predicted (dots) vs realised (bars) forward return "
                                                   "in bp, and the resulting equity vs buy & hold" if hold.task == "return"
                                                   else "P(up) (dots) vs the realised direction (bars), and the resulting equity"))
            ax = axes[0, 0]
            if hold.task == "return":
                rows, net = daily_strategy(p, hold.next_logret, hold.horizon, hold.cost_bps)
                t, pr, yr = idx[rows], p[rows] * 1e4, y[rows] * 1e4
                ax.bar(t, yr, width=0.8, color=style.sign_colors(yr), alpha=0.6, lw=0, label="realised")
                ax.scatter(t, pr, s=10, color=style.INK["primary"], zorder=3, label="predicted")
                ax.set_ylabel("forward return (bp)")
                style.zero_line(ax)
                eq_idx, eq = t, np.cumsum(net) * 1e2
                bh = np.cumsum(np.asarray(hold.next_logret, float)[rows]) * 1e2
            else:
                step = max(1, len(idx) // 400)
                t, pr, yr = idx[::step], p[::step], y[::step]
                ax.bar(t, yr - 0.5, width=0.04 * step, color=style.sign_colors(yr - 0.5), alpha=0.5, lw=0, label="realised up")
                ax.scatter(t, pr - 0.5, s=6, color=style.INK["primary"], zorder=3, label="P(up) − 0.5")
                ax.set_ylabel("P(up) − 0.5")
                style.zero_line(ax)
                net = strategy_returns(p, hold.next_logret, hold.cost_bps)
                eq_idx, eq = idx, np.cumsum(net) * 1e2
                bh = np.cumsum(np.asarray(hold.next_logret, float)) * 1e2
            ax.legend(loc="upper left", ncol=2)
            ax = axes[1, 0]
            peak = np.maximum.accumulate(eq)
            ax.fill_between(eq_idx, eq, peak, color=style.NEG, alpha=0.18, lw=0, label="drawdown")
            ax.plot(eq_idx, eq, color=style.color_for(hold.name), label=f"sign trade ({hold.cost_bps} bp cost)")
            ax.plot(eq_idx, bh, color=style.BENCHMARK, lw=1.2, label="buy & hold")
            style.zero_line(ax)
            ax.set_ylabel("cumulative log return (%)")
            ax.legend(loc="upper left", ncol=3)
            style.date_axis(ax)
            style.finish(fig)
        display_figure(fig, name="post_holdout_timeline",
                       caption="Dots far from zero on the wrong side of a bar are the costly days; a flat equity line "
                               "with a positive metric means the ranking is right but the sign bets are not.")
        o = hold.overall
        if hold.task == "return":
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
                         f"{o['strategy_bps']:+.3f} bp/bar after {hold.cost_bps} bp costs. "
                         "AUC near 0.5 means no usable direction signal.", kind=kind)


class VerdictReporter:
    """One status box: is there a signal, is it stable across folds, does it survive the holdout and costs?"""

    subtitle = "The sentence a reader needs."

    def display(self, result, **_):
        best = result.best
        pm, chance, cv_val = _primary(best)
        hold = result.holdout
        ho_val = float(hold.overall[pm]) if hold is not None else np.nan
        folds = best.folds[f"test_{pm}"] if best.folds is not None and f"test_{pm}" in best.folds else None
        gen = (result.diagnostics or {}).get("generalization")
        gap = float(gen.loc[result.best_name, pm]) if gen is not None and result.best_name in gen.index else np.nan
        pnl_cv = float(best.overall["strategy_bps"])
        pnl_ho = float(hold.overall["strategy_bps"]) if hold is not None else np.nan
        edge = 0.03 if pm == "spearman" else 0.02
        checks = {
            "CV edge": cv_val - chance > edge,
            "holdout edge": (ho_val - chance > edge) if hold is not None else False,
            "fold stability": bool(folds is not None and (folds - chance).min() > -edge) if folds is not None else False,
            "P&L positive (CV and holdout)": pnl_cv > 0 and (pnl_ho > 0 if hold is not None else False),
            "no memorisation": bool(gap < 0.5) if not np.isnan(gap) else False,
        }
        passed = sum(checks.values())
        status = "good" if passed >= 4 else ("warning" if passed >= 2 else "serious")
        unit = "bp/day" if best.task == "return" else "bp/bar"
        parts = [f"<b>{result.best_name}</b>: CV {pm} <b>{cv_val:+.4f}</b>"
                 + (f", holdout <b>{ho_val:+.4f}</b>" if hold is not None else "")
                 + (f", folds {folds.min():+.3f} … {folds.max():+.3f}" if folds is not None else "")
                 + f", daily sign trade {pnl_cv:+.2f} {unit} (CV)"
                 + (f" / {pnl_ho:+.2f} {unit} (holdout)" if hold is not None else "")
                 + (f", in-sample − out-of-sample gap {gap:+.2f}" if not np.isnan(gap) else "") + "."]
        marks = " · ".join(f"{'✔' if v else '✖'} {k}" for k, v in checks.items())
        verdict = {"good": "A real, stable research signal; next step is sizing and costs.",
                   "warning": "A weak or unstable signal: worth another arm, not worth trading.",
                   "serious": "No usable signal at this stage."}[status]
        display_status(" ".join(parts) + f"<br><span style='font-size:0.9em'>{marks}</span><br>{verdict}", status,
                       title=f"Verdict: {passed}/{len(checks)} checks pass")


class PostAnalysisReporter:
    """Compose the post-analysis sections under a banner with numbered headers."""

    def __init__(self, sections: dict[str, Reporter] | None = None):
        self.sections = sections if sections is not None else {
            "Run": DataReporter(),
            "Leaderboard": LeaderboardReporter(),
            "Diagnostics": TrainingReporter(),
            "Experiments": ExperimentsReporter(),
            "Performance over time": PerformanceTimelineReporter(),
            "Slices": SliceReporter(),
            "Regime performance": RegimePerformanceReporter(),
            "Residuals": ResidualReporter(),
            "Feature importance": ImportanceReporter(),
            "Holdout": HoldoutReporter(),
            "Verdict": VerdictReporter(),
        }

    def display(self, result, **context):
        m = result.meta
        pm = result.best.primary_metric
        chips = {"task": m.get("task", "direction"), "winner": result.best_name,
                 f"CV {pm}": f"{result.best.overall[pm]:+.4f}",
                 "dev rows": f"{m['n_dev']:,}", "holdout": f"{m['holdout_start'][:10]} → {m['holdout_end'][:10]}"}
        if result.holdout is not None:
            chips[f"holdout {pm}"] = f"{result.holdout.overall[pm]:+.4f}"
        display_banner(f"Post-analysis — {result.exp_name}", chips,
                       subtitle="walk-forward results, diagnostics and the untouched holdout")
        for i, (title, rep) in enumerate(self.sections.items(), 1):
            display_title("##", title, subtitle=getattr(rep, "subtitle", None), number=f"{i}.")
            rep.display(result, **context)
