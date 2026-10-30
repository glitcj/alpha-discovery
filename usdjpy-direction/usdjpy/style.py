"""One place for colours and matplotlib chrome.

Colour is assigned by the *job* it does (dataviz skill): categorical slots in a fixed order for
identity (models, asset groups), one blue ramp for magnitude, blue <-> red through a neutral grey for
sign / polarity, and a reserved status palette for verdict boxes only. Figures are built with the
helpers here (``new_figure`` / ``finish`` / ``date_axis`` / ``hbar`` / ``heatmap`` / ``shade``) so every
reporter and display looks the same without repeating rcParams.
"""

from __future__ import annotations

import contextlib
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- palette
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
SEQUENTIAL = ("#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
              "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b")
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
       "axis": "#c3c2b7", "surface": "#fcfcfb", "page": "#f9f9f7", "neutral": "#f0efec", "grey": "#b5b3ac"}
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
POS, NEG = SERIES[0], SERIES[7]                                   # diverging poles (blue up, red down)
BENCHMARK = INK["grey"]                                           # buy & hold, baselines, reference lines

# asset groups keep one slot each across every figure (colour follows the entity)
GROUP_COLORS = {"target": SERIES[0], "fx": SERIES[1], "rates": SERIES[2], "equity": SERIES[3],
                "commodity": SERIES[4], "crypto": SERIES[6], "other": INK["grey"]}
# the 6 joint regimes: up-trend = blues, down-trend = reds, darker = more volatile
REGIME_COLORS = {
    "down-trend / low vol": "#f4b5b5", "down-trend / mid vol": "#e87878", "down-trend / high vol": "#b32d2d",
    "up-trend / low vol": "#b7d3f6", "up-trend / mid vol": "#5598e7", "up-trend / high vol": "#1c5cab",
    "unknown": INK["neutral"],
}
BASELINE_NAMES = {"persistence", "mean", "best_feature", "random", "majority", "momentum", "buy & hold"}


_SLOTS: dict[str, str] = {}


def color_for(name: str) -> str:
    """Categorical colour for a model / experiment name: baselines grey, every other model gets the next
    free slot the first time it is seen and keeps it (colour follows the entity, never its rank)."""
    base = name.split(" (")[0].strip()
    if base in BASELINE_NAMES:
        return BENCHMARK
    if base not in _SLOTS:
        _SLOTS[base] = SERIES[len(_SLOTS) % len(SERIES)]
    return _SLOTS[base]


def sign_colors(values) -> np.ndarray:
    v = np.asarray(values, float)
    return np.where(v >= 0, POS, NEG)


def cmap_sequential():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("usdjpy_seq", list(SEQUENTIAL))


def cmap_diverging():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("usdjpy_div", [NEG, "#f4b5b5", INK["neutral"], "#9ec5f4", POS])


# --------------------------------------------------------------------------- chrome
RC = {
    "figure.facecolor": INK["surface"], "figure.dpi": 110, "figure.titlesize": 11.5,
    "figure.titleweight": "bold",
    "axes.facecolor": INK["surface"], "axes.edgecolor": INK["axis"], "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "axes.grid": True, "grid.color": INK["grid"], "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlecolor": INK["primary"], "axes.titlepad": 8,
    "axes.labelsize": 8.5, "axes.labelcolor": INK["secondary"],
    "xtick.labelsize": 8, "ytick.labelsize": 8, "xtick.color": INK["muted"], "ytick.color": INK["muted"],
    "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 4, "ytick.major.pad": 4,
    "axes.prop_cycle": None,                    # filled in by use(); needs matplotlib
    "lines.linewidth": 1.6, "lines.markersize": 5,
    "legend.frameon": False, "legend.fontsize": 8, "legend.labelcolor": INK["secondary"],
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9, "text.color": INK["primary"],
    "savefig.facecolor": INK["surface"], "savefig.bbox": "tight", "savefig.pad_inches": 0.15,
}


def rc_params() -> dict:
    from cycler import cycler

    rc = dict(RC)
    rc["axes.prop_cycle"] = cycler(color=list(SERIES))
    return rc


def use() -> None:
    """Apply the chrome globally (notebook cells)."""
    import matplotlib

    matplotlib.rcParams.update(rc_params())


@contextlib.contextmanager
def plot_style():
    """Apply the chrome for the figures built inside the block only."""
    import matplotlib

    with matplotlib.rc_context(rc_params()):
        yield


# --------------------------------------------------------------------------- figure helpers
def new_figure(nrows: int = 1, ncols: int = 1, *, title: str | None = None, subtitle: str | None = None,
               width: float = 11.0, height: float | None = None, sharex=False, sharey=False, **kw):
    """Figure + axes array (always 2-D) with a left-aligned title and a muted subtitle."""
    import matplotlib.pyplot as plt

    height = height or (3.0 * nrows + (0.5 if title else 0))
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), sharex=sharex, sharey=sharey, squeeze=False,
                             **kw)
    fig._usdjpy_title = title
    fig._usdjpy_subtitle = subtitle
    return fig, axes


def finish(fig, *, title: str | None = None, subtitle: str | None = None, top: float | None = None):
    """Tight layout, then title (semibold, left) and subtitle (muted) above the axes."""
    title = title or getattr(fig, "_usdjpy_title", None)
    subtitle = subtitle or getattr(fig, "_usdjpy_subtitle", None)
    h = fig.get_figheight()
    inches = (0.32 if title else 0.0) + (0.22 if subtitle else 0.0)      # header band height
    pad = inches / h
    fig.tight_layout(rect=(0, 0, 1, 1 - pad) if pad else None)
    if title:
        fig.text(0.01, 0.995, title, ha="left", va="top", fontsize=11.5, fontweight="bold", color=INK["primary"])
    if subtitle:
        fig.text(0.01, 0.995 - (0.3 if title else 0.0) / h, subtitle, ha="left", va="top", fontsize=8.5,
                 color=INK["muted"])
    return fig


def date_axis(ax, *, rotate: bool = False):
    """Concise date ticks (no rotated labels)."""
    import matplotlib.dates as mdates

    loc = mdates.AutoDateLocator(minticks=4, maxticks=9)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    if rotate:
        ax.tick_params(axis="x", labelrotation=30)
    return ax


def zero_line(ax, y: float = 0.0, **kw):
    ax.axhline(y, color=INK["axis"], lw=0.8, zorder=1, **kw)


def band(ax, lo: float, hi: float, color: str | None = None, alpha: float = 0.5, label: str | None = None):
    ax.axhspan(lo, hi, color=color or INK["neutral"], alpha=alpha, lw=0, zorder=0, label=label)


def shade(ax, index: pd.Index, labels: pd.Series | np.ndarray, colors: dict, alpha: float = 0.35,
          y0: float = 0.0, y1: float = 1.0):
    """Contiguous spells of ``labels`` as coloured bands between axes fractions ``y0``..``y1``."""
    lab = pd.Series(np.asarray(labels).astype(str), index=index)
    change = (lab != lab.shift()).to_numpy()
    starts = np.flatnonzero(change)
    ends = np.append(starts[1:], len(lab))
    for s, e in zip(starts, ends):
        c = colors.get(lab.iloc[s])
        if c is None:
            continue
        x1 = index[e] if e < len(index) else index[-1]
        ax.axvspan(index[s], x1, ymin=y0, ymax=y1, color=c, alpha=alpha, lw=0, zorder=0)


def vlines(ax, xs: Iterable, color: str | None = None, **kw):
    for x in xs:
        ax.axvline(x, color=color or INK["axis"], lw=0.8, zorder=1, **kw)


def hbar(ax, values: pd.Series, colors=None, *, labels: bool = True, fmt: str = "{:.3f}", gap: float = 0.18):
    """Horizontal bars (top of the series at the top), thin, with a surface gap and value labels."""
    s = values.iloc[::-1]
    cols = colors if colors is not None else SERIES[0]
    if isinstance(cols, (list, tuple, np.ndarray, pd.Series)):
        cols = list(np.asarray(cols)[::-1])
    bars = ax.barh(np.arange(len(s)), s.to_numpy(), color=cols, height=1 - gap, zorder=2)
    ax.set_yticks(np.arange(len(s)), [str(i) for i in s.index])
    ax.grid(axis="y", visible=False)
    if labels:
        vals = s.to_numpy()
        span = float(np.nanmax(np.abs(vals))) if len(s) else 1.0
        for i, v in enumerate(vals):
            if np.isnan(v):
                continue
            # positive: just past the bar end; negative: just right of the zero line (keeps tick labels clear)
            ax.text(max(v, 0) + 0.015 * span, i, fmt.format(v), va="center", ha="left", fontsize=7,
                    color=INK["secondary"])
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo, max(hi, float(np.nanmax(np.r_[vals, 0])) + 0.22 * span))
    return bars


def vbar(ax, values: pd.Series, colors=None, *, labels=None, gap: float = 0.18, fmt: str = "{:.2f}"):
    """Vertical bars over a categorical index with optional small text per bar (e.g. ``n=...``)."""
    x = np.arange(len(values))
    cols = colors if colors is not None else SERIES[0]
    bars = ax.bar(x, values.to_numpy(), color=cols, width=1 - gap, zorder=2)
    ax.set_xticks(x, [str(i) for i in values.index])
    ax.grid(axis="x", visible=False)
    if labels is not None:
        ymin, ymax = ax.get_ylim()
        for i, (v, lab) in enumerate(zip(values.to_numpy(), labels)):
            ax.text(i, v + 0.01 * (ymax - ymin) * (1 if v >= 0 else -1), str(lab), ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=7, color=INK["muted"])
    return bars


def heatmap(ax, frame: pd.DataFrame, *, cmap=None, fmt: str = "{:.2f}", vmin=None, vmax=None, center=None,
            cbar: bool = True, label: str | None = None, annotate: bool = True, fontsize: float = 7):
    """Imshow heatmap with in-cell labels in ink (white on dark cells)."""
    data = frame.to_numpy(dtype=float)
    cmap = cmap or cmap_sequential()
    if center is not None:
        lim = float(np.nanmax(np.abs(data - center))) if np.isfinite(data).any() else 1.0
        vmin, vmax = center - lim, center + lim
    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(frame.shape[1]), [str(c) for c in frame.columns])
    ax.set_yticks(np.arange(frame.shape[0]), [str(i) for i in frame.index])
    ax.grid(visible=False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    if annotate and data.size <= 600 and data.shape[1] <= 16:
        norm = im.norm
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data[i, j]
                if np.isnan(v):
                    continue
                r, g, b, _ = cmap(norm(v))
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=fontsize,
                        color="#ffffff" if lum < 0.5 else INK["primary"])
    if cbar:
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=7, color=INK["muted"], labelcolor=INK["muted"], size=0)
        if label:
            cb.set_label(label, fontsize=7.5, color=INK["secondary"])
    return im


def annotate_end(ax, x, y, text: str, color: str, dx: float = 4):
    ax.annotate(text, (x, y), xytext=(dx, 0), textcoords="offset points", va="center", fontsize=7.5, color=color)


def dot_range(ax, point: pd.Series, lo: pd.Series | None, hi: pd.Series | None, colors=None, *,
              fmt: str = "{:+.3f}", highlight: str | None = None):
    """A dot per row with a thin lo..hi whisker (fold range), best at the top."""
    s = point.iloc[::-1]
    y = np.arange(len(s))
    cols = list(np.asarray(colors)[::-1]) if colors is not None else [SERIES[0]] * len(s)
    if lo is not None and hi is not None:
        ax.hlines(y, lo.iloc[::-1].to_numpy(), hi.iloc[::-1].to_numpy(), color=cols, lw=1.4, alpha=0.6, zorder=2)
    ax.scatter(s.to_numpy(), y, color=cols, s=36, zorder=3, edgecolor=INK["surface"], linewidth=1)
    ax.set_yticks(y, [str(i) for i in s.index])
    for lab in ax.get_yticklabels():
        if highlight and lab.get_text() == highlight:
            lab.set_fontweight("bold")
            lab.set_color(INK["primary"])
    ax.grid(axis="y", visible=False)
    span = float(np.nanmax(np.abs(np.r_[s.to_numpy(), (hi.to_numpy() if hi is not None else [])]))) or 1.0
    for yi, v in zip(y, s.to_numpy()):
        if not np.isnan(v):
            x = (hi.iloc[::-1].to_numpy()[yi] if hi is not None else v)
            x = np.nanmax([x, v, 0.0])                   # never left of the zero / chance line
            ax.text(x + 0.03 * span, yi, fmt.format(v), va="center", fontsize=7, color=INK["secondary"])
    ax.set_xlim(right=ax.get_xlim()[1] + 0.2 * span)


def legend_patches(ax, colors: dict, title: str | None = None, order: Sequence[str] | None = None, **kw):
    from matplotlib.patches import Patch

    keys = list(order) if order else list(colors)
    handles = [Patch(facecolor=colors[k], label=k) for k in keys if k in colors]
    return ax.legend(handles=handles, title=title, **kw)
