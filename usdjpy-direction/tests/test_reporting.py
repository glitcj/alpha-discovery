"""Style module, regime helpers, display primitives and the figure set written by the reporters."""

import matplotlib
import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_bars
from tests.test_pipeline import _run
from tests.test_v003 import basket, long_bars, v003_cache  # noqa: F401  (fixtures)
from usdjpy import style
from usdjpy.analysis import display
from usdjpy.evaluation import REGIME_LABELS, DirectionSlicer
from usdjpy.exp import v003_cross_asset as v003
from usdjpy.features import BarCleaner, MultiWindowFeatureEngineer
from usdjpy.regimes import DECILE_LABELS, decile_regimes, fold_spans, joint_regimes, regime_stats

PRE_FIGURES = ("pre_coverage", "pre_market", "pre_target", "pre_acf", "pre_regimes_timeline", "pre_regimes_dist",
               "pre_regimes_deciles", "pre_seasonality", "pre_feature_rho")
POST_FIGURES = ("post_leaderboard", "post_training", "post_experiments", "post_timeline", "post_calendar",
                "post_slices", "post_regime_perf", "post_importance", "post_holdout_timeline")
RETURN_ONLY = ("pre_target_acf", "post_regime_deciles", "post_pred_vs_real", "post_residuals")
BASKET_ONLY = ("pre_crossasset_corr", "pre_crossasset_rolling", "pre_basket_freshness")


# --------------------------------------------------------------------------- style
def test_palette_and_helpers():
    assert style.SERIES[0] == "#2a78d6" and style.SERIES[7] == "#e34948" and len(style.SERIES) == 8
    assert style.color_for("lgbm_reg") == style.color_for("lgbm_reg (FX assets only)")      # colour follows the entity
    assert style.color_for("persistence") == style.BENCHMARK == style.color_for("random")
    assert list(style.sign_colors([1.0, -0.5, 0.0])) == [style.POS, style.NEG, style.POS]
    mid = style.cmap_diverging()(0.5)
    assert np.allclose(mid[:3], matplotlib.colors.to_rgb(style.INK["neutral"]), atol=0.02)
    assert set(style.REGIME_COLORS) >= set(REGIME_LABELS)
    before = matplotlib.rcParams["axes.spines.top"]
    with style.plot_style():
        assert matplotlib.rcParams["axes.spines.top"] is False
        assert matplotlib.rcParams["figure.facecolor"] == style.INK["surface"]
        fig, axes = style.new_figure(2, 2, title="t", subtitle="s")
        assert axes.shape == (2, 2)
        style.hbar(axes[0, 0], pd.Series([0.3, -0.1], index=["a", "b"]))
        style.heatmap(axes[0, 1], pd.DataFrame(np.arange(6.0).reshape(2, 3)), cmap=style.cmap_diverging(), center=0)
        style.dot_range(axes[1, 0], pd.Series([0.1, 0.0], index=["m", "n"]), pd.Series([0.0, -0.1], index=["m", "n"]),
                        pd.Series([0.2, 0.1], index=["m", "n"]), highlight="m")
        idx = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
        style.shade(axes[1, 1], idx, ["a"] * 5 + ["b"] * 5, {"a": "#ff0000", "b": "#0000ff"})
        style.finish(fig)
        matplotlib.pyplot.close(fig)
    assert matplotlib.rcParams["axes.spines.top"] == before


# --------------------------------------------------------------------------- regimes
def test_joint_regimes_match_slicer_and_have_balanced_terciles():
    bars = make_bars(n_days=260, seed=5)
    grid = BarCleaner(regular_grid=True).fit_transform(bars)
    reg = joint_regimes(grid, 504)
    known = reg[reg != "unknown"]
    assert set(known.unique()) <= set(REGIME_LABELS) and len(known) > 0.8 * len(reg)
    vol = known.str.split(" / ").str[1].value_counts(normalize=True)
    assert (vol - 1 / 3).abs().max() < 0.05
    # the slice in evaluation.py derives the same labels from the v002 feature columns
    X = MultiWindowFeatureEngineer(types=("pct_change", "change_volatility"), windows=(504,), shifts=(0,),
                                   calendar=True).fit_transform(grid)
    from_x = DirectionSlicer(("regime",)).transform(X.dropna())["regime"]
    both = pd.concat([reg.rename("bars"), from_x.rename("x")], axis=1).dropna()
    both = both[(both["bars"] != "unknown") & (both["x"] != "unknown")]
    assert (both["bars"] == both["x"]).mean() > 0.9           # qcut edges differ slightly (slicer drops warm-up rows)
    stats = regime_stats(reg, np.log(grid["close"]).diff().shift(-1))
    assert stats["share"].sum() == pytest.approx(1.0) and list(stats.index) == [r for r in REGIME_LABELS if r in stats.index]
    assert {"n", "share", "mean_bp", "std_bp", "up_share", "mean_spell_bars"} <= set(stats.columns)


def test_decile_regimes_past_is_causal_and_future_is_a_shift():
    bars = make_bars(n_days=200, seed=6)
    past = decile_regimes(bars, 100, 1, kind="trend", when="past")
    assert set(past.unique()) <= set(DECILE_LABELS) | {"unknown"}
    truncated = decile_regimes(bars.iloc[:1500], 100, 1, kind="trend", when="past")
    # deciles are sample-wide quantiles, so labels can move by one bucket; the ordering must not
    a = past.iloc[:1500].map(lambda s: DECILE_LABELS.index(s) if s in DECILE_LABELS else np.nan)
    b = truncated.map(lambda s: DECILE_LABELS.index(s) if s in DECILE_LABELS else np.nan)
    assert pd.concat([a, b], axis=1).dropna().corr(method="spearman").iloc[0, 1] > 0.95
    fut = decile_regimes(bars, 100, 1, kind="vol", when="future")
    pst = decile_regimes(bars, 100, 1, kind="vol", when="past")
    assert (fut.iloc[:-101].to_numpy() == pst.iloc[101:].to_numpy()).mean() > 0.99
    with pytest.raises(ValueError):
        decile_regimes(bars, 10, 1, kind="nope")
    assert fold_spans(None) == []
    spans = fold_spans(pd.DataFrame({"test_start": ["2024-01-01"], "test_end": ["2024-02-01"]}))
    assert spans == [(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01"))]


# --------------------------------------------------------------------------- display primitives
def test_display_text_mode_and_figure_saving(tmp_path, capsys, monkeypatch):
    display.display_title("##", "Residuals", subtitle="what is left", number="3.")
    display.display_banner("Pre-analysis — x", {"task": "return"}, subtitle="dev rows")
    display.display_caption("look here")
    display.display_status("fine", "good", title="Verdict")
    display.display_info_grid({"a": 1}, title="Information")
    out = capsys.readouterr().out
    assert "## 3. Residuals — what is left" in out and "# Pre-analysis — x" in out and "task: return" in out
    assert "_look here_" in out and "[good] ✔ Verdict: fine" in out and "- a: 1" in out
    import matplotlib.pyplot as plt

    fig = plt.figure()
    assert display.display_figure(fig, name="nope") is None                  # FIGURE_DIR unset -> not saved
    monkeypatch.setattr(display, "FIGURE_DIR", tmp_path / "figs")
    fig = plt.figure()
    path = display.display_figure(fig, name="yes", caption="cap")
    assert path == tmp_path / "figs" / "yes.png" and path.exists()
    assert "[figure] yes.png" in capsys.readouterr().out


# --------------------------------------------------------------------------- reporters end to end
def test_direction_run_writes_all_figures(bars, tmp_path, capsys):
    res = _run(bars, tmp_path, figures=True, display_preanalysis=True, display_postanalysis=True)
    figs = {p.stem for p in (tmp_path / "figures").glob("*.png")}
    expected = set(POST_FIGURES) - {"post_experiments"} - ({"post_importance"} if res.best_name != "lgbm" else set())
    assert set(PRE_FIGURES) <= figs and expected <= figs
    assert not (set(RETURN_ONLY) & figs) and not (set(BASKET_ONLY) & figs)
    assert {"lgbm", "majority", "momentum", f"holdout_{res.best_name}"} <= figs   # per-model 2x2 panels kept
    out = capsys.readouterr().out
    assert "## 1. Coverage" in out and "## 2. Market" in out and "## 5. Regimes" in out
    assert "no basket columns" in out and "Verdict:" in out and "residual diagnostics apply to the return task" in out
    assert display.FIGURE_DIR is None and res.holdout is not None


def test_v003_run_writes_return_and_basket_figures(v003_cache, tmp_path, capsys):  # noqa: F811
    res = v003.run(cv=2, models="random,lgbm_reg", output_dir=tmp_path / "out", experiments=True, figures=True,
                   display_preanalysis=True, display_postanalysis=True)
    figs = {p.stem for p in (tmp_path / "out" / "figures").glob("*.png")}
    assert set(PRE_FIGURES) | set(POST_FIGURES) | set(RETURN_ONLY) | set(BASKET_ONLY) <= figs
    out = capsys.readouterr().out
    assert "## 6. Cross-asset" in out and "## 7. Regime performance" in out and "## 11. Verdict" in out
    assert "Jarque-Bera" in out and "by joint regime" in out and "[figure] post_calendar.png" in out
    assert res.best.task == "return"


def test_figures_cleared_once_per_run(bars, tmp_path):
    (tmp_path / "figures").mkdir()
    stale = tmp_path / "figures" / "old.png"
    stale.write_bytes(b"x")
    _run(bars, tmp_path, figures=True, display_preanalysis=True, display_postanalysis=False, diagnostics=False)
    assert not stale.exists() and (tmp_path / "figures" / "pre_market.png").exists()
    _run(bars, tmp_path, figures=False, diagnostics=False)
    assert (tmp_path / "figures" / "pre_market.png").exists()                # figures=False leaves the folder alone
