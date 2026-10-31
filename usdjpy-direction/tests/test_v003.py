"""v003: cross-asset basket alignment, prefixed multi-asset features, ml-pipe metric set, diagnostics."""

import json

import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_bars
from usdjpy import experiments, train
from usdjpy.data import BASKET, align_basket, basket_path, cache_path, make_forward_return_target
from usdjpy.evaluation import (METRICS, RANK_WEIGHTS, ReturnEvaluator, compute_return_metrics, leaderboard,
                               rank_score)
from usdjpy.exp import v003_cross_asset as v003
from usdjpy.features import (CALENDAR_FEATURES, BarCleaner, ColumnSubset, MultiWindowFeatureEngineer, asset_of,
                             feature_type_of)
from usdjpy.models import make_candidate
from usdjpy.pipeline import prepare

SMALL = dict(windows=(2, 6, 24), shifts=(0, 3), calendar=False)
SYN_ASSETS = ("AAA", "BBB", "CCC")


def _synthetic_basket(n_days=220):
    """Three assets: an FX-like 24/5 series, a weekday-business-hours 'equity' at :30, a sparse one."""
    fx = make_bars(n_days=n_days, seed=11)
    eq = make_bars(n_days=n_days, seed=12)
    eq = eq[(eq.index.hour >= 14) & (eq.index.hour < 21)]
    eq.index = eq.index + pd.Timedelta(minutes=30)           # exchange bars print at :30
    sparse = make_bars(n_days=n_days, seed=13).iloc[::40]
    return {"AAA": fx, "BBB": eq, "CCC": sparse}


@pytest.fixture(scope="module")
def long_bars():
    return make_bars(n_days=220, seed=3)


@pytest.fixture(scope="module")
def basket():
    return _synthetic_basket()


@pytest.fixture
def v003_cache(tmp_path, monkeypatch, long_bars, basket):
    monkeypatch.setenv("USDJPY_DATA_DIR", str(tmp_path / "dataset"))
    cache_path().parent.mkdir(parents=True)
    long_bars.to_csv(cache_path(), index_label="timestamp")
    basket_path("AAA").parent.mkdir(parents=True)
    for name, bars in basket.items():
        bars.to_csv(basket_path(name), index_label="timestamp")
    monkeypatch.setattr(v003, "ASSETS", SYN_ASSETS)
    monkeypatch.setattr(v003, "FX_ASSETS", ("AAA",))
    return tmp_path


# --------------------------------------------------------------------------- alignment
def test_align_basket_asof_and_stale(long_bars, basket):
    grid = BarCleaner(regular_grid=True).fit_transform(long_bars).index
    a = align_basket(basket, grid)
    assert "close_AAA" in a and "close_BBB" in a and "close_CCC" not in a   # sparse one dropped
    # the :30 equity prints are picked up as-of the next grid hour, not missed
    t = grid[(grid.hour == 15) & (grid.day_of_week == 2)][5]
    expected = basket["BBB"]["close"].loc[:t].iloc[-1]
    assert a.loc[t, "close_BBB"] == expected
    assert a.loc[t, "stale_BBB"] == 0.5
    # forward-fill through the equity close (last print 20:30): stale grows by one per hour
    nxt = grid[grid.get_loc(t) + 8]                                 # 23:00
    assert a.loc[nxt, "stale_BBB"] == pytest.approx(2.5)
    assert a.loc[nxt, "close_BBB"] == a.loc[t + pd.Timedelta(hours=6), "close_BBB"]
    assert (a["stale_AAA"].dropna() <= 1.0).sum() > 0.9 * a["stale_AAA"].notna().sum()


# --------------------------------------------------------------------------- features
def test_prefixed_columns_and_helpers(long_bars, basket):
    grid_bars = BarCleaner(regular_grid=True).fit_transform(long_bars)
    X = grid_bars.join(align_basket(basket, grid_bars.index))
    fe = MultiWindowFeatureEngineer(assets=SYN_ASSETS, **SMALL)
    f = fe.fit_transform(X)
    own = [c for c in f.columns if "__" not in c]
    aaa = [c for c in f.columns if c.startswith("AAA__")]
    assert len(aaa) == len(own) + 1 and "AAA__stale" in f          # same set + stale
    assert not any(c.startswith("CCC__") for c in f.columns)         # dropped asset is skipped
    assert asset_of("AAA__ma_div_w6_s0") == "AAA" and asset_of("ma_div_w6_s0") == "USDJPY"
    assert feature_type_of("AAA__ma_div_w6_s3") == "ma_div" and feature_type_of("AAA__stale") == "stale"
    assert feature_type_of("hour_sin") == "calendar"
    sub = ColumnSubset(assets=("USDJPY",))(f)
    assert sub == own
    assert all(c.startswith("BBB__") for c in ColumnSubset(exclude_assets=("USDJPY", "AAA"))(f))


def test_multiasset_no_lookahead(long_bars, basket):
    grid_bars = BarCleaner(regular_grid=True).fit_transform(long_bars)
    X = grid_bars.join(align_basket(basket, grid_bars.index))
    fe = lambda: MultiWindowFeatureEngineer(assets=SYN_ASSETS, **SMALL)  # noqa: E731
    full = fe().fit_transform(X)
    pd.testing.assert_frame_equal(fe().fit_transform(X.iloc[:900]), full.iloc[:900], check_freq=False)
    shocked = X.copy()
    shocked.loc[shocked.index[700:], "close_AAA"] *= 1.5        # shock one asset's future
    out = fe().fit_transform(shocked)
    pd.testing.assert_frame_equal(out.iloc[:700], full.iloc[:700], check_freq=False)
    own = [c for c in full.columns if "__" not in c]
    pd.testing.assert_frame_equal(out[own], full[own], check_freq=False)   # USDJPY columns untouched


# --------------------------------------------------------------------------- metrics
def test_return_metric_set_hand_case():
    y = np.array([0.01, -0.02, 0.03, -0.01, 0.0])
    m = compute_return_metrics(y, y * 2)
    for k in ("spearman", "pearson", "kendall", "weighted_tau", "r2", "mse", "mae", "wasserstein", "mcc", "f1",
              "precision", "recall", "roc_auc", "bin_acc", "sharpe_ratio"):
        assert k in m and k in METRICS
    assert m["spearman"] == pytest.approx(1.0) and m["kendall"] == pytest.approx(1.0)
    assert m["weighted_tau"] == pytest.approx(1.0)
    assert m["bin_acc"] == 1.0 and m["f1"] == 1.0 and m["precision"] == 1.0 and m["recall"] == 1.0
    assert m["mae"] == pytest.approx(np.abs(y).mean()) and m["wasserstein"] > 0
    assert m["sharpe_ratio"] > 0 and m["hit_rate"] == m["bin_acc"] and m["auc"] == m["roc_auc"]
    flat = compute_return_metrics(y, np.zeros(5))
    assert flat["weighted_tau"] == 0.0 and flat["sharpe_ratio"] == -1.0


def test_rank_score_prefers_better_models():
    board = pd.DataFrame({"spearman": [0.05, 0.0, -0.05], "pearson": [0.04, 0.0, -0.04],
                          "mse": [1.0, 2.0, 3.0], "r2": [0.1, 0.0, -0.1], "mcc": [0.1, 0.0, -0.1],
                          "f1": [0.6, 0.5, 0.4], "roc_auc": [0.6, 0.5, 0.4]}, index=["good", "mid", "bad"])
    r = rank_score(board)
    assert r["good"] > r["mid"] > r["bad"]
    assert r["good"] == pytest.approx(3 * sum(w for _, w in RANK_WEIGHTS))


def test_leaderboard_has_rank_score(long_bars):
    data = prepare(long_bars, BarCleaner(regular_grid=True), MultiWindowFeatureEngineer(**SMALL), 0.15,
                   target_builder=lambda b: make_forward_return_target(b, 1, 1), purge=2)
    ev = ReturnEvaluator(cv=2, gap=30, horizon=1)
    cs = ColumnSubset()
    board, reports = ev.compare({"random": make_candidate("random", columns=cs),
                                 "persistence": make_candidate("persistence", columns=cs)
                                 .set_params(model__column="pct_change_w24_s0")}, data.X, data.y, data.ret)
    assert "rank_score" in board and board["rank_score"].notna().all()
    assert "regime" in reports["random"].slices and reports["random"].slice_table("regime")["n"].sum() == len(reports["random"].y_true)


# --------------------------------------------------------------------------- the v003 arm
def test_v003_factories():
    assert v003.EXP_NAME == "v003_cross_asset" and v003.TASK == "return"
    assert set(v003.ASSETS) == set(BASKET) and len(v003.make_experiments(0, 3)) == 4
    pipe = v003.make_candidates(0)["lgbm_reg"]
    assert pipe.get_params()["model__regressor__colsample_bytree"] == v003.COLSAMPLE


def test_v003_run_and_diagnostics(v003_cache, tmp_path, capsys):
    res = v003.run(cv=2, models="random,lgbm_reg", output_dir=tmp_path / "out", experiments=True, figures=False,
                   display_preanalysis=True, display_postanalysis=True)
    meta = json.loads((tmp_path / "out" / "run_meta.json").read_text())
    assert meta["basket_used"] == ["AAA", "BBB"] and meta["n_features"] > 3 * 100
    assert any(c.startswith("AAA__") for c in res.data.X.columns)
    gen = res.diagnostics["generalization"]
    assert "spearman" in gen and "in_sample_spearman" in gen and "lgbm_reg" in gen.index
    assert {"lgbm_reg (USDJPY only = v002)", "lgbm_reg (FX assets only)"} <= set(res.experiments)
    assert not any("__" in c for c in res.experiments["lgbm_reg (USDJPY only = v002)"].estimator_ or [])
    out = capsys.readouterr().out
    assert "by asset" in out and "generalization gap" in out and "Residuals" in out and "Jarque-Bera" in out


def test_v003_cli_predict_and_ladder(v003_cache, tmp_path, capsys):
    out_dir = tmp_path / "cli"
    assert train.main(["--exp", "v003_cross_asset", "--models", "persistence,lgbm_reg", "--cv", "2", "--output-dir", str(out_dir),
                       "--no-experiments", "--no-figures", "--no-diagnostics"]) == 0
    assert "rank_score" in capsys.readouterr().out
    assert train.main(["--predict-only", "--output-dir", str(out_dir)]) == 0
    assert "predicted log return" in capsys.readouterr().out
    ladder = tmp_path / "experiments.json"
    assert experiments.main(["--cv", "2", "--only", "USDJPY only,USDJPY + FX", "--out", str(ladder)]) == 0
    res = json.loads(ladder.read_text())
    assert res["arm"] == "v003" and len(res["results"]) == 2
