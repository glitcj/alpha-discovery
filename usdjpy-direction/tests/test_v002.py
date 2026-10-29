"""v002: FX-week grid, ml-pipe forward log-return target, multi-window features, return evaluation."""

import json

import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_bars
from usdjpy import experiments, train
from usdjpy.data import cache_path, make_forward_return_target, split_holdout
from usdjpy.evaluation import ReturnEvaluator, compute_return_metrics, daily_strategy, plot_report
from usdjpy.exp import v002_fwd_logret_1d as v002
from usdjpy.features import (CALENDAR_FEATURES, BarCleaner, ColumnSubset, MultiWindowFeatureEngineer,
                             _weighted_window, fx_week_mask, to_fx_week_grid)
from usdjpy.models import REGRESSION_CANDIDATES, make_candidate
from usdjpy.pipeline import prepare

SMALL = dict(windows=(2, 6, 24), shifts=(0, 3), calendar=False)


@pytest.fixture(scope="module")
def long_bars():
    return make_bars(n_days=220, seed=3)


@pytest.fixture
def long_cache(tmp_path, monkeypatch, long_bars):
    monkeypatch.setenv("USDJPY_DATA_DIR", str(tmp_path / "dataset"))
    path = cache_path()
    path.parent.mkdir(parents=True)
    long_bars.to_csv(path, index_label="timestamp")
    return path


# --------------------------------------------------------------------------- target + grid
def _closes(values, start="2024-01-02"):
    idx = pd.date_range(start, periods=len(values), freq="h", tz="UTC")
    c = np.asarray(values, float)
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c}, index=idx)


def test_forward_return_target_matches_ml_pipe_definition():
    bars = _closes(np.arange(100, 110))
    t = make_forward_return_target(bars, horizon=3, offset=1)
    # y_t = log(close[t+1+3] / close[t+1]); the last 4 rows have no label
    assert len(t) == 10 - 4
    assert np.isclose(t["y"].iloc[0], np.log(104 / 101))
    assert np.isclose(t["y"].iloc[5], np.log(109 / 106))
    assert (t["y"] == t["next_logret"]).all()
    with pytest.raises(ValueError):
        make_forward_return_target(bars, horizon=0)


def test_fx_week_mask_new_york_time():
    summer = pd.DatetimeIndex(["2024-07-05 20:00", "2024-07-05 21:00", "2024-07-06 12:00",
                               "2024-07-07 20:00", "2024-07-07 21:00", "2024-07-08 03:00"], tz="UTC")
    # Fri 16:00 EDT in, Fri 17:00 EDT out, Saturday out, Sun 16:00 EDT out, Sun 17:00 EDT in, Monday in
    assert fx_week_mask(summer).tolist() == [True, False, False, False, True, True]
    winter = pd.DatetimeIndex(["2024-01-05 21:00", "2024-01-05 22:00", "2024-01-07 22:00"], tz="UTC")
    assert fx_week_mask(winter).tolist() == [True, False, True]   # EST: 17:00 NY = 22:00 UTC


def test_grid_fills_missing_hours_and_drops_weekend():
    bars = _closes([100, 101, 102, 103, 104, 105], start="2024-01-03 00:00")
    bars = bars.drop(bars.index[2])                              # a missing Wednesday hour
    sat = pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0]},
                       index=pd.DatetimeIndex(["2024-01-06 12:00"], tz="UTC"))
    g = to_fx_week_grid(pd.concat([bars, sat]))
    assert g.loc["2024-01-03 02:00", "close"].item() == 101       # forward-filled
    assert not (g.index.day_of_week == 5).any()                  # Saturday removed
    assert fx_week_mask(g.index).all()
    assert (g.index.to_series().diff().dropna() >= pd.Timedelta(hours=1)).all()


def test_cleaner_regular_grid(long_bars):
    g = BarCleaner(regular_grid=True).fit_transform(long_bars)
    step = g.index.to_series().diff().dropna()
    assert fx_week_mask(g.index).all()
    assert set(step[step > pd.Timedelta(hours=1)].dt.days.unique()) <= {1, 2}   # only weekends


def test_holdout_purge():
    idx = pd.date_range("2024-01-01", periods=100, freq="h", tz="UTC")
    dev, hold = split_holdout(idx, 0.2, purge=5)
    assert len(hold) == 20 and len(dev) == 75
    assert idx.get_loc(hold[0]) - idx.get_loc(dev[-1]) == 6


# --------------------------------------------------------------------------- features
def test_multiwindow_columns():
    fe = MultiWindowFeatureEngineer()
    cols = fe.fit(pd.DataFrame({"close": [1.0]}, index=pd.DatetimeIndex(["2024-01-01"], tz="UTC")))\
        .get_feature_names_out()
    n_base = 9 + 6 * 18          # window 2 supports 9 types, the 6 longer windows all 18
    assert len(cols) == 2 * n_base + len(CALENDAR_FEATURES)
    assert "ma_div_w24_s0" in cols and "ma_div_w24_s23" in cols and "zscore_w2_s0" not in cols


def test_multiwindow_no_lookahead(long_bars):
    fe = MultiWindowFeatureEngineer(**SMALL)
    full = fe.fit_transform(long_bars)
    for k in (300, 1000):
        part = MultiWindowFeatureEngineer(**SMALL).fit_transform(long_bars.iloc[:k])
        pd.testing.assert_frame_equal(part, full.iloc[:k], check_freq=False)
    shocked = long_bars.copy()
    shocked.iloc[700:] *= 1.3
    pd.testing.assert_frame_equal(MultiWindowFeatureEngineer(**SMALL).fit_transform(shocked).iloc[:700],
                                  full.iloc[:700], check_freq=False)


def test_multiwindow_spot_values():
    bars = _closes([10, 11, 12, 11, 13, 12, 14])
    f = MultiWindowFeatureEngineer(windows=(3,), shifts=(0, 1), calendar=False).fit_transform(bars)
    c = bars["close"]
    i = 6
    ma = c.iloc[i - 2:i + 1].mean()
    assert np.isclose(f["ma_div_w3_s0"].iloc[i], (c.iloc[i] - ma) / ma)
    hi = c.iloc[i - 3:i].max()                                     # shift 1 -> window ending at i-1
    assert np.isclose(f["drawdown_w3_s1"].iloc[i], (hi - c.iloc[i - 1]) / hi)
    d = c.diff().iloc[i - 2:i + 1]
    rs = d.clip(lower=0).mean() / (-d).clip(lower=0).mean()
    assert np.isclose(f["rsi_w3_s0"].iloc[i], 1 - 2 / (1 + rs))
    assert np.isclose(f["pct_change_w3_s0"].iloc[i], c.iloc[i] / c.iloc[i - 3] - 1)


def test_weighted_window_matches_ml_pipe_kernel():
    x = np.random.default_rng(0).normal(size=50)
    w = 7
    alpha = 2 / (1 + w)
    wts = np.array([(1 - alpha) ** k for k in range(w)][::-1])
    i = 30
    win = x[i - w + 1:i + 1]
    assert np.isclose(_weighted_window(x, w, centred=False)[i], win @ wts / wts.sum())
    expect = np.sqrt(((win - win.mean()) ** 2) @ wts / wts.sum() * w / (w - 1))
    assert np.isclose(_weighted_window(x, w, centred=True)[i], expect)


# --------------------------------------------------------------------------- metrics + evaluator
def test_return_metrics_hand_case():
    y = np.array([0.01, -0.02, 0.03, -0.01])
    m = compute_return_metrics(y, y * 0.5, y, cost_bps=0, horizon=1)
    assert m["spearman"] == pytest.approx(1.0) and m["pearson"] == pytest.approx(1.0)
    assert m["hit_rate"] == 1.0 and m["mcc"] == pytest.approx(1.0)
    assert m["strategy_bps"] == pytest.approx(np.abs(y).mean() * 1e4)
    flat = compute_return_metrics(y, np.zeros(4))
    assert np.isnan(flat["spearman"])


def test_daily_strategy_samples_every_horizon_rows():
    rows, net = daily_strategy(np.ones(10), np.full(10, 0.001), horizon=4, cost_bps=1.0)
    assert rows.tolist() == [0, 4, 8]
    np.testing.assert_allclose(net, [0.001 - 1e-4, 0.001, 0.001])


def test_return_evaluator_random_walk(long_bars):
    """1-bar-ahead target on a random walk: no model should find a rank correlation."""
    data = prepare(long_bars, BarCleaner(regular_grid=True), MultiWindowFeatureEngineer(**SMALL), 0.15,
                   target_builder=lambda b: make_forward_return_target(b, horizon=1, offset=1), purge=2)
    ev = ReturnEvaluator(cv=3, gap=30, horizon=1)
    for tr, te in ev._cv().split(data.X):
        assert tr.max() + 30 < te.min()
    pipe = make_candidate("lgbm_reg", columns=ColumnSubset())
    rep = ev.evaluate(pipe, data.X, data.y, data.ret, name="lgbm_reg")
    assert rep.task == "return" and abs(rep.primary) < 0.08
    assert rep.slice_table("confidence")["n"].sum() == len(rep.y_true)
    import matplotlib.pyplot as plt
    plt.close(plot_report(rep))


@pytest.mark.parametrize("name", REGRESSION_CANDIDATES)
def test_regressors_fit_predict(long_bars, name):
    data = prepare(long_bars, v002.make_cleaner(), v002.make_feature_engineer(), 0.15,
                   target_builder=v002.make_target_builder(), purge=25)
    X, y = data.X.iloc[:800], data.y.iloc[:800]
    pred = make_candidate(name).fit(X, y).predict(data.X.iloc[800:900])
    assert pred.shape == (100,) and np.isfinite(pred).all()


# --------------------------------------------------------------------------- the v002 arm
def test_v002_factories():
    assert v002.EXP_NAME == "v002_fwd_logret_1d" and v002.TASK == "return"
    assert v002.GAP >= v002.HORIZON + v002.OFFSET
    assert set(v002.make_candidates(0)) == {"lgbm_reg"}
    assert v002._split_models("persistence,lgbm_reg") == (("persistence",), ("lgbm_reg",))
    assert len(v002.make_experiments(0, 3)) == 3
    with pytest.raises(ValueError):
        v002._split_models("lgbm")


def test_v002_run(long_bars, tmp_path, capsys):
    res = v002.run(bars=long_bars, cv=2, models="persistence,random,lgbm_reg", output_dir=tmp_path,
                   experiments=False, figures=True, display_preanalysis=True, display_postanalysis=True)
    assert res.best.task == "return" and list(res.board.columns)[0] == "spearman"
    d = res.data
    assert d.X.index.max() < d.X_hold.index.min()
    meta = json.loads((tmp_path / "run_meta.json").read_text())
    assert meta["task"] == "return" and meta["target"]["horizon"] == 24 and meta["purge"] == 25
    pred = pd.read_csv(tmp_path / "predictions.csv")
    assert {"y_pred", "pred_sign", "y_true"} <= set(pred.columns)
    assert "spearman" in json.loads((tmp_path / "metrics.json").read_text())["holdout"]
    assert res.diagnostics["lgbm"]["best_metric"] == "l2"
    out = capsys.readouterr().out
    assert "Post-analysis" in out and "Holdout Spearman" in out


def test_v002_cli_and_predict(long_cache, tmp_path, capsys):
    rc = train.main(["--models", "persistence,lgbm_reg", "--cv", "2", "--output-dir", str(tmp_path),
                     "--no-experiments", "--no-figures", "--no-diagnostics"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "higher spearman is better" in out and "=== holdout" in out
    assert train.main(["--predict-only", "--output-dir", str(tmp_path)]) == 0
    assert "predicted log return" in capsys.readouterr().out


def test_return_ladder(long_cache, tmp_path):
    out = tmp_path / "experiments.json"
    assert experiments.main(["--cv", "2", "--only", "persistence,pct_change only", "--out", str(out)]) == 0
    res = json.loads(out.read_text())
    assert res["task"] == "return" and res["primary_metric"] == "spearman" and len(res["results"]) == 2
