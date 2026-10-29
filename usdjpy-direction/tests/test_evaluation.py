import numpy as np
import pandas as pd
import pytest

from usdjpy.data import make_target
from usdjpy.evaluation import (DirectionEvaluator, DirectionSlicer, compute_metrics, confidence_labels,
                               leaderboard, md_table, plot_report, positions, save_report, strategy_returns)
from usdjpy.features import make_feature_frame
from usdjpy.models import make_candidate


@pytest.fixture(scope="module")
def dev(bars):
    f = make_feature_frame(bars)
    t = make_target(bars)
    X = f.loc[t.index].iloc[130:]
    return X, t.loc[X.index, "y"], t.loc[X.index, "next_logret"]


def test_positions_and_costs():
    assert positions([0.6, 0.4, 0.5, 0.51], threshold=0.02).tolist() == [1, -1, 0, 0]
    net = strategy_returns([0.6, 0.6, 0.4], [0.001, -0.002, -0.001], cost_bps=1.0)
    # enter long (1 unit cost), hold, flip short (2 units cost)
    np.testing.assert_allclose(net, [0.001 - 1e-4, -0.002, 0.001 - 2e-4])


def test_metrics_hand_case():
    m = compute_metrics([0, 1, 1, 0], [0.2, 0.8, 0.6, 0.4], [-0.001, 0.002, 0.001, -0.001], cost_bps=0)
    assert m["auc"] == 1.0 and m["accuracy"] == 1.0 and m["strategy_bps"] == pytest.approx(12.5)
    single = compute_metrics([1, 1], [0.6, 0.7])
    assert np.isnan(single["auc"])


def test_walk_forward_folds_are_ordered_with_gap(dev):
    X, y, r = dev
    ev = DirectionEvaluator(cv=4, gap=24)
    for tr, te in ev._cv().split(X):
        assert tr.max() + 24 < te.min()
    rep = ev.evaluate(make_candidate("majority"), X, y, r, name="majority")
    f = rep.folds
    assert (f["train_end"] < f["test_start"]).all() and len(f) == 4
    assert (f["test_auc"] == 0.5).all()           # constant within a fold
    assert len(rep.y_true) == f["n_test"].sum()


def test_random_walk_has_no_edge(dev):
    """On a pure random walk the out-of-fold AUC must sit near 0.5 (leakage guard)."""
    X, y, r = dev
    rep = DirectionEvaluator(cv=3, gap=24).evaluate(make_candidate("lgbm"), X, y, r, name="lgbm")
    assert abs(rep.auc - 0.5) < 0.04


def test_slices_report_and_save(dev, tmp_path):
    X, y, r = dev
    ev = DirectionEvaluator(cv=3)
    board, reports = ev.compare({"majority": make_candidate("majority"),
                                 "momentum": make_candidate("momentum")}, X, y, r)
    assert list(board.columns)[:2] == ["auc", "log_loss"]
    s = DirectionSlicer().transform(X)
    assert set(s.columns) == {"session", "time_of_day", "weekday", "volatility", "regime"}
    rep = reports["momentum"]
    for name in ("session", "confidence", "weekday"):
        t = rep.slice_table(name)
        assert t["n"].sum() == len(rep.y_true)
    assert set(confidence_labels([0.5, 0.515, 0.53, 0.6])) == {"<1pp", "1-2pp", "2-5pp", "5pp+"}
    path = save_report(reports, leaderboard(reports), tmp_path, best="momentum", figures=True)
    assert path.exists() and (tmp_path / "figures" / "momentum.png").exists()
    assert "| model |" in md_table(board)
    import matplotlib.pyplot as plt
    plt.close(plot_report(rep))
