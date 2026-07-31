"""§6 検証手続き自体のテスト。"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from src.analysis import validation


def test_folds_are_contiguous_and_ordered():
    folds = validation.make_folds(0, 900, 9)
    assert len(folds) == 9
    for i in range(1, 9):
        assert folds[i][0] == folds[i - 1][1]      # 隙間も重複もない
    assert folds[0][0] == 0 and folds[-1][1] == 901


def test_walk_forward_freezes_train_choice(params):
    """訓練 fold で最良だった組み合わせが、そのまま次 fold の評価に使われること。"""
    rows = []
    for fold in range(3):
        for sigma in (3.0, 4.0):
            # fold0 では sigma=4 が良く、fold1 では sigma=3 が良い
            metric = {0: {3.0: 0.01, 4.0: 0.05}, 1: {3.0: 0.09, 4.0: -0.02},
                      2: {3.0: 0.0, 4.0: 0.0}}[fold][sigma]
            rows.append({"sigma": sigma, "fold": fold, "n": 100, "n_wins": 60,
                         "net_mean_pct": metric})
    wf = validation.walk_forward(pl.DataFrame(rows), ["sigma"], params)
    first = wf.row(0, named=True)
    assert '"sigma": 4.0' in first["selected"]
    assert first["oos_metric"] == -0.02      # 後知恵で sigma=3 を選ばない


def test_walk_forward_respects_min_wins(params):
    rows = [{"sigma": 3.0, "fold": f, "n": 10, "n_wins": 1, "net_mean_pct": 1.0} for f in range(2)]
    wf = validation.walk_forward(pl.DataFrame(rows), ["sigma"], params)
    assert wf.row(0, named=True)["reason"] == "insufficient_train_wins"


def test_deflated_sharpe_penalises_more_trials():
    rng = np.random.default_rng(0)
    r = rng.normal(0.02, 1.0, 500)
    few = validation.deflated_sharpe(r, n_trials=2, sr_variance=0.01)
    many = validation.deflated_sharpe(r, n_trials=5000, sr_variance=0.01)
    assert few["sr0"] < many["sr0"]
    assert many["dsr"] < few["dsr"]


def test_expected_max_sharpe_grows_with_trials():
    a = validation.expected_max_sharpe(10, 0.04)
    b = validation.expected_max_sharpe(1000, 0.04)
    assert 0 < a < b


def test_block_bootstrap_ci_contains_mean_and_respects_blocks():
    rng = np.random.default_rng(1)
    v = rng.normal(0.5, 1.0, 600)
    blocks = np.repeat(np.arange(60), 10)
    res = validation.block_bootstrap_ci(v, blocks, n_resamples=500, ci=0.95)
    assert res["lo"] < res["mean"] < res["hi"]
    assert res["n_blocks"] == 60
    assert res["p_le_zero"] < 0.05


def test_block_bootstrap_flags_zero_mean():
    rng = np.random.default_rng(2)
    v = rng.normal(0.0, 1.0, 600)
    blocks = np.repeat(np.arange(60), 10)
    res = validation.block_bootstrap_ci(v, blocks, n_resamples=500, ci=0.95)
    assert res["lo"] < 0 < res["hi"]


def test_day_blocks_group_by_day():
    ts = np.array([0, 100, 86_400, 90_000], dtype=np.int64)
    b = validation.day_blocks(ts, 86_400)
    assert list(b) == [0, 0, 1, 1]


def test_plateau_rejects_single_sharp_peak(params):
    grid = {"sigma": [3.0, 4.0, 5.0], "tick": [5.0, 10.0, 20.0]}
    rows = []
    for s in grid["sigma"]:
        for t in grid["tick"]:
            rows.append({"sigma": s, "tick": t, "net_mean_pct": 0.001})
    summary = pl.DataFrame(rows).with_columns(
        pl.when((pl.col("sigma") == 4.0) & (pl.col("tick") == 10.0))
        .then(1.0).otherwise(pl.col("net_mean_pct")).alias("net_mean_pct")
    )
    best = {"sigma": 4.0, "tick": 10.0, "net_mean_pct": 1.0}
    res = validation.plateau_check(summary, grid, best, params)
    assert not res["passed"]

    flat = pl.DataFrame(rows).with_columns(pl.lit(1.0).alias("net_mean_pct"))
    res2 = validation.plateau_check(flat, grid, best, params)
    assert res2["passed"]


def test_trials_log_accumulates(tmp_path, params):
    log = validation.TrialsLog(tmp_path / "trials.json")
    log.add(972, "grid sweep")
    log.add(5, "manual tuning")           # §9 手動調整も加算する
    assert log.total == 977
    assert len(log.read()["entries"]) == 2


def test_sample_gate(params):
    assert validation.sample_gate(30, params)["passed"]
    assert not validation.sample_gate(29, params)["passed"]


def test_sharpe_matches_definition():
    r = np.array([0.1, -0.05, 0.2, 0.0, 0.15])
    assert math.isclose(validation.sharpe(r), r.mean() / r.std(ddof=1), rel_tol=1e-12)
