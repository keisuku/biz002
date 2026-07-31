"""「乗って放置」手法の R 倍率測定のテスト。

この測定の要点は「小さく負けて、たまに大きく勝つ」形を正しく扱えること。
中央値がマイナスでも平均 R がプラスなら成立する、という非対称性を壊さないよう固定する。
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.analysis import trend_ride as ride
from tests.conftest import make_bars


def _bars(n, close, seed=1):
    b = make_bars(n, seed=seed)
    c = np.asarray(close, dtype=float)
    return b.with_columns(
        pl.Series("close", c), pl.Series("high", c), pl.Series("low", c),
        pl.Series("open", c), pl.Series("gap_max_bps", np.zeros(n)),
    )


def _starts(bars, idx, direction=1, impulse_bps=100.0):
    return pl.DataFrame({
        "ts": [int(bars["ts"][idx])],
        "direction": pl.Series([direction], dtype=pl.Int8),
        "impulse_bps": [impulse_bps],
        "symbol": ["T"],
    })


def test_stop_loss_gives_exactly_minus_one_r(params):
    n = 2000
    close = np.full(n, 100.0)
    close[200:] = 98.0                      # 初動 100bps → ストップ幅 1% → 99.0
    bars = _bars(n, close)
    r = ride.simulate_rides(_starts(bars, 100), bars, params, max_hold=900)
    row = r.row(0, named=True)
    assert row["exit_reason"] == "stop"
    assert row["r_multiple"] == pytest.approx(-1.0, abs=1e-9)


def test_a_long_run_gives_a_large_positive_r(params):
    n = 4000
    close = np.full(n, 100.0)
    close[110:] = 105.0                     # +5% 伸びる。ストップ幅 1% → +5R
    bars = _bars(n, close)
    r = ride.simulate_rides(_starts(bars, 100), bars, params, max_hold=3600)
    row = r.row(0, named=True)
    assert row["exit_reason"] == "max_hold"
    assert row["r_multiple"] == pytest.approx(5.0, rel=1e-6)
    assert row["mfe_r"] == pytest.approx(5.0, rel=1e-6)


def test_short_side_is_symmetric(params):
    n = 4000
    close = np.full(n, 100.0)
    close[110:] = 95.0
    bars = _bars(n, close)
    r = ride.simulate_rides(_starts(bars, 100, direction=-1), bars, params, max_hold=3600)
    assert r.row(0, named=True)["r_multiple"] == pytest.approx(5.0, rel=1e-6)


def test_stop_is_checked_before_the_run(params):
    """先に逆行してストップに掛かれば、その後いくら伸びても -1R。"""
    n = 4000
    close = np.full(n, 100.0)
    close[150:200] = 98.0                   # まず逆行してストップ
    close[200:] = 120.0                     # その後大きく伸びる
    bars = _bars(n, close)
    r = ride.simulate_rides(_starts(bars, 100), bars, params, max_hold=3600)
    row = r.row(0, named=True)
    assert row["exit_reason"] == "stop"
    assert row["r_multiple"] == pytest.approx(-1.0, abs=1e-9)


def test_entry_is_taken_at_the_configured_age(params):
    n = 2000
    close = np.linspace(100.0, 101.0, n)
    bars = _bars(n, close)
    age = int(params.get_path("trend_ride.entry_age_seconds"))
    r = ride.simulate_rides(_starts(bars, 100), bars, params, max_hold=900)
    assert r.row(0, named=True)["entry_price"] == pytest.approx(close[100 + age])


def test_long_real_frame_can_transition_from_null_to_finite_context(params):
    """先頭100件超がNullでも、後続のtrend_z実数で型推論が停止しない。"""
    n = 5000
    bars = _bars(n, np.full(n, 100.0))
    indices = list(range(100, 230))
    starts = pl.DataFrame({
        "ts": [int(bars["ts"][i]) for i in indices],
        "direction": pl.Series([1] * len(indices), dtype=pl.Int8),
        "impulse_bps": [100.0] * len(indices),
        "symbol": ["T"] * len(indices),
    })
    context = bars.select("ts").with_columns(
        pl.when(pl.int_range(0, n) < 220)
        .then(pl.lit(None, dtype=pl.Float64))
        .otherwise(pl.lit(0.5))
        .alias("trend_z"),
        pl.lit(None, dtype=pl.Float64).alias("vol_z"),
    )
    result = ride.simulate_rides(
        starts, bars, params, max_hold=900, context=context
    )
    assert result.height == len(indices)
    assert result["trend_z"].dtype == pl.Float64
    assert result["trend_z"].drop_nulls().len() > 0


def test_mean_r_positive_while_median_is_negative(params):
    """中央値がマイナスでも平均 R がプラスなら成立する、という非対称性。

    この形を潰す集計をしていたら（＝中央値で判定していたら）このテストが落ちる。
    """
    rides = pl.DataFrame({
        "max_hold_s": [3600] * 10,
        "r_multiple": [-1.0] * 8 + [12.0, 9.0],   # 8 敗 2 勝
        "mfe_r": [0.2] * 8 + [12.0, 9.0],
        "cost_r": [0.02] * 10,
        "hold_seconds": [100] * 10,
        "exit_reason": ["stop"] * 8 + ["max_hold"] * 2,
    })
    s = ride.summarize(rides, params)
    row = s.row(0, named=True)
    assert row["median_r"] < 0
    assert row["mean_r"] == pytest.approx(1.3)
    assert row["win_rate"] == pytest.approx(0.2)
    assert row["profit_factor"] == pytest.approx(21.0 / 8.0)
    assert row["p_ge_3r"] == pytest.approx(0.2)
    v = ride.verdict(s, params)
    assert v["verdict"] == "POSITIVE_EXPECTANCY"


def test_verdict_is_negative_when_the_tail_is_missing(params):
    rides = pl.DataFrame({
        "max_hold_s": [3600] * 10,
        "r_multiple": [-1.0] * 8 + [1.0, 1.5],
        "mfe_r": [0.2] * 10, "cost_r": [0.02] * 10,
        "hold_seconds": [100] * 10, "exit_reason": ["stop"] * 10,
    })
    v = ride.verdict(ride.summarize(rides, params), params)
    assert v["verdict"] == "NEGATIVE_EXPECTANCY"


def test_trend_context_is_backward_looking_only():
    n = 4000
    close = np.concatenate([np.full(2000, 100.0), np.full(2000, 110.0)])
    bars = _bars(n, close)
    ctx = ride.trend_context(bars, lookback=600, ref_seconds=1200)
    # 上昇の前の時点では、直前トレンドはまだ 0
    assert ctx["trend_ret_bps"][1500] == pytest.approx(0.0, abs=1e-6)
    # 上昇をまたぐ窓（2300-600=1700 は上昇前）では正
    assert ctx["trend_ret_bps"][2300] > 0
    # 上昇が窓から抜けきれば再び 0（後ろ向きの窓しか見ていない証拠）
    assert ctx["trend_ret_bps"][2800] == pytest.approx(0.0, abs=1e-6)


def test_stratify_splits_with_and_counter_trend(params):
    rides = pl.DataFrame({
        "max_hold_s": [3600] * 4,
        "r_multiple": [3.0, 2.0, -1.0, -1.0],
        "mfe_r": [3.0, 2.0, 0.1, 0.1], "cost_r": [0.02] * 4,
        "hold_seconds": [10] * 4, "exit_reason": ["max_hold"] * 4,
        "direction": pl.Series([1, 1, -1, -1], dtype=pl.Int8),
        "trend_z": [2.0, 2.0, 2.0, 2.0],
        "vol_z": [3.0, 3.0, 0.0, 0.0],
        "trend_align": [True, True, False, False],
    })
    s = ride.stratify(rides, params)
    with_trend = s.filter(pl.col("stratum") == "align:with_trend").row(0, named=True)
    counter = s.filter(pl.col("stratum") == "align:counter_trend").row(0, named=True)
    assert with_trend["mean_r"] == pytest.approx(2.5)
    assert counter["mean_r"] == pytest.approx(-1.0)


def test_stratify_does_not_pool_signal_definitions(params):
    rides = pl.DataFrame({
        "window_s": [3, 3, 10, 10],
        "sigma_mult": [4.0, 4.0, 6.0, 6.0],
        "max_hold_s": [3600] * 4,
        "r_multiple": [2.0, 2.0, -1.0, -1.0],
        "mfe_r": [2.0, 2.0, 0.1, 0.1],
        "cost_r": [0.02] * 4,
        "hold_seconds": [3600] * 4,
        "exit_reason": ["max_hold"] * 4,
        "direction": pl.Series([1, 1, 1, 1], dtype=pl.Int8),
        "trend_z": [1.0] * 4,
        "vol_z": [0.0] * 4,
        "trend_align": [True] * 4,
    })
    strata = ride.stratify(rides, params)
    aligned = strata.filter(pl.col("stratum") == "align:with_trend")
    assert aligned.height == 2
    by_window = {
        int(row["window_s"]): row for row in aligned.iter_rows(named=True)
    }
    assert by_window[3]["mean_r"] == pytest.approx(2.0)
    assert by_window[10]["mean_r"] == pytest.approx(-1.0)


def test_random_entries_match_direction_and_stop_distribution(params):
    """対照群は「いつ入るか」だけが違い、方向とストップ幅の分布は同じであること。"""
    n = 5000
    bars = _bars(n, np.full(n, 100.0))
    template = pl.DataFrame({
        "ts": bars["ts"].to_numpy()[[100, 200, 300]],
        "direction": pl.Series([1, -1, -1], dtype=pl.Int8),
        "impulse_bps": [50.0, 80.0, 120.0],
        "symbol": ["T"] * 3,
    })
    r = ride.random_entries(bars, template, seed=1, multiplier=10)
    assert r.height == 30
    assert set(r["direction"].to_list()) <= {1, -1}
    assert set(r["impulse_bps"].to_list()) <= {50.0, 80.0, 120.0}
    # 時刻はテンプレートの 3 点に限定されず、広く散らばること
    assert r["ts"].n_unique() > 3
    assert r["ts"].is_sorted()


def test_random_entries_are_deterministic_for_a_seed(params):
    n = 3000
    bars = _bars(n, np.full(n, 100.0))
    t = pl.DataFrame({"ts": bars["ts"].to_numpy()[[10]],
                      "direction": pl.Series([1], dtype=pl.Int8),
                      "impulse_bps": [30.0], "symbol": ["T"]})
    a = ride.random_entries(bars, t, seed=7, multiplier=20)
    b = ride.random_entries(bars, t, seed=7, multiplier=20)
    assert a["ts"].to_list() == b["ts"].to_list()


def test_random_entries_stay_inside_the_requested_region(params):
    n = 3000
    bars = _bars(n, np.full(n, 100.0))
    t = pl.DataFrame({"ts": bars["ts"].to_numpy()[[10]],
                      "direction": pl.Series([1], dtype=pl.Int8),
                      "impulse_bps": [30.0], "symbol": ["T"]})
    lo = int(bars["ts"][1000])
    hi = int(bars["ts"][2000])
    r = ride.random_entries(bars, t, seed=3, multiplier=50, region=(lo, hi))
    assert r["ts"].min() >= lo and r["ts"].max() < hi


def test_robustness_flags_a_result_carried_by_one_block():
    """1 ブロックだけが利益を作っている場合、それを検出すること。"""
    block = 259_200
    rides = pl.DataFrame({
        "impulse_ts": [0, 100, block, block + 100, 2 * block, 2 * block + 100],
        "r_multiple": [-1.0, -1.0, 20.0, 20.0, -1.0, -1.0],
    })
    rob = ride.robustness(rides, block)
    v = ride.robustness_verdict(rob)
    assert v["verdict"] == "DEPENDS_ON_ONE_BLOCK"
    assert v["mean_r_all"] > 0
    assert v["mean_r_without_top_block"] < 0


def test_robustness_passes_when_gains_are_spread_out():
    block = 259_200
    rides = pl.DataFrame({
        "impulse_ts": [0, block, 2 * block, 3 * block],
        "r_multiple": [3.0, 3.0, 3.0, 3.0],
    })
    v = ride.robustness_verdict(ride.robustness(rides, block))
    assert v["verdict"] == "SURVIVES_LEAVE_ONE_OUT"
    assert v["mean_r_without_top_block"] > 0


def test_signal_vs_random_separates_the_two_groups(params):
    rides = pl.DataFrame({
        "max_hold_s": [3600] * 4,
        "source": ["signal", "signal", "random", "random"],
        "r_multiple": [4.0, 2.0, -1.0, 0.0],
        "mfe_r": [4.0, 2.0, 0.1, 0.2], "cost_r": [0.02] * 4,
        "hold_seconds": [10] * 4, "exit_reason": ["max_hold"] * 4,
    })
    t = ride.signal_vs_random(rides, params)
    by_source = {r["source"]: r for r in t.iter_rows(named=True)}
    assert by_source["signal"]["mean_r"] == pytest.approx(3.0)
    assert by_source["random"]["mean_r"] == pytest.approx(-0.5)
