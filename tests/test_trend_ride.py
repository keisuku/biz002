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
