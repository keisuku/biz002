"""§3.3 発火抽出 / §3.4 結果指標 / §4.4 出口ルールのテスト。"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src import events as ev_mod
from src import exits as exits_mod
from src import outcomes as out_mod
from tests.conftest import make_bars


def test_cooldown_merges_consecutive_fires():
    ts = np.array([100, 110, 130, 200, 205, 400], dtype=np.int64)
    d = np.array([1, 1, 1, 1, 1, 1], dtype=np.int64)
    keep = ev_mod.apply_cooldown(ts, d, cooldown=60, scope="direction")
    assert list(keep) == [True, False, False, True, False, True]


def test_cooldown_is_per_direction():
    ts = np.array([100, 110, 120], dtype=np.int64)
    d = np.array([1, -1, 1], dtype=np.int64)
    keep = ev_mod.apply_cooldown(ts, d, cooldown=60, scope="direction")
    assert list(keep) == [True, True, False]
    keep_sym = ev_mod.apply_cooldown(ts, d, cooldown=60, scope="symbol")
    assert list(keep_sym) == [True, False, False]


def _fire_frame(params, n=200):
    """発火条件をちょうど 1 点だけ満たす特徴量フレームを人工的に作る。"""
    ts = np.arange(1_700_000_000, 1_700_000_000 + n, dtype=np.int64)
    k = int(params.get_path("features.velocity_primary_k"))
    f = pl.DataFrame({
        "ts": ts,
        "close": np.full(n, 100.0),
        "volume": np.full(n, 1.0),
        "trade_count": np.full(n, 5, dtype=np.int64),
        "range_bps_1s": np.full(n, 2.0),
        "gap_max_bps_w": np.full(n, 1.0),
        "gap_mean_bps_ref": np.full(n, 1.0),
        "rv_calm_prev": np.full(n, 1e-5),
        "calm_threshold": np.full(n, 1e-4),
        "tick_count_w": np.full(n, 50.0),
        "tick_ref": np.full(n, 5.0),
        "tick_ratio": np.full(n, 1.0),
        "ofi_w": np.full(n, 0.1),
        "vol_w": np.full(n, 1.0),
        "ofi_ratio": np.full(n, 0.1),
        "impact": np.full(n, 1.0),
        "impact_ref": np.full(n, 1.0),
        "impact_ratio": np.full(n, 1.0),
        "active_ratio_1h": np.full(n, 1.0),
        "is_calm": np.ones(n, dtype=bool),
        "data_ok": np.ones(n, dtype=bool),
        f"ret_{k}s": np.full(n, 1e-4),
        f"sigma_ref_{k}s": np.full(n, 1e-4),
        f"velocity_{k}": np.full(n, 1.0),
    })
    for kk in params.get_path("features.velocity_ks"):
        if f"velocity_{kk}" not in f.columns:
            f = f.with_columns(pl.lit(1.0).alias(f"velocity_{kk}"),
                               pl.lit(1e-4).alias(f"ret_{kk}s"),
                               pl.lit(1e-4).alias(f"sigma_ref_{kk}s"))
    hit = np.zeros(n, dtype=bool)
    hit[100] = True
    return f.with_columns(
        pl.when(pl.Series(hit)).then(9.0).otherwise(pl.col(f"velocity_{k}")).alias(f"velocity_{k}"),
        pl.when(pl.Series(hit)).then(30.0).otherwise(pl.col("tick_ratio")).alias("tick_ratio"),
        pl.when(pl.Series(hit)).then(0.9).otherwise(pl.col("ofi_ratio")).alias("ofi_ratio"),
        pl.when(pl.Series(hit)).then(6.0).otherwise(pl.col("impact_ratio")).alias("impact_ratio"),
    )


def test_all_four_conditions_must_hold(params):
    f = _fire_frame(params)
    ev = ev_mod.extract_events(f, params, "T")
    assert ev.height == 1
    assert ev["direction"][0] == 1
    # どれか 1 つでも閾値を下回れば発火しない
    for col, val in (("tick_ratio", 1.0), ("ofi_ratio", 0.1), ("impact_ratio", 1.0)):
        broken = f.with_columns(pl.lit(val).alias(col))
        assert ev_mod.extract_events(broken, params, "T").height == 0
    not_calm = f.with_columns(pl.lit(False).alias("is_calm"))
    assert ev_mod.extract_events(not_calm, params, "T").height == 0
    bad_data = f.with_columns(pl.lit(False).alias("data_ok"))
    assert ev_mod.extract_events(bad_data, params, "T").height == 0


def test_mfe_mae_signs_for_both_directions(params):
    n = 300
    close = np.full(n, 100.0)
    close[101:130] = 101.0     # 発火の次の秒から +1%
    bars = make_bars(n, seed=11).with_columns(
        pl.Series("close", close), pl.Series("high", close), pl.Series("low", close)
    )
    ts = int(bars["ts"][100])
    long_ev = pl.DataFrame({"event_ts": [ts], "direction": [1], "entry_price": [100.0],
                            "symbol": ["T"], "range_bps_1s": [2.0]})
    short_ev = long_ev.with_columns(pl.lit(-1).cast(pl.Int8).alias("direction"))
    lo = out_mod.add_outcomes(long_ev, bars, params)
    sh = out_mod.add_outcomes(short_ev, bars, params)
    assert lo["mfe_15"][0] == pytest.approx(1.0, rel=1e-9)
    assert lo["mae_15"][0] == pytest.approx(0.0, abs=1e-12)
    assert sh["mfe_15"][0] == pytest.approx(0.0, abs=1e-12)
    assert sh["mae_15"][0] == pytest.approx(-1.0, rel=1e-9)
    assert lo["ret_15"][0] == pytest.approx(1.0, rel=1e-9)
    assert sh["ret_15"][0] == pytest.approx(-1.0, rel=1e-9)


def test_costs_are_round_trip(params):
    ev = pl.DataFrame({"event_ts": [1], "direction": [1], "entry_price": [100.0], "symbol": ["T"],
                       "gap_mean_bps_ref": [2.0], "gap_max_bps_w": [3.0], "range_bps_1s": [4.0]})
    out = out_mod.add_costs(ev, params)
    assert out["fee_pct"][0] == pytest.approx(0.10)      # 0.05% × 2
    slip = out["slippage_est_pct"][0]
    assert slip == pytest.approx((1.0 * 2.0 / 2 + 1.0 * 3.0 + 0.5 * 4.0) / 100.0)
    assert out["cost_pct"][0] == pytest.approx(0.10 + 2 * slip)
    doubled = out_mod.add_costs(ev, params, multiplier=2.0)
    assert doubled["cost_pct"][0] == pytest.approx(2 * out["cost_pct"][0])


def _exit_bars(n, close, tc, ofi, seed=12):
    b = make_bars(n, seed=seed)
    return b.with_columns(
        pl.Series("close", close), pl.Series("high", close), pl.Series("low", close),
        pl.Series("trade_count", np.array(tc, dtype=np.int64)),
        pl.Series("ofi", np.array(ofi, dtype=float)),
        pl.Series("gap_max_bps", np.zeros(n)),
    )


def test_e1_fires_when_tick_rate_collapses(params):
    n = 200
    close = np.full(n, 100.0)
    tc = np.full(n, 100)
    tc[120:] = 5           # 発火時ピークの 5%
    bars = _exit_bars(n, close, tc, np.ones(n))
    ev = pl.DataFrame({"event_ts": [int(bars["ts"][100])], "direction": [1],
                       "entry_price": [100.0], "symbol": ["T"], "range_bps_1s": [10.0],
                       "slippage_est_pct": [0.0], "fee_pct": [0.0]})
    sim = exits_mod.simulate(ev, bars, params, fixed_horizon=30)
    e1 = sim.filter(pl.col("rule_id") == "E1_20").row(0, named=True)
    assert e1["exit_reason"] == "E1_20"
    assert e1["hold_seconds"] == 20      # index 120 = 発火 +20 秒


def test_e2_fires_on_ofi_sign_flip(params):
    n = 200
    ofi = np.ones(n)
    ofi[110:] = -1.0
    bars = _exit_bars(n, np.full(n, 100.0), np.full(n, 100), ofi)
    ev = pl.DataFrame({"event_ts": [int(bars["ts"][100])], "direction": [1],
                       "entry_price": [100.0], "symbol": ["T"], "range_bps_1s": [10.0],
                       "slippage_est_pct": [0.0], "fee_pct": [0.0]})
    sim = exits_mod.simulate(ev, bars, params, fixed_horizon=30)
    e2 = sim.filter(pl.col("rule_id") == "E2").row(0, named=True)
    assert e2["exit_reason"] == "E2" and e2["hold_seconds"] == 10


def test_hard_stop_takes_precedence_and_caps_loss(params):
    n = 200
    close = np.full(n, 100.0)
    close[105:] = 99.0     # 逆行。K=2 × レンジ 50bps → ストップは -1% = 99.0
    bars = _exit_bars(n, close, np.full(n, 100), np.ones(n))
    ev = pl.DataFrame({"event_ts": [int(bars["ts"][100])], "direction": [1],
                       "entry_price": [100.0], "symbol": ["T"], "range_bps_1s": [50.0],
                       "slippage_est_pct": [0.0], "fee_pct": [0.0]})
    sim = exits_mod.simulate(ev, bars, params, fixed_horizon=30)
    row = sim.filter(pl.col("rule_id") == "E2").row(0, named=True)
    assert row["exit_reason"] == "hard_stop"
    # K=2, レンジ 100bps → ストップは -2% だが、その秒の安値 99.0 より悪くはならない
    assert row["gross_pct"] == pytest.approx(-1.0, rel=1e-9)


def test_max_hold_is_enforced(params):
    n = 400
    bars = _exit_bars(n, np.full(n, 100.0), np.full(n, 100), np.ones(n))
    ev = pl.DataFrame({"event_ts": [int(bars["ts"][100])], "direction": [1],
                       "entry_price": [100.0], "symbol": ["T"], "range_bps_1s": [1000.0],
                       "slippage_est_pct": [0.0], "fee_pct": [0.0]})
    sim = exits_mod.simulate(ev, bars, params, fixed_horizon=30)
    row = sim.filter(pl.col("rule_id") == "E2").row(0, named=True)
    assert row["exit_reason"] == "max_hold"
    assert row["hold_seconds"] == int(params.get_path("exits.max_hold_seconds"))


def test_e4_is_fixed_time(params):
    n = 300
    bars = _exit_bars(n, np.full(n, 100.0), np.full(n, 100), np.ones(n))
    ev = pl.DataFrame({"event_ts": [int(bars["ts"][100])], "direction": [1],
                       "entry_price": [100.0], "symbol": ["T"], "range_bps_1s": [1000.0],
                       "slippage_est_pct": [0.0], "fee_pct": [0.0]})
    sim = exits_mod.simulate(ev, bars, params, fixed_horizon=30)
    row = sim.filter(pl.col("rule_id") == "E4_30s").row(0, named=True)
    assert row["hold_seconds"] == 30
