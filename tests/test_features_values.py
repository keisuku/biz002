"""§3.2 各特徴量が定義どおりの値になっているかの単体テスト。"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from src import features as feat
from tests.conftest import make_bars


def test_realized_vol_matches_definition(params):
    bars = make_bars(400, seed=5)
    f = feat.add_calm_columns(feat.compute_core_features(bars, params), None, 60, 10)
    r1 = np.diff(np.log(bars["close"].to_numpy()))
    i = 300
    expected = np.std(r1[i - 60: i], ddof=1) * math.sqrt(60)
    assert math.isclose(f["rv_calm"][i], expected, rel_tol=1e-9)
    # rv_calm_prev は calm_exclude 秒前の値
    assert math.isclose(f["rv_calm_prev"][i], f["rv_calm"][i - 10], rel_tol=1e-12)


def test_velocity_is_return_over_shifted_sigma(params):
    bars = make_bars(400, seed=6)
    f = feat.compute_core_features(bars, params)
    logc = np.log(bars["close"].to_numpy())
    k = 10
    i = 350
    ret = logc[i] - logc[i - k]
    assert math.isclose(f[f"ret_{k}s"][i], ret, rel_tol=1e-12)
    assert math.isclose(f[f"velocity_{k}"][i], abs(ret) / f[f"sigma_ref_{k}s"][i], rel_tol=1e-12)
    # 参照 σ は k 秒ぶんずれた窓（現在の k 秒リターンと重ならない）
    ret_series = pl.Series(logc) - pl.Series(logc).shift(k)
    window = ret_series[i - k - 120 + 1: i - k + 1].drop_nulls()
    assert math.isclose(f[f"sigma_ref_{k}s"][i], window.std(ddof=1), rel_tol=1e-9)


def test_tick_ofi_impact_ratios(params):
    bars = make_bars(400, seed=7)
    f = feat.compute_core_features(bars, params)
    i = 300
    tc = bars["trade_count"].to_numpy()
    vol = bars["volume"].to_numpy()
    ofi = bars["ofi"].to_numpy()
    close = bars["close"].to_numpy()

    assert f["tick_count_w"][i] == tc[i - 9: i + 1].sum()
    assert math.isclose(f["ofi_w"][i], ofi[i - 9: i + 1].sum(), rel_tol=1e-12)
    assert math.isclose(f["vol_w"][i], vol[i - 9: i + 1].sum(), rel_tol=1e-12)
    assert math.isclose(f["ofi_ratio"][i], abs(f["ofi_w"][i]) / f["vol_w"][i], rel_tol=1e-12)
    impact = abs(close[i] / close[i - 10] - 1.0) / vol[i - 9: i + 1].sum()
    assert math.isclose(f["impact"][i], impact, rel_tol=1e-12)
    assert math.isclose(f["impact_ratio"][i], f["impact"][i] / f["impact_ref"][i], rel_tol=1e-12)


def test_ofi_ratio_is_bounded_and_one_sided_flow_hits_one(params):
    n = 300
    bars = make_bars(n, seed=8).with_columns(
        pl.col("volume").alias("taker_buy_volume"),
        pl.lit(0.0).alias("taker_sell_volume"),
    ).with_columns((pl.col("taker_buy_volume") - pl.col("taker_sell_volume")).alias("ofi"))
    f = feat.compute_core_features(bars, params)
    vals = f["ofi_ratio"].drop_nulls().to_numpy()
    assert vals.max() <= 1.0 + 1e-12
    assert math.isclose(vals[-1], 1.0, rel_tol=1e-12)


def test_data_ok_false_inside_gaps(params):
    n = 400
    bars = make_bars(n, seed=9).to_dicts()
    for i in range(200, 260):  # 60 秒の欠損（max_fill_gap=30 を超える）
        bars[i]["is_filled"] = True
        bars[i]["trade_count"] = 0
        bars[i]["volume"] = 0.0
    f = feat.compute_core_features(pl.DataFrame(bars), params)
    assert not f["data_ok"][230]
    assert f["has_long_gap_1h"][280]


def test_calm_flag_requires_low_prior_volatility(params):
    n = 600
    ts = np.arange(1_700_000_000, 1_700_000_000 + n, dtype=np.int64)
    bars = make_bars(n, seed=10).with_columns(pl.Series("ts", ts))
    thr = pl.DataFrame({"ts": ts[::10], "calm_threshold": np.full(len(ts[::10]), 1e-3)})
    f = feat.add_calm_columns(feat.compute_core_features(bars, params), thr, 60, 10)
    assert f["is_calm"][400]  # 閾値が十分高ければ凪と判定される
    thr_low = thr.with_columns(pl.lit(0.0).alias("calm_threshold"))
    f2 = feat.add_calm_columns(feat.compute_core_features(bars, params), thr_low, 60, 10)
    assert not f2["is_calm"][400]
