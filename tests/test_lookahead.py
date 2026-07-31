"""§6.7 ルックアヘッド検査。

「発火時点以降のデータを渡すと計算結果が変わるか」を全特徴量に対して機械的に確認する。
変わったらバグ。本プロジェクトで最も起こりやすい致命的バグなので、特徴量を追加したら
必ずこのテストの対象一覧（features.feature_column_names）に載せること。
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from src import features as feat
from src import outcomes as out_mod
from tests.conftest import make_bars


def _cmp(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if math.isnan(float(a)) and math.isnan(float(b)):
            return True
        return math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-15)
    return a == b


def test_features_do_not_change_when_future_is_appended(params):
    bars = make_bars(900, seed=1)
    full = feat.compute_core_features(bars, params)
    cols = [c for c in feat.feature_column_names(params) if c in full.columns]
    assert len(cols) >= 15, "特徴量の一覧が取れていない"

    for t_idx in (400, 600, 800):
        truncated = feat.compute_core_features(bars.head(t_idx + 1), params)
        row_full = full.row(t_idx, named=True)
        row_trunc = truncated.row(t_idx, named=True)
        for c in cols:
            assert _cmp(row_full[c], row_trunc[c]), (
                f"lookahead detected in {c!r} at index {t_idx}: "
                f"{row_full[c]} != {row_trunc[c]}"
            )


def test_calm_threshold_uses_only_past_samples():
    """凪の分位点は、その時刻より前のサンプルだけから作られること。"""
    n = 2000
    ts = np.arange(0, n * 60, 60, dtype=np.int64)
    rv = np.linspace(1.0, 2.0, n)
    rv[-1] = 1e9  # 未来に極端な値を置く
    mr = pl.DataFrame({"ts": ts, "rv_60": rv})
    thr = feat.calm_threshold_series(mr, 60, 25, ref_days=7, sample_seconds=60)
    # 最終行の閾値が、直前までの分布から作られている（外れ値の影響を受けない）こと
    trunc = feat.calm_threshold_series(mr.head(n - 1), 60, 25, ref_days=7, sample_seconds=60)
    assert _cmp(thr["calm_threshold"][n - 2], trunc["calm_threshold"][n - 2])
    assert thr["calm_threshold"][n - 1] < 1e6


def test_reference_windows_exclude_the_firing_move(params):
    """発火の急変が自分自身の参照統計（sigma_ref など）に混入しないこと。"""
    bars = make_bars(600, seed=2).to_dicts()
    for i in range(500, 510):  # 500〜509 秒に大きな上昇を差し込む
        bars[i]["close"] *= 1.02
        for j in range(i + 1, 600):
            bars[j]["close"] *= 1.0
    spiked = pl.DataFrame(bars)
    base = pl.DataFrame(make_bars(600, seed=2).to_dicts())
    k = int(params.get_path("features.velocity_primary_k"))
    f_spiked = feat.compute_core_features(spiked, params)
    f_base = feat.compute_core_features(base, params)
    # 発火の 1 秒前の参照統計は、発火の有無で変わってはならない
    assert _cmp(f_spiked[f"sigma_ref_{k}s"][499], f_base[f"sigma_ref_{k}s"][499])


def test_outcomes_exclude_the_firing_second(params):
    """MFE は発火した秒自身の高値を含めない（含めると見えている値動きで儲かることになる）。"""
    n = 200
    close = np.full(n, 100.0)
    high = close.copy()
    high[100] = 130.0  # 発火した秒だけ巨大な高値
    bars = make_bars(n, seed=3).with_columns(
        pl.Series("close", close), pl.Series("high", high), pl.Series("low", close)
    )
    events = pl.DataFrame({
        "event_ts": [int(bars["ts"][100])], "direction": [1], "entry_price": [100.0],
        "symbol": ["T"], "range_bps_1s": [1.0],
    })
    res = out_mod.add_outcomes(events, bars, params)
    assert res["mfe_15"][0] == 0.0
