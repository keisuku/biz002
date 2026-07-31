"""Phase 1 の集約（aggTrades → 1 秒バー）と is_buyer_maker の極性テスト。"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from src.data.to_seconds import SECONDS_PER_DAY, aggtrades_to_seconds, day_start_ts, gap_report
from src.data.verify_flags import check_bars


def _trades(ts_ms, price, qty, maker):
    return pl.DataFrame({
        "agg_trade_id": np.arange(len(ts_ms), dtype=np.int64),
        "ts_ms": np.array(ts_ms, dtype=np.int64),
        "price": np.array(price, dtype=float),
        "quantity": np.array(qty, dtype=float),
        "is_buyer_maker": np.array(maker, dtype=bool),
    })


def test_ohlcv_and_taker_split():
    t0 = day_start_ts(date(2024, 1, 1)) * 1000
    tr = _trades(
        [t0 + 100, t0 + 500, t0 + 900, t0 + 1500],
        [100.0, 102.0, 101.0, 105.0],
        [1.0, 2.0, 3.0, 4.0],
        [False, True, False, True],   # taker buy, taker sell, taker buy, taker sell
    )
    bars = aggtrades_to_seconds(tr, day=date(2024, 1, 1))
    assert bars.height == SECONDS_PER_DAY
    first = bars.row(0, named=True)
    assert first["open"] == 100.0 and first["high"] == 102.0
    assert first["low"] == 100.0 and first["close"] == 101.0
    assert first["volume"] == 6.0
    assert first["trade_count"] == 3
    # is_buyer_maker=False がテイカー買い
    assert first["taker_buy_volume"] == 4.0
    assert first["taker_sell_volume"] == 2.0
    assert first["ofi"] == 2.0
    assert not first["is_filled"]


def test_empty_seconds_are_filled_and_flagged():
    t0 = day_start_ts(date(2024, 1, 1)) * 1000
    tr = _trades([t0, t0 + 5000], [100.0, 110.0], [1.0, 1.0], [False, False])
    bars = aggtrades_to_seconds(tr, day=date(2024, 1, 1))
    assert bars["is_filled"][1] and bars["close"][1] == 100.0
    assert bars["volume"][1] == 0.0 and bars["trade_count"][1] == 0
    assert not bars["is_filled"][5]
    gaps = gap_report(bars, max_gap=2)
    assert gaps.height >= 1
    assert int(gaps["seconds"].max()) > 2


def test_gap_bps_uses_previous_trade_price():
    t0 = day_start_ts(date(2024, 1, 1)) * 1000
    tr = _trades([t0, t0 + 10, t0 + 20], [100.0, 100.1, 100.0], [1.0, 1.0, 1.0], [False] * 3)
    bars = aggtrades_to_seconds(tr, day=date(2024, 1, 1))
    assert abs(bars["gap_max_bps"][0] - 10.0) < 0.1  # 0.1/100 = 10bps


def test_is_buyer_maker_polarity_check_detects_inversion():
    """フラグを反転させたデータでは verify_flags が FAIL を返すこと。"""
    rng = np.random.default_rng(0)
    n = 4000
    t0 = 1_700_000_000
    ret = rng.normal(0, 1e-4, n)
    ret[2000:2010] = 5e-3  # 急騰
    close = 100 * np.exp(np.cumsum(ret))
    vol = np.full(n, 10.0)
    # 上昇時にテイカー買いが優勢 = ofi が正
    ofi = np.sign(ret) * vol * 0.6
    bars = pl.DataFrame({
        "ts": np.arange(t0, t0 + n, dtype=np.int64), "close": close,
        "ofi": ofi, "volume": vol,
    })
    good = check_bars(bars)
    assert good["verdict"] == "PASS"
    flipped = bars.with_columns((-pl.col("ofi")).alias("ofi"))
    bad = check_bars(flipped)
    assert bad["verdict"] == "FAIL_CHECK_FLAG_POLARITY"
