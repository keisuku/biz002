"""集約ステップ（aggTrades → 1 秒バー）のルックアヘッド検査。

`tests/test_lookahead.py` は「すでに密になった 1 秒バー」を入力にして特徴量を検査する。
そのため、密にする過程そのもので未来の価格が過去の秒に入る事故は拾えなかった。
実際に `close.forward_fill().backward_fill()` で、その日の最初の約定より前の秒に
初約定価格（=未来の価格）が入っていた。ここで固定する。
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from src.data.to_seconds import aggtrades_to_seconds, build_day, day_start_ts
from src.data.download import raw_path


def _trades(day: date, offsets_s, prices, qty=1.0):
    t0 = day_start_ts(day) * 1000
    n = len(offsets_s)
    return pl.DataFrame({
        "agg_trade_id": np.arange(n, dtype=np.int64),
        "ts_ms": np.array([t0 + int(o * 1000) for o in offsets_s], dtype=np.int64),
        "price": np.array(prices, dtype=float),
        "quantity": np.full(n, qty, dtype=float),
        "is_buyer_maker": np.zeros(n, dtype=bool),
        "first_trade_id": np.arange(n, dtype=np.int64),
        "last_trade_id": np.arange(n, dtype=np.int64),
    })


def test_seconds_before_the_first_trade_never_hold_a_future_price():
    day = date(2025, 1, 1)
    bars = aggtrades_to_seconds(_trades(day, [100, 101, 102], [500.0, 501.0, 502.0]), day)
    # 最初の約定は 100 秒目。0〜99 秒目にその価格が入っていてはいけない。
    leading = bars.head(100)
    assert leading["close"].null_count() == 100
    assert bool(leading["is_filled"].all())
    assert bars["close"][100] == 500.0


def test_previous_day_close_fills_the_leading_seconds():
    day = date(2025, 1, 1)
    bars = aggtrades_to_seconds(
        _trades(day, [100, 101], [500.0, 501.0]), day, prev_close=480.0
    )
    assert bars["close"][0] == 480.0     # 過去の価格で埋まる
    assert bars["close"][99] == 480.0
    assert bars["close"][100] == 500.0


def test_day_boundary_gap_is_measured_from_the_previous_close():
    day = date(2025, 1, 1)
    plain = aggtrades_to_seconds(_trades(day, [0], [505.0]), day)
    assert plain["gap_max_bps"][0] == 0.0            # 前日終値が無ければ 0
    carried = aggtrades_to_seconds(_trades(day, [0], [505.0]), day, prev_close=500.0)
    assert abs(carried["gap_max_bps"][0] - 100.0) < 1e-6   # 5/500 = 1% = 100bps


def test_build_day_carries_the_previous_day_close(tmp_path, params):
    p = params.with_overrides({
        "data.raw_dir": str(tmp_path / "raw"),
        "data.bars_dir": str(tmp_path / "bars"),
    })
    d0, d1 = date(2025, 1, 1), date(2025, 1, 2)
    for day, prices in ((d0, [100.0, 101.0]), (d1, [200.0])):
        src = raw_path(p, "aggTrades", "SYM", day)
        src.parent.mkdir(parents=True, exist_ok=True)
        offsets = [0, 1] if day == d0 else [500]
        _trades(day, offsets, prices).write_parquet(src)
        build_day(p, "SYM", day)

    from src.data.to_seconds import bars_path

    second = pl.read_parquet(bars_path(p, "SYM", d1))
    # 2 日目の最初の約定は 500 秒目。それ以前は前日終値 101.0 で埋まっていること。
    assert second["close"][0] == 101.0
    assert second["close"][499] == 101.0
    assert second["close"][500] == 200.0


def test_forward_fill_still_holds_inside_the_day():
    day = date(2025, 1, 1)
    bars = aggtrades_to_seconds(_trades(day, [0, 10], [100.0, 110.0]), day)
    assert bars["close"][5] == 100.0     # 直前の約定価格（過去）で埋まる
    assert bars["is_filled"][5]
    assert bars["close"][10] == 110.0
