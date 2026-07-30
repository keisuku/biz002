from __future__ import annotations

from datetime import datetime

import polars as pl

from momentum_ignition.data.to_seconds import aggregate_trades_to_seconds


def test_aggregation_separates_aggregate_rows_from_estimated_fills() -> None:
    trades = pl.DataFrame(
        {
            "agg_trade_id": [10, 11, 12],
            "price": [100.0, 101.0, 102.0],
            "quantity": [2.0, 3.0, 1.0],
            "first_trade_id": [20, 23, 25],
            "last_trade_id": [22, 24, 25],
            "transact_time": [
                1_767_225_600_100,
                1_767_225_600_800,
                1_767_225_602_100,
            ],
            "is_buyer_maker": [False, True, False],
        }
    )

    bars = aggregate_trades_to_seconds(trades)

    assert bars.height == 3
    first = bars.row(0, named=True)
    assert first["aggtrade_count"] == 2
    assert first["trade_count_est"] == 5
    assert first["taker_buy_volume"] == 2.0
    assert first["taker_sell_volume"] == 3.0
    assert first["signed_taker_volume"] == -1.0

    gap = bars.row(1, named=True)
    assert gap["ts"] == datetime(2026, 1, 1, 0, 0, 1)
    assert gap["close"] == 101.0
    assert gap["volume"] == 0.0
    assert gap["trade_count_est"] == 0

