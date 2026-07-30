from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from momentum_ignition.outcomes import simulate_outcomes


def test_execution_occurs_after_completed_signal_second(compact_config: dict) -> None:
    start = datetime(2026, 1, 1)
    bars = pl.DataFrame(
        [
            {
                "ts": start + timedelta(seconds=index),
                "open": 100.0 + index,
                "high": 100.5 + index,
                "low": 99.5 + index,
                "close": 100.0 + index,
            }
            for index in range(40)
        ]
    )
    signals = pl.DataFrame(
        [
            {
                "symbol": "BTCUSDT",
                "hypothesis": "instant_continuation",
                "impulse_start_ts": start,
                "signal_ts": start + timedelta(seconds=9),
                "direction": 1,
                "impulse_price": 109.0,
                "signal_price": 109.0,
                "trigger_velocity": 4.0,
                "tick_ratio": 10.0,
                "flow_ratio": 0.8,
                "impact_ratio": 3.0,
                "wait_after_base_s": 0,
            }
        ]
    )
    compact_config["execution"]["decision_to_market_seconds"] = [0, 10]
    compact_config["execution"]["outcome_horizons_seconds"] = [15]

    outcomes = simulate_outcomes(bars, signals, compact_config).sort("reaction_delay_s")
    immediate = outcomes.row(0, named=True)
    delayed = outcomes.row(1, named=True)

    assert immediate["entry_ts"] == start + timedelta(seconds=10)
    assert immediate["entry_age_s"] == 10
    assert delayed["entry_ts"] == start + timedelta(seconds=20)
    assert delayed["entry_age_s"] == 20
    assert immediate["entry_price"] == 110.0
    assert delayed["entry_price"] == 120.0

