from __future__ import annotations

import polars as pl

from src.analysis import latency
from tests.conftest import make_bars


def test_latency_uses_delayed_executable_price_and_reports_impulse_age(params):
    bars = make_bars(200, seed=31, start_ts=1_700_000_000).with_columns(
        pl.when(pl.col("ts") >= 1_700_000_100)
        .then(110.0)
        .otherwise(100.0)
        .alias("close")
    ).with_columns(
        pl.col("close").alias("open"),
        pl.col("close").alias("high"),
        pl.col("close").alias("low"),
    )
    events = pl.DataFrame(
        {
            "symbol": ["T"],
            "event_ts": [1_700_000_095],
            "direction": [1],
            "cost_pct": [0.1],
        }
    )
    p = params.with_overrides(
        {
            "latency.reaction_delay_seconds": [0, 5],
            "latency.diagnostic_horizon_seconds": 20,
        }
    )
    result = latency.simulate(events, bars, p).sort("reaction_delay_s")
    assert result["minimum_impulse_age_s"].to_list() == [10, 15]
    assert result["mfe_pct"][0] == 10.0
    assert result["mfe_pct"][1] == 0.0


def test_latency_summary_is_grouped_by_delay(params):
    rows = pl.DataFrame(
        {
            "reaction_delay_s": [0, 0, 10, 10],
            "minimum_impulse_age_s": [10, 10, 20, 20],
            "mfe_pct": [1.0, 3.0, 0.0, 2.0],
            "net_pct": [0.5, -0.5, -1.0, -2.0],
        }
    )
    result = latency.summarize(rows)
    assert result["reaction_delay_s"].to_list() == [0, 10]
    assert result["median_mfe_pct"].to_list() == [2.0, 1.0]
