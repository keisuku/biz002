"""Executable-entry latency diagnostic for manual notification workflows.

The detector observes a completed k-second return before it can fire. A
reaction delay of zero is therefore already at least k seconds after the
impulse began. This module reports both reaction delay and that minimum impulse
age instead of presenting signal-time fills as instantly executable.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ..config import Params


def simulate(
    events: pl.DataFrame,
    seconds: pl.DataFrame,
    params: Params,
) -> pl.DataFrame:
    if events.height == 0 or seconds.height == 0:
        return pl.DataFrame(
            schema={
                "symbol": pl.String,
                "event_ts": pl.Int64,
                "direction": pl.Int8,
                "reaction_delay_s": pl.Int64,
                "entry_ts": pl.Int64,
                "minimum_impulse_age_s": pl.Int64,
                "mfe_pct": pl.Float64,
                "ret_pct": pl.Float64,
                "net_pct": pl.Float64,
            }
        )
    delays = [int(v) for v in params.get_path("latency.reaction_delay_seconds")]
    horizon = int(params.get_path("latency.diagnostic_horizon_seconds"))
    primary_k = int(params.get_path("features.velocity_primary_k"))
    ts = seconds["ts"].to_numpy()
    if ts[-1] - ts[0] + 1 != len(ts):
        raise ValueError("seconds frame must be a dense 1s grid")
    ts0 = int(ts[0])
    high = seconds["high"].to_numpy()
    low = seconds["low"].to_numpy()
    close = seconds["close"].to_numpy()
    rows: list[dict] = []
    for event in events.iter_rows(named=True):
        direction = int(event["direction"])
        for delay in delays:
            entry_ts = int(event["event_ts"]) + delay
            i = entry_ts - ts0
            end = i + horizon
            if i < 0 or end >= len(ts):
                continue
            entry = float(close[i])
            if direction > 0:
                favourable = (float(np.max(high[i + 1 : end + 1])) - entry) / entry * 100
            else:
                favourable = (entry - float(np.min(low[i + 1 : end + 1]))) / entry * 100
            ret = direction * (float(close[end]) - entry) / entry * 100
            cost = float(event["cost_pct"])
            rows.append(
                {
                    "symbol": event["symbol"],
                    "event_ts": int(event["event_ts"]),
                    "direction": direction,
                    "reaction_delay_s": delay,
                    "entry_ts": entry_ts,
                    "minimum_impulse_age_s": primary_k + delay,
                    "mfe_pct": max(favourable, 0.0),
                    "ret_pct": ret,
                    "net_pct": ret - cost,
                }
            )
    return pl.DataFrame(rows)


def summarize(rows: pl.DataFrame) -> pl.DataFrame:
    if rows.height == 0:
        return pl.DataFrame(
            schema={
                "reaction_delay_s": pl.Int64,
                "n": pl.Int64,
                "minimum_impulse_age_s": pl.Int64,
                "median_mfe_pct": pl.Float64,
                "median_net_pct": pl.Float64,
                "mean_net_pct": pl.Float64,
                "win_rate": pl.Float64,
            }
        )
    return (
        rows.group_by("reaction_delay_s")
        .agg(
            pl.len().alias("n"),
            pl.col("minimum_impulse_age_s").median(),
            pl.col("mfe_pct").median().alias("median_mfe_pct"),
            pl.col("net_pct").median().alias("median_net_pct"),
            pl.col("net_pct").mean().alias("mean_net_pct"),
            (pl.col("net_pct") > 0).mean().alias("win_rate"),
        )
        .sort("reaction_delay_s")
    )
