from __future__ import annotations

from typing import Any

import polars as pl


def simulate_outcomes(
    bars: pl.DataFrame,
    signals: pl.DataFrame,
    config: dict[str, Any],
) -> pl.DataFrame:
    if signals.is_empty():
        return signals
    execution = config["execution"]
    delays = [int(value) for value in execution["decision_to_market_seconds"]]
    horizons = [int(value) for value in execution["outcome_horizons_seconds"]]
    round_trip_cost = 2 * (
        float(execution["taker_fee_bps_per_side"])
        + float(execution["slippage_bps_per_side"])
    ) / 10_000

    bar_rows = bars.sort("ts").to_dicts()
    index_by_ts = {row["ts"]: index for index, row in enumerate(bar_rows)}
    output: list[dict[str, Any]] = []

    for signal in signals.to_dicts():
        signal_index = index_by_ts.get(signal["signal_ts"])
        if signal_index is None:
            continue
        direction = int(signal["direction"])
        for delay in delays:
            # The signal uses the completed signal second. Even delay=0 enters
            # no earlier than the next second's open.
            entry_index = signal_index + delay + 1
            if entry_index >= len(bar_rows):
                continue
            entry = bar_rows[entry_index]
            entry_price = float(entry["open"])
            base = {
                **signal,
                "reaction_delay_s": delay,
                "entry_ts": entry["ts"],
                "signal_age_s": int(
                    (signal["signal_ts"] - signal["impulse_start_ts"]).total_seconds()
                ),
                "entry_age_s": int(
                    (entry["ts"] - signal["impulse_start_ts"]).total_seconds()
                ),
                "entry_price": entry_price,
                "round_trip_cost": round_trip_cost,
            }
            for horizon in horizons:
                end_index = entry_index + horizon
                if end_index >= len(bar_rows):
                    base[f"ret_{horizon}s"] = None
                    base[f"net_ret_{horizon}s"] = None
                    base[f"mfe_{horizon}s"] = None
                    base[f"mae_{horizon}s"] = None
                    continue
                path = bar_rows[entry_index + 1 : end_index + 1]
                exit_price = float(bar_rows[end_index]["close"])
                gross_return = direction * (exit_price / entry_price - 1)
                if direction > 0:
                    mfe = max(float(row["high"]) for row in path) / entry_price - 1
                    mae = min(float(row["low"]) for row in path) / entry_price - 1
                else:
                    mfe = 1 - min(float(row["low"]) for row in path) / entry_price
                    mae = 1 - max(float(row["high"]) for row in path) / entry_price
                base[f"ret_{horizon}s"] = gross_return
                base[f"net_ret_{horizon}s"] = gross_return - round_trip_cost
                base[f"mfe_{horizon}s"] = mfe
                base[f"mae_{horizon}s"] = mae
            output.append(base)
    return pl.DataFrame(output)

