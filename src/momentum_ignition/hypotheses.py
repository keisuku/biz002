from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl


@dataclass(frozen=True)
class Signal:
    symbol: str
    hypothesis: str
    impulse_start_ts: object
    signal_ts: object
    direction: int
    impulse_price: float
    signal_price: float
    trigger_velocity: float
    tick_ratio: float
    flow_ratio: float
    impact_ratio: float
    wait_after_base_s: int


def base_impulses(features: pl.DataFrame, config: dict[str, Any]) -> pl.DataFrame:
    trigger = config["trigger"]
    window = int(trigger["velocity_window_seconds"])
    direction = pl.col(f"log_ret_{window}s").sign().cast(pl.Int8)
    condition = (
        pl.col(f"is_calm_pre_{window}s").fill_null(False)
        & (pl.col(f"velocity_{window}s") >= float(trigger["sigma_threshold"])).fill_null(False)
        & (pl.col("tick_ratio") >= float(trigger["tick_threshold"])).fill_null(False)
        & (pl.col("flow_ratio") >= float(trigger["flow_threshold"])).fill_null(False)
        & (pl.col("impact_ratio") >= float(trigger["impact_threshold"])).fill_null(False)
    )
    if trigger.get("require_price_flow_alignment", True):
        condition &= (
            pl.col("signed_flow_ratio").sign() == pl.col(f"log_ret_{window}s").sign()
        ).fill_null(False)
    return (
        features.with_columns(direction.alias("direction"))
        .filter(condition)
        .with_columns(
            (pl.col("ts") - pl.duration(seconds=window - 1)).alias("impulse_start_ts"),
            pl.col("ts").alias("base_signal_ts"),
        )
    )


def _signal_from_row(
    symbol: str,
    hypothesis: str,
    base: dict[str, Any],
    candidate: dict[str, Any],
    wait: int,
) -> Signal:
    return Signal(
        symbol=symbol.upper(),
        hypothesis=hypothesis,
        impulse_start_ts=base["impulse_start_ts"],
        signal_ts=candidate["ts"],
        direction=int(base["direction"]),
        impulse_price=float(base["close"]),
        signal_price=float(candidate["close"]),
        trigger_velocity=float(base["velocity_10s"]),
        tick_ratio=float(base["tick_ratio"]),
        flow_ratio=float(base["flow_ratio"]),
        impact_ratio=float(base["impact_ratio"]),
        wait_after_base_s=wait,
    )


def generate_signals(
    symbol: str,
    features: pl.DataFrame,
    config: dict[str, Any],
) -> list[Signal]:
    """Generate candidates using an online-equivalent, forward scan.

    Candidate scans move forward from a fully observed base impulse. At each
    candidate second they use only rows available at that candidate timestamp.
    """
    bases = base_impulses(features, config)
    if bases.is_empty():
        return []
    rows = features.to_dicts()
    index_by_ts = {row["ts"]: index for index, row in enumerate(rows)}
    hypotheses = config["hypotheses"]
    trigger_window = int(config["trigger"]["velocity_window_seconds"])
    cooldown = int(config["trigger"]["cooldown_seconds"])
    signals: list[Signal] = []
    last_base_index = -cooldown - 1

    for base in bases.to_dicts():
        base_index = index_by_ts[base["ts"]]
        if base_index - last_base_index <= cooldown:
            continue
        last_base_index = base_index
        direction = int(base["direction"])
        impulse_size = abs(float(base[f"log_ret_{trigger_window}s"]))
        if impulse_size <= 0:
            continue

        if hypotheses["instant_continuation"]["enabled"]:
            signals.append(
                _signal_from_row(symbol, "instant_continuation", base, rows[base_index], 0)
            )

        pause = hypotheses["pause_then_continue"]
        if pause["enabled"]:
            end = min(len(rows), base_index + int(pause["search_seconds"]) + 1)
            peak_price = float(base["close"])
            saw_pause = False
            for candidate_index in range(base_index + 1, end):
                candidate = rows[candidate_index]
                price = float(candidate["close"])
                peak_price = max(peak_price, price) if direction > 0 else min(peak_price, price)
                signed_move_1s = (
                    direction
                    * (price - float(rows[candidate_index - 1]["close"]))
                    / float(rows[candidate_index - 1]["close"])
                )
                if abs(signed_move_1s) < impulse_size / 10:
                    saw_pause = True
                giveback = direction * (peak_price - price) / float(base["close"])
                restart_window = int(pause["restart_window_seconds"])
                if candidate_index < base_index + restart_window:
                    continue
                restart_price = float(rows[candidate_index - restart_window]["close"])
                restarted = direction * (price - restart_price) / restart_price > 0
                flow_aligned = (candidate.get("signed_flow_ratio") or 0.0) * direction > 0
                if (
                    saw_pause
                    and restarted
                    and flow_aligned
                    and giveback <= impulse_size * float(pause["maximum_giveback_fraction"])
                ):
                    signals.append(
                        _signal_from_row(
                            symbol,
                            "pause_then_continue",
                            base,
                            candidate,
                            candidate_index - base_index,
                        )
                    )
                    break

        pullback = hypotheses["pullback_reclaim"]
        if pullback["enabled"]:
            end = min(len(rows), base_index + int(pullback["search_seconds"]) + 1)
            extreme_price = float(base["close"])
            saw_valid_retrace = False
            for candidate_index in range(base_index + 1, end):
                candidate = rows[candidate_index]
                price = float(candidate["close"])
                previous_extreme = extreme_price
                extreme_price = (
                    max(extreme_price, price) if direction > 0 else min(extreme_price, price)
                )
                extension = abs(extreme_price - float(base["close"])) / float(base["close"])
                impulse_reference = max(impulse_size, extension)
                retrace = direction * (extreme_price - price) / float(base["close"])
                retrace_fraction = retrace / impulse_reference if impulse_reference > 0 else 0.0
                if (
                    float(pullback["minimum_retrace_fraction"])
                    <= retrace_fraction
                    <= float(pullback["maximum_retrace_fraction"])
                ):
                    saw_valid_retrace = True
                reclaim_window = int(pullback["reclaim_window_seconds"])
                if candidate_index < base_index + reclaim_window:
                    continue
                earlier = float(rows[candidate_index - reclaim_window]["close"])
                reclaiming = direction * (price - earlier) / earlier > 0
                below_extreme = (
                    price <= previous_extreme if direction > 0 else price >= previous_extreme
                )
                if saw_valid_retrace and reclaiming and below_extreme:
                    signals.append(
                        _signal_from_row(
                            symbol,
                            "pullback_reclaim",
                            base,
                            candidate,
                            candidate_index - base_index,
                        )
                    )
                    break
    return signals


def signals_frame(signals: list[Signal]) -> pl.DataFrame:
    if not signals:
        return pl.DataFrame(
            schema={
                "symbol": pl.String,
                "hypothesis": pl.String,
                "impulse_start_ts": pl.Datetime,
                "signal_ts": pl.Datetime,
                "direction": pl.Int8,
                "impulse_price": pl.Float64,
                "signal_price": pl.Float64,
                "trigger_velocity": pl.Float64,
                "tick_ratio": pl.Float64,
                "flow_ratio": pl.Float64,
                "impact_ratio": pl.Float64,
                "wait_after_base_s": pl.Int64,
            }
        )
    return pl.DataFrame([signal.__dict__ for signal in signals])
