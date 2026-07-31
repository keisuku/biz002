"""Deterministic synthetic aggTrades used only to test pipeline mechanics.

Synthetic results are never evidence that a trading hypothesis is profitable.
They only prove that known injected shapes are detected in the expected
direction and that a no-follow-through control fails the viability gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl

from .to_seconds import SECONDS_PER_DAY, day_start_ts

PROFILE_DOC = (
    "Synthetic data validates mechanics only; it is not market evidence. "
    "Profiles: realistic, strong, null."
)


@dataclass(frozen=True)
class SyntheticConfig:
    profile: str
    seed: int
    impulse_log_return_per_second: float
    continuation_log_return_per_second: float
    events_per_day: int


PROFILES = {"realistic", "strong", "null"}


def make_config(profile: str, seed: int | None = None) -> SyntheticConfig:
    if profile not in PROFILES:
        raise ValueError(f"unknown synthetic profile: {profile}")
    values = {
        "realistic": (0.00025, 0.00002, 8),
        "strong": (0.0010, 0.0010, 10),
        "null": (0.0010, -0.00015, 8),
    }
    impulse, continuation, count = values[profile]
    return SyntheticConfig(
        profile=profile,
        seed=seed if seed is not None else 17,
        impulse_log_return_per_second=impulse,
        continuation_log_return_per_second=continuation,
        events_per_day=count,
    )


def generate_day(
    day: date, config: SyntheticConfig, seed_offset: int = 0
) -> tuple[pl.DataFrame, pl.DataFrame]:
    rng = np.random.default_rng(config.seed + seed_offset)
    n = SECONDS_PER_DAY
    returns = rng.normal(0.0, 1.5e-6, n)
    event_seconds = np.linspace(
        7_200, n - 7_200, config.events_per_day, dtype=int
    )
    directions = np.where(np.arange(config.events_per_day) % 2 == 0, 1, -1)
    for event_s, direction in zip(event_seconds, directions):
        returns[event_s - 9 : event_s + 1] += (
            direction * config.impulse_log_return_per_second
        )
        returns[event_s + 1 : event_s + 121] += (
            direction * config.continuation_log_return_per_second
        )
    close = 100.0 * np.exp(np.cumsum(returns))
    t0_ms = day_start_ts(day) * 1000

    ts_ms: list[int] = []
    prices: list[float] = []
    quantities: list[float] = []
    makers: list[bool] = []
    impulse_direction: dict[int, int] = {}
    for event_s, direction in zip(event_seconds, directions):
        for second in range(event_s - 9, event_s + 1):
            impulse_direction[int(second)] = int(direction)

    previous = close[0]
    for second in range(n):
        current = float(close[second])
        direction = impulse_direction.get(second)
        if direction is None:
            ts_ms.append(t0_ms + second * 1000 + 500)
            prices.append(current)
            quantities.append(1.0)
            makers.append(bool(rng.integers(0, 2)))
        else:
            for sub in range(21):
                fraction = (sub + 1) / 21
                ts_ms.append(t0_ms + second * 1000 + 20 + sub * 40)
                prices.append(previous + (current - previous) * fraction)
                quantities.append(0.2 if sub else 1.0)
                makers.append(direction < 0)
        previous = current

    size = len(ts_ms)
    trades = pl.DataFrame(
        {
            "agg_trade_id": np.arange(size, dtype=np.int64),
            "ts_ms": np.asarray(ts_ms, dtype=np.int64),
            "price": np.asarray(prices, dtype=np.float64),
            "quantity": np.asarray(quantities, dtype=np.float64),
            "is_buyer_maker": np.asarray(makers, dtype=bool),
            "first_trade_id": np.arange(size, dtype=np.int64),
            "last_trade_id": np.arange(size, dtype=np.int64),
        }
    )
    truth = pl.DataFrame(
        {
            "event_ts": [
                day_start_ts(day) + int(second) for second in event_seconds
            ],
            "direction": directions.astype(np.int8),
        }
    )
    return trades, truth
