"""Deterministic market-regime period selection.

The selection rule is fixed before aggTrades outcomes are inspected:

1. rank BTCUSDT daily candles by ``(high - low) / close``;
2. take the top ``n`` candidate dates;
3. sample ``n`` control dates from the remainder with a fixed seed;
4. expand every anchor to a three-day window.

The pure selection function is intentionally separated from data acquisition so
the anti-cherry-picking rule can be tested without network access.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl


def select_periods(
    daily: pl.DataFrame,
    *,
    n: int = 8,
    seed: int = 20260731,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return anchor dates and their expanded three-day window slots."""
    required = {"date", "high", "low", "close"}
    missing = required.difference(daily.columns)
    if missing:
        raise ValueError(f"daily candles missing columns: {sorted(missing)}")
    if daily.height < n * 2:
        raise ValueError(f"need at least {n * 2} daily candles, got {daily.height}")

    ranked = (
        daily.select(
            pl.col("date").cast(pl.Date),
            ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("range_rate"),
        )
        .drop_nulls()
        .filter(pl.col("range_rate").is_finite() & (pl.col("range_rate") >= 0))
        .sort(["range_rate", "date"], descending=[True, False])
    )
    if ranked.height < n * 2:
        raise ValueError(f"need at least {n * 2} valid daily candles, got {ranked.height}")

    extreme = ranked.head(n).with_columns(
        pl.lit("extreme").alias("group"),
        pl.int_range(1, n + 1, eager=True).alias("rank"),
        pl.lit(None, dtype=pl.Int64).alias("draw_order"),
    )
    extreme_dates = set(extreme["date"].to_list())
    remaining = ranked.filter(~pl.col("date").is_in(list(extreme_dates))).sort("date")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(remaining.height, size=n, replace=False)
    control = remaining[chosen.tolist()].with_columns(
        pl.lit("control").alias("group"),
        pl.lit(None, dtype=pl.Int64).alias("rank"),
        pl.Series("draw_order", np.arange(1, n + 1, dtype=np.int64)),
    )

    anchors = pl.concat([extreme, control], how="diagonal_relaxed").select(
        "group", "date", "range_rate", "rank", "draw_order"
    )
    slots: list[dict] = []
    for row in anchors.iter_rows(named=True):
        anchor = row["date"]
        if not isinstance(anchor, date):
            raise TypeError(f"expected date, got {type(anchor)!r}")
        for offset in (-1, 0, 1):
            slots.append(
                {
                    "group": row["group"],
                    "anchor_date": anchor,
                    "anchor_range_rate": row["range_rate"],
                    "offset": offset,
                    "date": anchor + timedelta(days=offset),
                }
            )
    windows = pl.DataFrame(slots).sort(["group", "anchor_date", "offset"])
    return anchors, windows


def contiguous_ranges(days: list[date]) -> list[tuple[date, date]]:
    """Coalesce unique dates into inclusive contiguous ranges."""
    ordered = sorted(set(days))
    if not ordered:
        return []
    out: list[tuple[date, date]] = []
    start = previous = ordered[0]
    for current in ordered[1:]:
        if current != previous + timedelta(days=1):
            out.append((start, previous))
            start = current
        previous = current
    out.append((start, previous))
    return out
