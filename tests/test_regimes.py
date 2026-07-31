from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from src.analysis.regimes import contiguous_ranges, select_periods


def _daily(n: int = 30) -> pl.DataFrame:
    start = date(2026, 1, 1)
    return pl.DataFrame(
        {
            "date": [start + timedelta(days=i) for i in range(n)],
            "high": [100.0 + i for i in range(n)],
            "low": [100.0 for _ in range(n)],
            "close": [100.0 for _ in range(n)],
        }
    )


def test_period_selection_is_deterministic_and_excludes_extremes_from_controls():
    anchors1, windows1 = select_periods(_daily(), n=4, seed=20260731)
    anchors2, windows2 = select_periods(_daily(), n=4, seed=20260731)
    assert anchors1.equals(anchors2)
    assert windows1.equals(windows2)
    extreme = set(anchors1.filter(pl.col("group") == "extreme")["date"].to_list())
    control = set(anchors1.filter(pl.col("group") == "control")["date"].to_list())
    assert not extreme.intersection(control)
    assert windows1.height == 24


def test_extreme_dates_are_ranked_by_range_rate_before_sampling():
    anchors, _ = select_periods(_daily(), n=3, seed=1)
    extreme = anchors.filter(pl.col("group") == "extreme").sort("rank")
    assert extreme["date"].to_list() == [
        date(2026, 1, 30),
        date(2026, 1, 29),
        date(2026, 1, 28),
    ]


def test_contiguous_ranges_deduplicates_and_coalesces_days():
    assert contiguous_ranges(
        [
            date(2026, 1, 3),
            date(2026, 1, 1),
            date(2026, 1, 2),
            date(2026, 1, 3),
            date(2026, 1, 8),
        ]
    ) == [(date(2026, 1, 1), date(2026, 1, 3)), (date(2026, 1, 8), date(2026, 1, 8))]
