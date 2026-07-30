from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from momentum_ignition.config import load_params


@pytest.fixture
def compact_config() -> dict:
    config = load_params()
    config["features"].update(
        {
            "calm_window_seconds": 30,
            "calm_reference_seconds": 180,
            "calm_percentile": 0.90,
            "sigma_reference_seconds": 120,
            "minimum_reference_seconds": 20,
        }
    )
    config["trigger"].update(
        {
            "sigma_threshold": 2.0,
            "tick_threshold": 2.0,
            "flow_threshold": 0.5,
            "impact_threshold": 1.5,
        }
    )
    return config


@pytest.fixture
def calm_then_impulse_bars() -> pl.DataFrame:
    start = datetime(2026, 1, 1)
    rows = []
    price = 100.0
    for second in range(500):
        if second < 440:
            calm_returns = [0.000010, -0.000018, 0.000006, 0.000014, -0.000009]
            price *= 1 + calm_returns[second % len(calm_returns)]
            volume = 1.0
            trades = 2
            signed = 0.1 if second % 2 == 0 else -0.1
        elif second < 450:
            price *= 1.0015
            volume = 12.0
            trades = 40
            signed = 10.0
        elif second < 453:
            price *= 1.00002
            volume = 4.0
            trades = 8
            signed = 1.0
        else:
            price *= 1.0005
            volume = 8.0
            trades = 20
            signed = 6.0
        rows.append(
            {
                "ts": start + timedelta(seconds=second),
                "open": price,
                "high": price * 1.0001,
                "low": price * 0.9999,
                "close": price,
                "volume": volume,
                "quote_volume": volume * price,
                "aggtrade_count": max(1, trades // 2),
                "trade_count_est": trades,
                "taker_buy_volume": (volume + signed) / 2,
                "taker_sell_volume": (volume - signed) / 2,
                "signed_taker_volume": signed,
            }
        )
    return pl.DataFrame(rows)
