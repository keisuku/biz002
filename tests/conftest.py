from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_params  # noqa: E402


@pytest.fixture()
def params():
    """窓幅を小さくした試験用パラメータ（意味論は同じ、計算量だけ小さい）。"""
    p = load_params()
    return p.with_overrides({
        "features.velocity_ks": [5, 10],
        "features.velocity_primary_k": 10,
        "features.sigma_ref_seconds": 120,
        "features.tick_ref_seconds": 120,
        "features.impact_ref_seconds": 120,
        "features.tick_window_seconds": 10,
        "features.ofi_window_seconds": 10,
        "features.impact_window_seconds": 10,
        "features.calm_ref_days": 1,
        "features.calm_ref_sample_seconds": 10,
        "thresholds.calm_window": 60,
        "bars.max_fill_gap_seconds": 30,
        "outcomes.horizons": [15, 30, 60],
        "exits.max_hold_seconds": 60,
        "exits.min_hold_seconds": 2,
        "exits.peak_lookback_seconds": 10,
    })


def make_bars(n: int = 900, seed: int = 0, start_ts: int = 1_700_000_000) -> pl.DataFrame:
    """密な 1 秒バーを合成する（テスト用の素材）。"""
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 3e-5, n)
    close = 100.0 * np.exp(np.cumsum(r))
    tc = rng.poisson(5, n).astype(np.int64)
    vol = rng.gamma(2.0, 0.5, n)
    buy = vol * rng.uniform(0.2, 0.8, n)
    return pl.DataFrame({
        "ts": np.arange(start_ts, start_ts + n, dtype=np.int64),
        "open": close, "high": close * 1.0001, "low": close * 0.9999, "close": close,
        "volume": vol, "quote_volume": vol * close, "trade_count": tc,
        "taker_buy_volume": buy, "taker_sell_volume": vol - buy,
        "ofi": buy - (vol - buy),
        "gap_max_bps": np.abs(r) * 1e4, "gap_mean_bps": np.abs(r) * 0.5e4,
        "is_filled": np.zeros(n, dtype=bool),
    })
