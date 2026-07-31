"""Aggregate Binance aggTrades into a dense UTC one-second grid."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from ..config import Params, resolve_dir
from .download import raw_path

SECONDS_PER_DAY = 86_400


def day_start_ts(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp())


def bars_path(params: Params, symbol: str, day: date) -> Path:
    return (
        resolve_dir(params, "bars_dir")
        / symbol.upper()
        / f"{day:%Y}"
        / f"{day:%m}"
        / f"{symbol.upper()}-1s-{day.isoformat()}.parquet"
    )


def aggtrades_to_seconds(
    trades: pl.DataFrame, day: date, prev_close: float | None = None
) -> pl.DataFrame:
    """Aggregate one UTC day of trades onto a dense one-second grid.

    `prev_close` is the previous day's final close. Seconds before the day's
    first trade are filled with it, never with a later price: filling those
    seconds backwards would place a future price on a past timestamp, which is
    a lookahead leak into every rolling statistic computed downstream. Without
    a previous close those seconds stay null and are excluded by `data_ok`.
    """
    t0 = day_start_ts(day)
    t1 = t0 + SECONDS_PER_DAY
    selected = (
        trades.filter(
            (pl.col("ts_ms") >= t0 * 1000) & (pl.col("ts_ms") < t1 * 1000)
        )
        .sort(["ts_ms", "agg_trade_id"])
        .with_columns(
            (pl.col("ts_ms") // 1000).cast(pl.Int64).alias("ts"),
            # 日の最初の約定のギャップは、前日終値が分かるならそこから測る。
            # 0 で埋めると日境界だけスリッページ推定が甘くなる。
            (
                (
                    pl.col("price")
                    / (
                        pl.col("price").shift(1)
                        if prev_close is None
                        else pl.col("price").shift(1).fill_null(prev_close)
                    )
                    - 1.0
                ).abs()
                * 1e4
            )
            .fill_null(0.0)
            .alias("_gap_bps"),
            (pl.col("price") * pl.col("quantity")).alias("_quote"),
        )
    )
    if selected.is_empty():
        return _empty_day(t0)

    grouped = (
        selected.group_by("ts", maintain_order=True)
        .agg(
            pl.col("price").first().alias("open"),
            pl.col("price").max().alias("high"),
            pl.col("price").min().alias("low"),
            pl.col("price").last().alias("close"),
            pl.col("quantity").sum().alias("volume"),
            pl.col("_quote").sum().alias("quote_volume"),
            pl.len().cast(pl.Int64).alias("trade_count"),
            pl.when(~pl.col("is_buyer_maker"))
            .then(pl.col("quantity"))
            .otherwise(0.0)
            .sum()
            .alias("taker_buy_volume"),
            pl.when(pl.col("is_buyer_maker"))
            .then(pl.col("quantity"))
            .otherwise(0.0)
            .sum()
            .alias("taker_sell_volume"),
            pl.col("_gap_bps").max().alias("gap_max_bps"),
            pl.col("_gap_bps").mean().alias("gap_mean_bps"),
        )
        .with_columns(
            (pl.col("taker_buy_volume") - pl.col("taker_sell_volume")).alias(
                "ofi"
            )
        )
    )
    timeline = pl.DataFrame(
        {"ts": np.arange(t0, t1, dtype=np.int64)}
    )
    zero = [
        "volume",
        "quote_volume",
        "trade_count",
        "taker_buy_volume",
        "taker_sell_volume",
        "ofi",
        "gap_max_bps",
        "gap_mean_bps",
    ]
    return (
        timeline.join(grouped, on="ts", how="left")
        .with_columns(pl.col("trade_count").is_null().alias("is_filled"))
        .with_columns(
            pl.col("close").forward_fill()
            if prev_close is None
            else pl.col("close").forward_fill().fill_null(prev_close)
        )
        .with_columns(
            pl.col("open").fill_null(pl.col("close")),
            pl.col("high").fill_null(pl.col("close")),
            pl.col("low").fill_null(pl.col("close")),
            *[pl.col(c).fill_null(0) for c in zero],
        )
        .with_columns(pl.col("trade_count").cast(pl.Int64))
    )


def _empty_day(t0: int) -> pl.DataFrame:
    nan = np.full(SECONDS_PER_DAY, np.nan)
    zeros = np.zeros(SECONDS_PER_DAY)
    return pl.DataFrame(
        {
            "ts": np.arange(t0, t0 + SECONDS_PER_DAY, dtype=np.int64),
            "open": nan,
            "high": nan,
            "low": nan,
            "close": nan,
            "volume": zeros,
            "quote_volume": zeros,
            "trade_count": np.zeros(SECONDS_PER_DAY, dtype=np.int64),
            "taker_buy_volume": zeros,
            "taker_sell_volume": zeros,
            "ofi": zeros,
            "gap_max_bps": zeros,
            "gap_mean_bps": zeros,
            "is_filled": np.ones(SECONDS_PER_DAY, dtype=bool),
        }
    )


def gap_report(bars: pl.DataFrame, max_gap: int) -> pl.DataFrame:
    ts = bars["ts"].to_numpy()
    filled = bars["is_filled"].to_numpy()
    rows: list[dict] = []
    start: int | None = None
    for i, flag in enumerate(np.append(filled, False)):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            length = i - start
            if length > max_gap:
                rows.append(
                    {
                        "start_ts": int(ts[start]),
                        "end_ts": int(ts[i - 1]),
                        "seconds": int(length),
                    }
                )
            start = None
    return (
        pl.DataFrame(rows)
        if rows
        else pl.DataFrame(
            schema={"start_ts": pl.Int64, "end_ts": pl.Int64, "seconds": pl.Int64}
        )
    )


def _previous_close(params: Params, symbol: str, day: date) -> float | None:
    """前日の最終終値。日境界の穴埋めを未来ではなく過去の価格で行うために使う。"""
    previous = bars_path(params, symbol, day - timedelta(days=1))
    if not previous.exists():
        return None
    closes = pl.read_parquet(previous, columns=["close"])["close"].drop_nulls()
    return float(closes[-1]) if closes.len() else None


def build_day(
    params: Params, symbol: str, day: date, overwrite: bool = False
) -> dict:
    source = raw_path(params, "aggTrades", symbol, day)
    destination = bars_path(params, symbol, day)
    if destination.exists() and not overwrite:
        bars = pl.read_parquet(destination)
        gaps = gap_report(
            bars, int(params.get_path("bars.max_fill_gap_seconds"))
        )
        return _build_record(day, destination, bars, gaps, "cached")
    if not source.exists():
        return {
            "date": day.isoformat(),
            "status": "missing_raw",
            "rows": 0,
            "long_gaps": 0,
            "gap_seconds": 0,
            "path": str(destination),
        }
    trades = pl.read_parquet(source)
    bars = aggtrades_to_seconds(trades, day, prev_close=_previous_close(params, symbol, day))
    destination.parent.mkdir(parents=True, exist_ok=True)
    bars.write_parquet(destination)
    gaps = gap_report(bars, int(params.get_path("bars.max_fill_gap_seconds")))
    return _build_record(day, destination, bars, gaps, "ok")


def _build_record(
    day: date,
    destination: Path,
    bars: pl.DataFrame,
    gaps: pl.DataFrame,
    status: str,
) -> dict:
    return {
        "date": day.isoformat(),
        "status": status,
        "rows": bars.height,
        "long_gaps": gaps.height,
        "gap_seconds": int(gaps["seconds"].sum()) if gaps.height else 0,
        "path": str(destination),
    }


def _utc_date(epoch_s: int) -> date:
    return datetime.fromtimestamp(epoch_s, tz=UTC).date()


def load_seconds(
    params: Params, symbol: str, start_ts: int, end_ts: int
) -> pl.DataFrame:
    if end_ts <= start_ts:
        return pl.DataFrame()
    first = _utc_date(start_ts)
    last = _utc_date(end_ts - 1)
    frames = []
    day = first
    while day <= last:
        path = bars_path(params, symbol, day)
        if path.exists():
            frames.append(pl.read_parquet(path))
        day += timedelta(days=1)
    if not frames:
        return pl.DataFrame()
    return (
        pl.concat(frames)
        .sort("ts")
        .filter((pl.col("ts") >= start_ts) & (pl.col("ts") < end_ts))
    )

