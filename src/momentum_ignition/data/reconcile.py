from __future__ import annotations

import argparse
import io
import json
import zipfile
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import polars as pl

KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trade_count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]


@dataclass(frozen=True)
class ReconciliationReport:
    symbol: str
    day: str
    compared_minutes: int
    missing_second_minutes: int
    total_volume_relative_error: float
    total_taker_buy_relative_error: float
    total_trade_count_relative_error: float
    maximum_volume_relative_error: float
    maximum_taker_buy_relative_error: float
    maximum_trade_count_absolute_error: int
    p99_volume_relative_error: float
    p99_taker_buy_relative_error: float
    taker_buy_correlation: float
    buyer_maker_orientation_consistent: bool


def read_kline_archive(path: str | Path) -> pl.DataFrame:
    with zipfile.ZipFile(path) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(csv_names) != 1:
            raise ValueError(f"Expected one CSV in {path}")
        payload = archive.read(csv_names[0])
    first_line = payload.splitlines()[0].decode("utf-8", errors="replace").lower()
    has_header = "open_time" in first_line or "open time" in first_line
    frame = pl.read_csv(
        io.BytesIO(payload),
        has_header=has_header,
        new_columns=None if has_header else KLINE_COLUMNS,
        infer_schema_length=2_000,
    )
    if has_header:
        frame.columns = [name.lower().replace(" ", "_") for name in frame.columns]
        if "count" in frame.columns and "trade_count" not in frame.columns:
            frame = frame.rename({"count": "trade_count"})
    return frame.select(KLINE_COLUMNS).with_columns(
        pl.col("open_time").cast(pl.Int64),
        pl.col("volume").cast(pl.Float64),
        pl.col("quote_volume").cast(pl.Float64),
        pl.col("trade_count").cast(pl.Int64),
        pl.col("taker_buy_volume").cast(pl.Float64),
    )


def _relative_error(left: pl.Expr, right: pl.Expr) -> pl.Expr:
    return (
        pl.when(right.abs() > 1e-12)
        .then((left - right).abs() / right.abs())
        .otherwise((left - right).abs())
    )


def reconcile_day(
    symbol: str,
    day: date,
    seconds_path: str | Path,
    kline_path: str | Path,
) -> ReconciliationReport:
    seconds = pl.read_parquet(seconds_path)
    minute_from_seconds = seconds.with_columns(
        pl.col("ts").dt.truncate("1m").alias("minute")
    ).group_by("minute").agg(
        pl.col("volume").sum().alias("seconds_volume"),
        pl.col("taker_buy_volume").sum().alias("seconds_taker_buy"),
        pl.col("trade_count_est").sum().alias("seconds_trade_count"),
        pl.len().alias("second_rows"),
    )
    klines = read_kline_archive(kline_path).with_columns(
        pl.from_epoch("open_time", time_unit="ms").alias("minute")
    )
    comparison = minute_from_seconds.join(klines, on="minute", how="inner").with_columns(
        _relative_error(pl.col("seconds_volume"), pl.col("volume")).alias("volume_error"),
        _relative_error(
            pl.col("seconds_taker_buy"),
            pl.col("taker_buy_volume"),
        ).alias("taker_buy_error"),
        (pl.col("seconds_trade_count") - pl.col("trade_count"))
        .abs()
        .alias("trade_count_error"),
    )
    maximum_volume_error = float(comparison["volume_error"].max() or 0.0)
    maximum_taker_error = float(comparison["taker_buy_error"].max() or 0.0)
    maximum_count_error = int(comparison["trade_count_error"].max() or 0)
    total_volume_error = abs(
        float(comparison["seconds_volume"].sum()) / float(comparison["volume"].sum()) - 1
    )
    total_taker_error = abs(
        float(comparison["seconds_taker_buy"].sum())
        / float(comparison["taker_buy_volume"].sum())
        - 1
    )
    total_count_error = abs(
        float(comparison["seconds_trade_count"].sum())
        / float(comparison["trade_count"].sum())
        - 1
    )
    taker_correlation = float(
        comparison.select(
            pl.corr("seconds_taker_buy", "taker_buy_volume")
        ).item()
        or 0.0
    )
    inverted_taker = comparison["seconds_volume"] - comparison["seconds_taker_buy"]
    actual_absolute_error = float(
        (comparison["seconds_taker_buy"] - comparison["taker_buy_volume"])
        .abs()
        .sum()
    )
    inverted_absolute_error = float(
        (inverted_taker - comparison["taker_buy_volume"]).abs().sum()
    )
    return ReconciliationReport(
        symbol=symbol.upper(),
        day=day.isoformat(),
        compared_minutes=comparison.height,
        missing_second_minutes=comparison.filter(pl.col("second_rows") != 60).height,
        total_volume_relative_error=total_volume_error,
        total_taker_buy_relative_error=total_taker_error,
        total_trade_count_relative_error=total_count_error,
        maximum_volume_relative_error=maximum_volume_error,
        maximum_taker_buy_relative_error=maximum_taker_error,
        maximum_trade_count_absolute_error=maximum_count_error,
        p99_volume_relative_error=float(comparison["volume_error"].quantile(0.99) or 0.0),
        p99_taker_buy_relative_error=float(
            comparison["taker_buy_error"].quantile(0.99) or 0.0
        ),
        taker_buy_correlation=taker_correlation,
        buyer_maker_orientation_consistent=(
            taker_correlation > 0.99 and actual_absolute_error < inverted_absolute_error
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconcile one-second bars to Binance klines.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start", required=True, type=date.fromisoformat)
    parser.add_argument("--end", required=True, type=date.fromisoformat)
    parser.add_argument("--seconds-root", default="data/seconds")
    parser.add_argument("--raw-root", default="data/raw")
    args = parser.parse_args()

    from momentum_ignition.data.download import iter_days

    reports: list[ReconciliationReport] = []
    for day in iter_days(args.start, args.end):
        seconds_path = (
            Path(args.seconds_root)
            / args.symbol.upper()
            / f"{day:%Y}"
            / f"{day:%m}"
            / f"{args.symbol.upper()}-1s-{day.isoformat()}.parquet"
        )
        kline_path = (
            Path(args.raw_root)
            / "klines"
            / args.symbol.upper()
            / "1m"
            / f"{day:%Y}"
            / f"{day:%m}"
            / f"{args.symbol.upper()}-1m-{day.isoformat()}.zip"
        )
        reports.append(
            reconcile_day(args.symbol, day, seconds_path, kline_path)
        )
    destination = (
        Path("reports")
        / "phase0"
        / f"{args.symbol.upper()}-kline-reconciliation.json"
    )
    destination.write_text(
        json.dumps([asdict(report) for report in reports], indent=2),
        encoding="utf-8",
    )
    print(destination)


if __name__ == "__main__":
    main()
