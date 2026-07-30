from __future__ import annotations

import argparse
import io
import json
import zipfile
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import polars as pl

AGGTRADE_COLUMNS = [
    "agg_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "transact_time",
    "is_buyer_maker",
]


@dataclass(frozen=True)
class AggregationQuality:
    symbol: str
    day: str
    aggregate_rows: int
    estimated_trades: int
    output_seconds: int
    first_ts: str
    last_ts: str
    missing_aggregate_id_count: int
    invalid_trade_span_count: int
    output_path: str


def _read_archive(path: str | Path) -> pl.DataFrame:
    archive_path = Path(path)
    with zipfile.ZipFile(archive_path) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(csv_names) != 1:
            raise ValueError(f"Expected one CSV in {archive_path}, got {csv_names}")
        payload = archive.read(csv_names[0])

    first_line = payload.splitlines()[0].decode("utf-8", errors="replace").lower()
    has_header = "price" in first_line and ("quantity" in first_line or "qty" in first_line)
    frame = pl.read_csv(
        io.BytesIO(payload),
        has_header=has_header,
        new_columns=None if has_header else AGGTRADE_COLUMNS,
        infer_schema_length=10_000,
    )
    if has_header:
        aliases = {
            "agg_trade_id": ["agg_trade_id", "aggregate_trade_id"],
            "price": ["price"],
            "quantity": ["quantity", "qty"],
            "first_trade_id": ["first_trade_id"],
            "last_trade_id": ["last_trade_id"],
            "transact_time": ["transact_time", "timestamp"],
            "is_buyer_maker": ["is_buyer_maker", "was_the_buyer_the_maker"],
        }
        rename: dict[str, str] = {}
        lower_to_original = {name.lower().strip(): name for name in frame.columns}
        for target, candidates in aliases.items():
            for candidate in candidates:
                if candidate in lower_to_original:
                    rename[lower_to_original[candidate]] = target
                    break
        frame = frame.rename(rename)
    missing = sorted(set(AGGTRADE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns in {archive_path}: {missing}; got {frame.columns}")
    return frame.select(AGGTRADE_COLUMNS)


def aggregate_trades_to_seconds(trades: pl.DataFrame) -> pl.DataFrame:
    typed = (
        trades.with_columns(
            pl.col("agg_trade_id").cast(pl.Int64),
            pl.col("price").cast(pl.Float64),
            pl.col("quantity").cast(pl.Float64),
            pl.col("first_trade_id").cast(pl.Int64),
            pl.col("last_trade_id").cast(pl.Int64),
            pl.col("transact_time").cast(pl.Int64),
            pl.col("is_buyer_maker").cast(pl.Boolean),
        )
        .with_columns(
            pl.from_epoch("transact_time", time_unit="ms").dt.truncate("1s").alias("ts"),
            (pl.col("price") * pl.col("quantity")).alias("quote_notional"),
            (pl.col("last_trade_id") - pl.col("first_trade_id") + 1).alias("trade_span"),
            pl.when(~pl.col("is_buyer_maker"))
            .then(pl.col("quantity"))
            .otherwise(0.0)
            .alias("taker_buy_volume"),
            pl.when(pl.col("is_buyer_maker"))
            .then(pl.col("quantity"))
            .otherwise(0.0)
            .alias("taker_sell_volume"),
        )
        .sort(["transact_time", "agg_trade_id"])
    )

    bars = typed.group_by("ts", maintain_order=True).agg(
        pl.col("price").first().alias("open"),
        pl.col("price").max().alias("high"),
        pl.col("price").min().alias("low"),
        pl.col("price").last().alias("close"),
        pl.col("quantity").sum().alias("volume"),
        pl.col("quote_notional").sum().alias("quote_volume"),
        pl.len().cast(pl.Int64).alias("aggtrade_count"),
        pl.col("trade_span").sum().cast(pl.Int64).alias("trade_count_est"),
        pl.col("taker_buy_volume").sum(),
        pl.col("taker_sell_volume").sum(),
    )
    bars = bars.with_columns(
        (pl.col("taker_buy_volume") - pl.col("taker_sell_volume")).alias(
            "signed_taker_volume"
        )
    )
    if bars.is_empty():
        return bars

    full_timeline = pl.DataFrame(
        {
            "ts": pl.datetime_range(
                bars["ts"].min(),
                bars["ts"].max(),
                interval="1s",
                eager=True,
            )
        }
    )
    zero_columns = [
        "volume",
        "quote_volume",
        "aggtrade_count",
        "trade_count_est",
        "taker_buy_volume",
        "taker_sell_volume",
        "signed_taker_volume",
    ]
    return (
        full_timeline.join(bars, on="ts", how="left")
        .with_columns(pl.col("close").forward_fill())
        .with_columns(
            pl.col("open").fill_null(pl.col("close")),
            pl.col("high").fill_null(pl.col("close")),
            pl.col("low").fill_null(pl.col("close")),
            *[pl.col(column).fill_null(0) for column in zero_columns],
        )
        .select(
            "ts",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "aggtrade_count",
            "trade_count_est",
            "taker_buy_volume",
            "taker_sell_volume",
            "signed_taker_volume",
        )
    )


def process_archive(
    path: str | Path,
    symbol: str,
    day: date,
    output_root: str | Path = "data/seconds",
) -> AggregationQuality:
    trades = _read_archive(path)
    sorted_ids = trades["agg_trade_id"].cast(pl.Int64).sort()
    id_gaps = sorted_ids.diff().drop_nulls()
    missing_ids = id_gaps.filter(id_gaps > 1)
    invalid_spans = trades.filter(
        pl.col("last_trade_id").cast(pl.Int64) < pl.col("first_trade_id").cast(pl.Int64)
    ).height
    bars = aggregate_trades_to_seconds(trades)

    destination = (
        Path(output_root)
        / symbol.upper()
        / f"{day:%Y}"
        / f"{day:%m}"
        / f"{symbol.upper()}-1s-{day.isoformat()}.parquet"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    bars.write_parquet(destination, compression="zstd", statistics=True)
    return AggregationQuality(
        symbol=symbol.upper(),
        day=day.isoformat(),
        aggregate_rows=trades.height,
        estimated_trades=int(
            trades.select(
                (
                    pl.col("last_trade_id").cast(pl.Int64)
                    - pl.col("first_trade_id").cast(pl.Int64)
                    + 1
                ).sum()
            ).item()
        ),
        output_seconds=bars.height,
        first_ts=str(bars["ts"].min()),
        last_ts=str(bars["ts"].max()),
        missing_aggregate_id_count=int(missing_ids.sum() - missing_ids.len()),
        invalid_trade_span_count=invalid_spans,
        output_path=str(destination),
    )


def locate_archive(root: str | Path, symbol: str, day: date) -> Path:
    return (
        Path(root)
        / "aggTrades"
        / symbol.upper()
        / f"{day:%Y}"
        / f"{day:%m}"
        / f"{symbol.upper()}-aggTrades-{day.isoformat()}.zip"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Binance aggTrades to one-second bars.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start", required=True, type=date.fromisoformat)
    parser.add_argument("--end", required=True, type=date.fromisoformat)
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--output-root", default="data/seconds")
    args = parser.parse_args()

    from momentum_ignition.data.download import iter_days

    reports: list[AggregationQuality] = []
    for day in iter_days(args.start, args.end):
        reports.append(
            process_archive(
                locate_archive(args.raw_root, args.symbol, day),
                args.symbol,
                day,
                args.output_root,
            )
        )
    report_path = Path("reports") / "phase0" / f"{args.symbol.upper()}-aggregation-quality.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps([asdict(report) for report in reports], indent=2),
        encoding="utf-8",
    )
    print(report_path)


if __name__ == "__main__":
    main()
