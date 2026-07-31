"""Download BTC daily candles and freeze extreme/control research windows.

Example:
    python -m scripts.select_regime_periods \
      --start 2023-08-01 --end 2026-07-29 --seed 20260731
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl
import requests

from src.analysis.regimes import contiguous_ranges, select_periods


BASE = "https://data.binance.vision/data/futures/um"


def _date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _next_month(day: date) -> date:
    return date(day.year + (day.month == 12), day.month % 12 + 1, 1)


def _fetch_archive(
    session: requests.Session, url: str, filename: str
) -> tuple[pl.DataFrame, dict]:
    archive_response = session.get(url, timeout=60)
    archive_response.raise_for_status()
    checksum_response = session.get(f"{url}.CHECKSUM", timeout=60)
    checksum_response.raise_for_status()
    expected = checksum_response.text.strip().split()[0].lower()
    actual = hashlib.sha256(archive_response.content).hexdigest()
    if actual != expected:
        raise RuntimeError(f"checksum mismatch: {filename}")
    with zipfile.ZipFile(io.BytesIO(archive_response.content)) as archive:
        members = [name for name in archive.namelist() if not name.endswith("/")]
        if len(members) != 1:
            raise RuntimeError(f"unexpected archive members: {filename}: {members}")
        payload = archive.read(members[0])
    frame = pl.read_csv(io.BytesIO(payload), has_header=True).select(
        pl.from_epoch(pl.col("open_time"), time_unit="ms").dt.date().alias("date"),
        pl.col("open").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
    )
    manifest = {
        "archive": filename,
        "url": url,
        "rows": frame.height,
        "checksum": expected,
        "checksum_verified": True,
    }
    return frame, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", type=_date, default=date(2023, 8, 1))
    parser.add_argument("--end", type=_date, default=date(2026, 7, 29))
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--top-n", type=int, default=8)
    args = parser.parse_args()
    if args.end < args.start:
        raise SystemExit("--end must be on or after --start")

    symbol = args.symbol.upper()
    out = Path("reports/period_selection")
    raw = Path("data/raw/klines_1d") / symbol
    out.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "momentum-ignition-research/1.0"

    frames = []
    manifests = []
    cursor = args.start
    while cursor <= args.end:
        month_start = cursor.replace(day=1)
        next_month = _next_month(month_start)
        month_end = next_month - timedelta(days=1)
        complete_month = cursor == month_start and month_end <= args.end
        if complete_month:
            filename = f"{symbol}-1d-{cursor:%Y-%m}.zip"
            url = f"{BASE}/monthly/klines/{symbol}/1d/{filename}"
            frame, manifest = _fetch_archive(session, url, filename)
            frames.append(frame)
            manifests.append(manifest)
            cursor = next_month
            continue
        filename = f"{symbol}-1d-{cursor.isoformat()}.zip"
        url = f"{BASE}/daily/klines/{symbol}/1d/{filename}"
        frame, manifest = _fetch_archive(session, url, filename)
        frames.append(frame)
        manifests.append(manifest)
        cursor += timedelta(days=1)

    daily = (
        pl.concat(frames)
        .unique(subset=["date"], keep="last")
        .sort("date")
        .filter(
            (pl.col("date") >= args.start) & (pl.col("date") <= args.end)
        )
    )
    expected_days = (args.end - args.start).days + 1
    if daily.height != expected_days:
        raise RuntimeError(
            f"daily coverage mismatch: expected={expected_days} actual={daily.height}"
        )
    daily.write_parquet(
        raw / f"{symbol}-1d-{args.start}_{args.end}.parquet"
    )
    pl.DataFrame(manifests).write_csv(out / "klines_1d_download_manifest.csv")

    anchors, windows = select_periods(
        daily, n=args.top_n, seed=args.seed
    )
    anchors.write_csv(out / "anchors.csv")
    windows.write_csv(out / "window_slots.csv")
    unique = (
        windows.group_by("date")
        .agg(
            pl.col("group").unique().sort().alias("groups"),
            pl.col("anchor_date")
            .cast(pl.String)
            .unique()
            .sort()
            .alias("anchors"),
        )
        .with_columns(
            pl.col("groups").list.join("|"),
            pl.col("anchors").list.join("|"),
        )
        .sort("date")
    )
    unique.write_csv(out / "unique_dates.csv")
    unique_days = unique["date"].to_list()
    summary = {
        "source_period": [args.start.isoformat(), args.end.isoformat()],
        "range_definition": "(high-low)/close",
        "top_n": args.top_n,
        "control_seed": args.seed,
        "daily_rows": daily.height,
        "archives": len(manifests),
        "checksums_verified": sum(
            bool(row["checksum_verified"]) for row in manifests
        ),
        "window_slots": windows.height,
        "unique_dates": len(unique_days),
        "contiguous_ranges": [
            [start.isoformat(), end.isoformat()]
            for start, end in contiguous_ranges(unique_days)
        ],
        "anchors": [
            {**row, "date": row["date"].isoformat()}
            for row in anchors.iter_rows(named=True)
        ],
    }
    (out / "selection.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
