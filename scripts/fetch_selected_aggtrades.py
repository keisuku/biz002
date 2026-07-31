"""Download only the pre-registered regime-window aggTrades archives.

Usage:
    python -m scripts.fetch_selected_aggtrades --symbol BTCUSDT
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import polars as pl

from src.config import load_params, resolve_dir
from src.data import download as dl
from src.analysis.regimes import contiguous_ranges


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    parser.add_argument(
        "--selection",
        default="reports/period_selection/unique_dates.csv",
    )
    args = parser.parse_args()

    symbol = args.symbol.upper()
    params = load_params()
    selected = pl.read_csv(args.selection, try_parse_dates=True)
    days = selected["date"].to_list()
    ranges = contiguous_ranges(days)
    layout = dl.discover_layout(params, symbol)
    dataset = dl.resolve_dataset_name(layout, "aggTrades")
    workers = int(params.get_path("data.max_download_workers"))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        frames = list(
            pool.map(
                lambda day: dl.fetch_range(
                    params, layout, dataset, [symbol], day, day
                ),
                days,
            )
        )
    manifest = pl.concat(frames).sort("date")
    out = resolve_dir(params, "reports_dir") / symbol
    out.mkdir(parents=True, exist_ok=True)
    manifest.write_csv(out / "download_aggTrades_selected.csv")
    summary = {
        "symbol": symbol,
        "window_slots": 48,
        "unique_dates": len(days),
        "ranges": [[start.isoformat(), end.isoformat()] for start, end in ranges],
        "ok": int((manifest["status"] == "ok").sum()),
        "cached": int((manifest["status"] == "cached").sum()),
        "missing": int((manifest["status"] == "missing").sum()),
        "checksums_verified": int(manifest["checksum_verified"].sum()),
        "all_checksums_verified": bool(manifest["checksum_verified"].all()),
    }
    (out / "download_aggTrades_selected.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
