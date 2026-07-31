"""Rebuild one-second bars for the pre-registered regime windows."""

from __future__ import annotations

import argparse
import json

import polars as pl

from src.config import load_params, resolve_dir
from src.data import to_seconds, verify_flags


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
    days = pl.read_csv(args.selection, try_parse_dates=True)["date"].to_list()
    rows = [
        to_seconds.build_day(params, symbol, day, overwrite=True)
        for day in days
    ]
    report = pl.DataFrame(rows).sort("date")
    out = resolve_dir(params, "reports_dir") / symbol
    out.mkdir(parents=True, exist_ok=True)
    report.write_csv(out / "bars_build_report.csv")
    summary = {
        "symbol": symbol,
        "overwrite": True,
        "days": report.height,
        "days_ok": int((report["status"] == "ok").sum()),
        "days_missing_raw": int((report["status"] == "missing_raw").sum()),
        "missing_dates": report.filter(
            pl.col("status") == "missing_raw"
        )["date"].to_list(),
        "days_with_long_gaps": int((report["long_gaps"] > 0).sum()),
        "total_gap_seconds": int(report["gap_seconds"].sum()),
    }
    (out / "bars_missing_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    frames = [
        pl.read_parquet(to_seconds.bars_path(params, symbol, day))
        for day in days
        if to_seconds.bars_path(params, symbol, day).exists()
    ]
    flag_result = verify_flags.check_bars(pl.concat(frames).sort("ts"))
    flag_result["symbol"] = symbol
    flag_result["days"] = len(frames)
    verify_flags.write_report(flag_result, out)
    print(
        json.dumps(
            {"bars": summary, "verify_flags": flag_result},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
