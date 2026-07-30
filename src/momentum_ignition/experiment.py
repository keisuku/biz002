from __future__ import annotations

import argparse
import json
from datetime import date, datetime, time
from pathlib import Path

import polars as pl

from momentum_ignition.config import load_params
from momentum_ignition.data.download import iter_days
from momentum_ignition.features import add_features
from momentum_ignition.hypotheses import generate_signals, signals_frame
from momentum_ignition.outcomes import simulate_outcomes


def load_seconds(symbol: str, start: date, end: date, root: str | Path) -> pl.DataFrame:
    paths = [
        Path(root)
        / symbol.upper()
        / f"{day:%Y}"
        / f"{day:%m}"
        / f"{symbol.upper()}-1s-{day.isoformat()}.parquet"
        for day in iter_days(start, end)
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing one-second files: {missing}")
    bars = pl.concat([pl.read_parquet(path) for path in paths], how="vertical").sort("ts")
    timeline = pl.DataFrame(
        {
            "ts": pl.datetime_range(
                datetime.combine(start, time.min),
                datetime.combine(end, time.max).replace(microsecond=0),
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
        timeline.join(bars, on="ts", how="left")
        .with_columns(pl.col("close").forward_fill().backward_fill())
        .with_columns(
            pl.col("open").fill_null(pl.col("close")),
            pl.col("high").fill_null(pl.col("close")),
            pl.col("low").fill_null(pl.col("close")),
            *[pl.col(column).fill_null(0) for column in zero_columns],
        )
    )


def summarize(outcomes: pl.DataFrame, config: dict) -> dict:
    if outcomes.is_empty():
        return {"status": "no_events", "event_rows": 0, "results": []}
    results: list[dict] = []
    horizons = config["execution"]["outcome_horizons_seconds"]
    for keys, group in outcomes.group_by(["hypothesis", "reaction_delay_s"]):
        hypothesis, delay = keys
        record = {
            "hypothesis": hypothesis,
            "reaction_delay_s": int(delay),
            "rows": group.height,
            "median_entry_age_s": float(group["entry_age_s"].median()),
        }
        for horizon in horizons:
            values = group[f"net_ret_{horizon}s"].drop_nulls()
            mfe = group[f"mfe_{horizon}s"].drop_nulls()
            record[f"mean_net_ret_{horizon}s"] = float(values.mean()) if len(values) else None
            record[f"median_net_ret_{horizon}s"] = float(values.median()) if len(values) else None
            record[f"win_rate_{horizon}s"] = (
                float((values > 0).mean()) if len(values) else None
            )
            record[f"median_mfe_{horizon}s"] = float(mfe.median()) if len(mfe) else None
        results.append(record)
    return {"status": "research_only", "event_rows": outcomes.height, "results": results}


def run_experiment(
    symbol: str,
    start: date,
    end: date,
    seconds_root: str | Path = "data/seconds",
    output_root: str | Path = "reports/phase0",
) -> dict:
    config = load_params()
    bars = load_seconds(symbol, start, end, seconds_root)
    features = add_features(bars, config)
    signals = signals_frame(generate_signals(symbol, features, config))
    outcomes = simulate_outcomes(bars, signals, config)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    signals.write_parquet(output / f"{symbol.upper()}-signals.parquet")
    outcomes.write_parquet(output / f"{symbol.upper()}-latency-outcomes.parquet")
    summary = summarize(outcomes, config)
    summary.update(
        {
            "symbol": symbol.upper(),
            "start": start.isoformat(),
            "end": end.isoformat(),
            "hypotheses": [
                "instant_continuation",
                "pause_then_continue",
                "pullback_reclaim",
            ],
            "warning": "No live trading conclusion may be drawn from Phase 0.",
        }
    )
    (output / f"{symbol.upper()}-summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fixed Phase 0 latency experiment.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start", required=True, type=date.fromisoformat)
    parser.add_argument("--end", required=True, type=date.fromisoformat)
    parser.add_argument("--seconds-root", default="data/seconds")
    parser.add_argument("--output-root", default="reports/phase0")
    args = parser.parse_args()
    print(
        json.dumps(
            run_experiment(
                args.symbol,
                args.start,
                args.end,
                args.seconds_root,
                args.output_root,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
