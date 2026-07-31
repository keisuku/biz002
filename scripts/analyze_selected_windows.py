"""Run refs, ride, and impulse-scan on pre-registered disjoint windows.

The standard CLI accepts one continuous date range.  Calling it once per
window would overwrite reports, while pretending all selected dates are one
continuous range would introduce false rolling context across multi-month
gaps.  This runner evaluates each contiguous segment independently and then
aggregates the resulting trades.
"""

from __future__ import annotations

import argparse
import json

import polars as pl

from src.analysis import impulse, trend_ride
from src.analysis.regimes import contiguous_ranges
from src.config import load_params, resolve_dir
from src import pipeline


def _add_selection_group(
    frame: pl.DataFrame, day_groups: pl.DataFrame
) -> pl.DataFrame:
    return (
        frame.with_columns(
            pl.from_epoch(pl.col("impulse_ts"), time_unit="s")
            .dt.date()
            .alias("date")
        )
        .join(day_groups, on="date", how="inner")
    )


def _selection_strata(rides: pl.DataFrame, params) -> pl.DataFrame:
    if rides.height == 0:
        return pl.DataFrame()
    edges = [float(x) for x in params.get_path("trend_ride.trend_z_buckets")]
    enriched = rides.with_columns(
        pl.when(pl.col("trend_align").is_null())
        .then(pl.lit("unknown"))
        .when(pl.col("trend_align"))
        .then(pl.lit("with_trend"))
        .otherwise(pl.lit("counter_trend"))
        .alias("align"),
        pl.when(pl.col("vol_z").is_null())
        .then(pl.lit("unknown"))
        .when(pl.col("vol_z") >= edges[-1])
        .then(pl.lit("vol_extreme"))
        .when(pl.col("vol_z") >= edges[-2])
        .then(pl.lit("vol_high"))
        .otherwise(pl.lit("vol_normal"))
        .alias("vol_bucket"),
        pl.when(pl.col("direction") > 0)
        .then(pl.lit("long"))
        .otherwise(pl.lit("short"))
        .alias("side"),
    )
    signal = ["window_s", "sigma_mult"]
    frames = []
    for cols, name in (
        (["selection_group"], "selection"),
        (["selection_group", "align"], "selection_x_align"),
        (["selection_group", "vol_bucket"], "selection_x_vol"),
        (["selection_group", "side"], "selection_x_side"),
    ):
        summary = trend_ride.summarize(
            enriched, params, by=signal + cols + ["max_hold_s"]
        )
        label_parts: list[pl.Expr] = [pl.lit(f"{name}:")]
        for index, column in enumerate(cols):
            if index:
                label_parts.append(pl.lit("|"))
            label_parts.append(pl.col(column).cast(pl.String))
        frames.append(
            summary.with_columns(
                pl.concat_str(label_parts, separator="").alias("stratum")
            ).drop(cols)
        )
    return pl.concat(frames, how="diagonal_relaxed")


def _segment_map(selected: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for group in sorted(selected["groups"].unique().to_list()):
        group_days = selected.filter(pl.col("groups") == group)["date"].to_list()
        for index, (start, end) in enumerate(
            contiguous_ranges(group_days), start=1
        ):
            for day in pipeline.day_list(start, end):
                rows.append(
                    {
                        "date": day,
                        "selection_group": group,
                        "segment_id": f"{group}_{index:02d}",
                    }
                )
    return pl.DataFrame(rows)


def _segment_robustness(rides: pl.DataFrame) -> pl.DataFrame:
    keys = [
        "selection_group",
        "window_s",
        "sigma_mult",
        "max_hold_s",
    ]
    overall = rides.group_by(keys).agg(
        pl.len().alias("overall_n"),
        pl.col("r_multiple").sum().alias("overall_total_r"),
        pl.col("r_multiple").mean().alias("overall_mean_r"),
    )
    segments = rides.group_by(keys + ["segment_id"]).agg(
        pl.len().alias("segment_n"),
        pl.col("r_multiple").sum().alias("segment_total_r"),
    )
    leave_one_out = segments.join(overall, on=keys).with_columns(
        (
            (pl.col("overall_total_r") - pl.col("segment_total_r"))
            / (pl.col("overall_n") - pl.col("segment_n"))
        ).alias("leave_one_segment_out_mean_r")
    )
    return (
        leave_one_out.group_by(keys)
        .agg(
            pl.len().alias("n_segments"),
            (pl.col("segment_total_r") > 0)
            .sum()
            .alias("positive_segments"),
            pl.col("overall_n").first(),
            pl.col("overall_mean_r").first(),
            pl.col("leave_one_segment_out_mean_r")
            .min()
            .alias("loo_min_mean_r"),
            pl.col("leave_one_segment_out_mean_r")
            .max()
            .alias("loo_max_mean_r"),
        )
        .sort(keys)
    )


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
    day_groups = selected.select(
        "date", pl.col("groups").alias("selection_group")
    )
    days = selected["date"].to_list()
    ranges = contiguous_ranges(days)

    ride_frames = []
    impulse_frames = []
    ref_frames = []
    for start, end in ranges:
        segment_days = pipeline.day_list(start, end)
        refs = pipeline.build_refs(params, symbol, segment_days)
        ref_frames.append(refs)
        rides = pipeline.build_trend_rides(
            params, symbol, segment_days, minute_rv=refs
        )
        if rides.height:
            ride_frames.append(_add_selection_group(rides, day_groups))
        entries = pipeline.build_impulse_scan(params, symbol, segment_days)
        if entries.height:
            impulse_frames.append(_add_selection_group(entries, day_groups))

    if not ride_frames:
        raise SystemExit("no trend rides were generated")
    rides = pl.concat(ride_frames, how="diagonal_relaxed")
    rides = rides.join(
        _segment_map(selected),
        on=["date", "selection_group"],
        how="left",
    )
    incomplete = rides.filter(
        (pl.col("exit_reason") == "max_hold")
        & (pl.col("hold_seconds") < pl.col("max_hold_s"))
    )
    rides = rides.filter(
        ~(
            (pl.col("exit_reason") == "max_hold")
            & (pl.col("hold_seconds") < pl.col("max_hold_s"))
        )
    )

    out = resolve_dir(params, "reports_dir") / symbol
    out.mkdir(parents=True, exist_ok=True)
    rides.write_parquet(out / "trend_rides.parquet")
    summary = trend_ride.summarize(
        rides,
        params,
        by=["window_s", "sigma_mult", "max_hold_s"],
    )
    summary.write_csv(out / "trend_ride_summary.csv")
    strata = pl.concat(
        [
            trend_ride.stratify(rides, params),
            _selection_strata(rides, params),
        ],
        how="diagonal_relaxed",
    ).sort(["stratum", "window_s", "sigma_mult", "max_hold_s"])
    strata.write_csv(out / "trend_ride_strata.csv")
    robustness = _segment_robustness(rides)
    robustness.write_csv(out / "trend_ride_segment_robustness.csv")

    group_verdicts = {}
    for group in sorted(rides["selection_group"].unique().to_list()):
        group_summary = trend_ride.summarize(
            rides.filter(pl.col("selection_group") == group),
            params,
            by=["window_s", "sigma_mult", "max_hold_s"],
        )
        group_verdict = trend_ride.verdict(group_summary, params)
        best_group = group_summary.sort(
            "mean_r", descending=True, nulls_last=True
        ).row(0, named=True)
        group_verdict["best_window_s"] = best_group["window_s"]
        group_verdict["best_sigma_mult"] = best_group["sigma_mult"]
        robust_row = robustness.filter(
            (pl.col("selection_group") == group)
            & (pl.col("window_s") == best_group["window_s"])
            & (pl.col("sigma_mult") == best_group["sigma_mult"])
            & (pl.col("max_hold_s") == best_group["max_hold_s"])
        ).row(0, named=True)
        group_verdict["segment_robustness"] = {
            key: robust_row[key]
            for key in (
                "n_segments",
                "positive_segments",
                "loo_min_mean_r",
                "loo_max_mean_r",
            )
        }
        group_verdicts[group] = group_verdict
    verdict = trend_ride.verdict(summary, params)
    best_overall = summary.sort(
        "mean_r", descending=True, nulls_last=True
    ).row(0, named=True)
    verdict.update(
        {
            "best_window_s": best_overall["window_s"],
            "best_sigma_mult": best_overall["sigma_mult"],
            "period_selection": "reports/period_selection/selection.json",
            "window_slots": 48,
            "unique_days": len(days),
            "incomplete_right_censored_rides_excluded": incomplete.height,
            "by_selection_group": group_verdicts,
            "warning": (
                "window_s / sigma_mult are pre-registered alternatives and are "
                "reported separately. The best row is exploratory, not a "
                "post-selection deployment parameter."
            ),
        }
    )
    (out / "trend_ride_verdict.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    if not impulse_frames:
        raise SystemExit("no impulse entries were generated")
    entries = pl.concat(impulse_frames, how="diagonal_relaxed")
    ceiling = impulse.ceiling_table(entries, params)
    ceiling.write_csv(out / "impulse_ceiling.csv")
    impulse_verdict = impulse.verdict(ceiling)
    (out / "impulse_ceiling_verdict.json").write_text(
        json.dumps(
            impulse_verdict, ensure_ascii=False, indent=2, default=str
        ),
        encoding="utf-8",
    )

    if ref_frames:
        pl.concat(ref_frames).unique(subset=["ts"]).sort("ts").write_parquet(
            pipeline.refs_path(params, symbol)
        )
    print(
        json.dumps(
            {
                "symbol": symbol,
                "rides": rides.height,
                "strata_rows": strata.height,
                "impulse_entries": entries.height,
                "trend_ride_verdict": verdict,
                "impulse_verdict": impulse_verdict,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
