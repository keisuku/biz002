from __future__ import annotations

import math
from typing import Any

import polars as pl

EPSILON = 1e-12


def _safe_ratio(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    return (
        pl.when(denominator.abs() > EPSILON)
        .then(numerator / denominator)
        .otherwise(None)
    )


def add_features(bars: pl.DataFrame, config: dict[str, Any]) -> pl.DataFrame:
    """Add strictly backward-looking features to a one-second bar frame.

    The returned value for a timestamp is invariant to appending future rows.
    References for a k-second trigger are shifted by k seconds, preventing the
    trigger shock from inflating its own scale estimate.
    """
    feature_config = config["features"]
    calm_window = int(feature_config["calm_window_seconds"])
    calm_ref_window = int(feature_config["calm_reference_seconds"])
    calm_percentile = float(feature_config["calm_percentile"])
    sigma_window = int(feature_config["sigma_reference_seconds"])
    velocity_windows = [int(value) for value in feature_config["velocity_windows_seconds"]]
    tick_window = int(feature_config["tick_window_seconds"])
    flow_window = int(feature_config["flow_window_seconds"])
    impact_window = int(feature_config["impact_window_seconds"])
    min_reference = int(feature_config.get("minimum_reference_seconds", 900))

    result = (
        bars.sort("ts")
        .with_columns((pl.col("close").log() - pl.col("close").shift(1).log()).alias("log_ret_1s"))
        .with_columns(
            (
                pl.col("log_ret_1s")
                .shift(1)
                .rolling_std(window_size=calm_window, min_samples=min(calm_window, min_reference))
                * math.sqrt(calm_window)
            ).alias(f"rv_prev_{calm_window}")
        )
        .with_columns(
            pl.col(f"rv_prev_{calm_window}")
            .shift(1)
            .rolling_quantile(
                quantile=calm_percentile,
                window_size=calm_ref_window,
                min_samples=min(calm_ref_window, min_reference),
            )
            .alias("calm_threshold")
        )
        .with_columns(
            (pl.col(f"rv_prev_{calm_window}") <= pl.col("calm_threshold")).alias("is_calm")
        )
    )

    velocity_expressions: list[pl.Expr] = []
    for window in velocity_windows:
        ret_name = f"log_ret_{window}s"
        sigma_name = f"sigma_ref_{window}s"
        result = result.with_columns(
            (pl.col("close").log() - pl.col("close").shift(window).log()).alias(ret_name)
        ).with_columns(
            pl.col(ret_name)
            .shift(window)
            .rolling_std(window_size=sigma_window, min_samples=min(sigma_window, min_reference))
            .alias(sigma_name)
        )
        velocity_expressions.append(
            _safe_ratio(pl.col(ret_name).abs(), pl.col(sigma_name)).alias(
                f"velocity_{window}s"
            )
        )
    result = result.with_columns(*velocity_expressions)
    result = result.with_columns(
        *[
            pl.col("is_calm").shift(window).alias(f"is_calm_pre_{window}s")
            for window in velocity_windows
        ]
    )

    result = result.with_columns(
        pl.col("trade_count_est").rolling_sum(tick_window).alias("trade_count_10s"),
        pl.col("aggtrade_count").rolling_sum(tick_window).alias("aggtrade_count_10s"),
        pl.col("volume").rolling_sum(flow_window).alias("volume_10s"),
        pl.col("signed_taker_volume").rolling_sum(flow_window).alias("signed_flow_10s"),
        (
            pl.col("close").log() - pl.col("close").shift(impact_window).log()
        ).abs().alias("absolute_return_10s"),
    ).with_columns(
        pl.col("trade_count_10s")
        .shift(tick_window)
        .rolling_median(window_size=sigma_window, min_samples=min(sigma_window, min_reference))
        .alias("trade_count_reference"),
        _safe_ratio(pl.col("signed_flow_10s").abs(), pl.col("volume_10s")).alias(
            "flow_ratio"
        ),
        _safe_ratio(pl.col("signed_flow_10s"), pl.col("volume_10s")).alias(
            "signed_flow_ratio"
        ),
        _safe_ratio(pl.col("absolute_return_10s"), pl.col("volume_10s")).alias("impact"),
    ).with_columns(
        _safe_ratio(pl.col("trade_count_10s"), pl.col("trade_count_reference")).alias(
            "tick_ratio"
        ),
        pl.col("impact")
        .shift(impact_window)
        .rolling_median(window_size=sigma_window, min_samples=min(sigma_window, min_reference))
        .alias("impact_reference"),
    ).with_columns(
        _safe_ratio(pl.col("impact"), pl.col("impact_reference")).alias("impact_ratio")
    )
    return result


def feature_columns(config: dict[str, Any]) -> list[str]:
    velocity_windows = config["features"]["velocity_windows_seconds"]
    return [
        "is_calm",
        *[f"is_calm_pre_{window}s" for window in velocity_windows],
        "calm_threshold",
        f"rv_prev_{config['features']['calm_window_seconds']}",
        *[f"log_ret_{window}s" for window in velocity_windows],
        *[f"sigma_ref_{window}s" for window in velocity_windows],
        *[f"velocity_{window}s" for window in velocity_windows],
        "trade_count_10s",
        "aggtrade_count_10s",
        "trade_count_reference",
        "tick_ratio",
        "volume_10s",
        "signed_flow_10s",
        "flow_ratio",
        "signed_flow_ratio",
        "impact",
        "impact_reference",
        "impact_ratio",
    ]
