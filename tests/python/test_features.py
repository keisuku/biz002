from __future__ import annotations

from datetime import timedelta

import polars as pl

from momentum_ignition.features import add_features, feature_columns


def test_features_do_not_change_when_future_rows_are_appended(
    calm_then_impulse_bars: pl.DataFrame,
    compact_config: dict,
) -> None:
    cutoff = calm_then_impulse_bars["ts"][420]
    prefix = calm_then_impulse_bars.filter(pl.col("ts") <= cutoff)

    prefix_result = add_features(prefix, compact_config).filter(pl.col("ts") == cutoff)
    full_result = add_features(calm_then_impulse_bars, compact_config).filter(
        pl.col("ts") == cutoff
    )

    for column in feature_columns(compact_config):
        assert prefix_result[column].to_list() == full_result[column].to_list(), column


def test_sigma_reference_excludes_trigger_window(
    calm_then_impulse_bars: pl.DataFrame,
    compact_config: dict,
) -> None:
    features = add_features(calm_then_impulse_bars, compact_config)
    trigger_ts = calm_then_impulse_bars["ts"][449]
    earlier = trigger_ts - timedelta(seconds=10)
    # Adding the shock changes the current return, but not a scale reference
    # explicitly shifted by the full trigger window.
    reference_at_trigger = features.filter(pl.col("ts") == trigger_ts)["sigma_ref_10s"][0]
    reference_at_earlier = features.filter(pl.col("ts") == earlier)["sigma_ref_10s"][0]
    assert reference_at_trigger is not None
    assert reference_at_earlier is not None
    assert reference_at_trigger < 0.01

