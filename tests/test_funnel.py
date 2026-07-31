"""発火条件ファネル診断のテスト。

この診断は「どの条件が件数を絞っているか」を読むためのもので、閾値を変更しない。
数え方が間違っていると誤った条件を疑うことになるので、既知の構成で固定する。
"""

from __future__ import annotations

import numpy as np
import polars as pl

from src.analysis import funnel


def _feats(params, n=1000, *, calm=True, velocity=9.0, tick=30.0, ofi=0.9, impact=6.0,
           data_ok=True):
    k = int(params.get_path("features.velocity_primary_k"))
    return pl.DataFrame({
        "ts": np.arange(1_700_000_000, 1_700_000_000 + n, dtype=np.int64),
        "is_calm": np.full(n, calm, dtype=bool),
        "data_ok": np.full(n, data_ok, dtype=bool),
        f"velocity_{k}": np.full(n, velocity),
        "tick_ratio": np.full(n, tick),
        "ofi_ratio": np.full(n, ofi),
        "impact_ratio": np.full(n, impact),
    })


def test_all_conditions_passing_counts_every_second(params):
    merged = funnel.merge_chunks([funnel.count_chunk(_feats(params), params)])
    assert merged["n_data_ok"] == 1000
    assert merged["n_all"] == 1000
    table = funnel.funnel_table(merged, params)
    assert set(table["condition"].to_list()) == set(funnel.CONDITIONS)
    # どの条件も絞っていないので、外しても増えない
    assert table["binding_factor"].max() == 1.0


def test_binding_condition_is_identified(params):
    """impact だけが全件を落としている構成で、impact が律速と判定されること。"""
    f = _feats(params, impact=0.5)          # impact 閾値 3.0 を全件が下回る
    merged = funnel.merge_chunks([funnel.count_chunk(f, params)])
    assert merged["n_all"] == 0
    table = funnel.funnel_table(merged, params)
    top = table.row(0, named=True)
    assert top["condition"] == "impact"
    assert top["pass_alone"] == 0
    assert top["pass_if_this_removed"] == 1000    # これを外せば全件通る


def test_two_conditions_failing_are_both_reported(params):
    f = _feats(params, impact=0.5, tick=1.0)
    merged = funnel.merge_chunks([funnel.count_chunk(f, params)])
    table = funnel.funnel_table(merged, params)
    # どちらか一方だけ外しても、もう一方が残るので通過数は 0 のまま
    for row in table.iter_rows(named=True):
        if row["condition"] in ("impact", "tick"):
            assert row["pass_if_this_removed"] == 0
            assert row["pass_alone"] == 0


def test_data_ok_gates_everything(params):
    merged = funnel.merge_chunks([funnel.count_chunk(_feats(params, data_ok=False), params)])
    assert merged["n_data_ok"] == 0
    assert merged["n_all"] == 0


def test_chunks_are_additive(params):
    a = funnel.count_chunk(_feats(params, n=100), params)
    b = funnel.count_chunk(_feats(params, n=250), params)
    merged = funnel.merge_chunks([a, b])
    assert merged["n_data_ok"] == 350
    assert merged["n_all"] == 350
    assert merged["alone"]["impact"] == 350


def test_threshold_percentile_locates_the_threshold_in_the_distribution(params):
    n = 1000
    k = int(params.get_path("features.velocity_primary_k"))
    f = _feats(params, n=n)
    # impact_ratio を 0..10 の一様分布にする。閾値 3.0 は約 30 パーセンタイル。
    f = f.with_columns(pl.Series("impact_ratio", np.linspace(0.0, 10.0, n)))
    merged = funnel.merge_chunks([funnel.count_chunk(f, params)])
    dist = funnel.ratio_distribution(merged, params)
    row = dist.filter(pl.col("ratio") == "impact").row(0, named=True)
    assert abs(row["threshold_percentile"] - 30.0) < 2.0
    assert abs(row["p50"] - 5.0) < 0.1


def test_ratio_distribution_uses_calm_seconds_only(params):
    merged = funnel.merge_chunks([funnel.count_chunk(_feats(params, calm=False), params)])
    assert merged["n_calm_ok"] == 0
    assert funnel.ratio_distribution(merged, params).height == 0
