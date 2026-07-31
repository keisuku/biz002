"""Phase 1 → Phase 2 の結合テスト。

既知の正解（合成データに埋め込んだ発火）を使って、パイプライン全体が
「埋め込んだものだけを、正しい向きで」拾うことを確認する。
併せて、チャンク幅（メモリ制御用の内部都合）が結果を変えないことを検査する。
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from src import pipeline as pipe
from src.analysis import viability
from src.data import synthetic
from src.data.download import raw_path
from src.data.to_seconds import build_day


@pytest.fixture(scope="module")
def synth_env(tmp_path_factory):
    """3 日分の合成データを一時ディレクトリに構築する。"""
    from src.config import load_params

    root = tmp_path_factory.mktemp("synth")
    p = load_params().with_overrides({
        "data.raw_dir": str(root / "raw"),
        "data.bars_dir": str(root / "bars"),
        "data.refs_dir": str(root / "refs"),
        "data.events_dir": str(root / "events"),
        "validation.chunk_days": 3,
    })
    days = [date(2025, 1, 1) + timedelta(days=i) for i in range(3)]
    cfg = synthetic.make_config("strong", seed=99)
    truths = []
    for i, d in enumerate(days):
        trades, truth = synthetic.generate_day(d, cfg, seed_offset=i)
        out = raw_path(p, "aggTrades", "SYN", d)
        out.parent.mkdir(parents=True, exist_ok=True)
        trades.write_parquet(out)
        truths.append(truth)
        build_day(p, "SYN", d)
    pipe.build_refs(p, "SYN", days)
    return p, days, pl.concat(truths)


def test_detected_events_correspond_to_injected_ignitions(synth_env):
    params, days, truth = synth_env
    ledger, _ = pipe.build_events(params, "SYN", days, with_exits=False)
    assert ledger.height > 0, "合成データから 1 件も検出できていない"

    t = truth["event_ts"].to_numpy()
    td = truth["direction"].to_numpy()
    matched = 0
    dir_ok = 0
    for ts, d in zip(ledger["event_ts"].to_numpy(), ledger["direction"].to_numpy()):
        j = int(np.argmin(np.abs(t - ts)))
        if abs(int(t[j]) - int(ts)) <= 20:
            matched += 1
            dir_ok += int(td[j] == d)
    # 埋め込んでいない場所で発火してはならない（適合率 100%）
    assert matched == ledger.height
    assert dir_ok == ledger.height


def test_cooldown_holds_across_the_whole_ledger(synth_env):
    params, days, _ = synth_env
    ledger, _ = pipe.build_events(params, "SYN", days, with_exits=False)
    cooldown = int(params.get_path("events.cooldown_seconds"))
    for d in (1, -1):
        ts = ledger.filter(pl.col("direction") == d)["event_ts"].to_numpy()
        if ts.size > 1:
            assert np.diff(np.sort(ts)).min() > cooldown


def test_chunk_width_does_not_change_results(synth_env):
    """チャンク幅は計算の都合であって、台帳の内容を変えてはならない。"""
    params, days, _ = synth_env
    a, _ = pipe.build_events(params, "SYN", days, with_exits=False)
    b, _ = pipe.build_events(params.with_overrides({"validation.chunk_days": 1}),
                             "SYN", days, with_exits=False)
    assert a["event_ts"].to_list() == b["event_ts"].to_list()
    assert a["direction"].to_list() == b["direction"].to_list()
    for col in ("velocity_10", "tick_ratio", "ofi_ratio", "impact_ratio", "mfe_60"):
        assert a[col].to_list() == pytest.approx(b[col].to_list(), rel=1e-12, nan_ok=True)


def test_viability_gate_fails_when_there_is_no_follow_through(tmp_path):
    """継続順行が無い世界では生死判定が FAIL を返すこと（常に PASS を返す壊れ方の検査）。"""
    from src.config import load_params

    p = load_params().with_overrides({
        "data.raw_dir": str(tmp_path / "raw"),
        "data.bars_dir": str(tmp_path / "bars"),
        "data.refs_dir": str(tmp_path / "refs"),
        "validation.chunk_days": 3,
    })
    days = [date(2025, 1, 1) + timedelta(days=i) for i in range(3)]
    cfg = synthetic.make_config("null", seed=5)
    for i, d in enumerate(days):
        trades, _ = synthetic.generate_day(d, cfg, seed_offset=i)
        out = raw_path(p, "aggTrades", "NUL", d)
        out.parent.mkdir(parents=True, exist_ok=True)
        trades.write_parquet(out)
        build_day(p, "NUL", d)
    pipe.build_refs(p, "NUL", days)
    ledger, _ = pipe.build_events(p, "NUL", days, with_exits=False)
    assert ledger.height > 0
    assert viability.viability_check(ledger, p)["verdict"] == "FAIL"
