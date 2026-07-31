"""初動エントリの天井測定のテスト。

この測定は**意図的に未来を使う**（初動の開始時刻のラベル付けのみ）。危険なのは
「未来の使用範囲が広がって、いつのまにか実行不可能なバックテストになる」こと。
そこで次の 2 点を機械で固定する:

* ラベルは初動の**最初の window 秒だけ**で決まり、その先の値動きに影響されないこと
  （= 勝ち馬だけを拾っていないこと）
* エントリ価格は t0 + window + delay 秒の**実勢終値**であること
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.analysis import impulse
from tests.conftest import make_bars


def _bars_with_impulse(n=1200, start=600, w=5, size=0.01, tail=0.0, seed=3):
    """start 秒から w 秒かけて size だけ動き、その後 tail だけ継続する系列。

    平坦だと過去のボラがゼロになり参照統計が作れないので、微小ノイズを敷く。
    """
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 1e-5, n)))
    for i in range(w):
        close[start + 1 + i:] *= (1 + size / w)
    if tail:
        for j in range(60):
            close[start + w + 1 + j:] *= (1 + tail / 60)
    b = make_bars(n, seed=seed)
    return b.with_columns(
        pl.Series("close", close), pl.Series("high", close),
        pl.Series("low", close), pl.Series("open", close),
    )


def test_labels_the_first_second_of_the_move_not_the_middle():
    bars = _bars_with_impulse(start=600, w=5, size=0.01)
    starts = impulse.label_impulse_starts(bars, window=5, sigma_mult=4.0,
                                          cooldown=60, ref_seconds=120)
    assert starts.height >= 1
    first = int(starts["ts"][0])
    # 動き出しの秒がラベルされること（候補フラグが立つ 5 秒前ではなく）
    assert first == int(bars["ts"][601])   # close[601] が最初に動く秒
    assert starts["direction"][0] == 1


def test_direction_follows_the_move():
    bars = _bars_with_impulse(start=600, w=5, size=-0.01)
    starts = impulse.label_impulse_starts(bars, window=5, sigma_mult=4.0,
                                          cooldown=60, ref_seconds=120)
    assert starts.height >= 1
    assert starts["direction"][0] == -1


def test_label_does_not_depend_on_what_happens_after_the_window():
    """継続順行の有無でラベルが変わらないこと（勝ち馬だけを選んでいない証拠）。"""
    kw = dict(window=5, sigma_mult=4.0, cooldown=60, ref_seconds=120)
    no_tail = impulse.label_impulse_starts(_bars_with_impulse(tail=0.0), **kw)
    with_tail = impulse.label_impulse_starts(_bars_with_impulse(tail=0.02), **kw)
    against = impulse.label_impulse_starts(_bars_with_impulse(tail=-0.02), **kw)
    assert no_tail["ts"].to_list() == with_tail["ts"].to_list() == against["ts"].to_list()
    assert no_tail["direction"].to_list() == against["direction"].to_list()


def test_entry_price_is_the_close_at_window_plus_delay(params):
    bars = _bars_with_impulse(start=600, w=5, size=0.01, tail=0.0)
    starts = pl.DataFrame({"ts": [int(bars["ts"][600])], "direction": pl.Series([1], dtype=pl.Int8),
                           "impulse_bps": [100.0]})
    e = impulse.oracle_entries(bars, starts, window=5, ages=[0, 7],
                               horizons=[30], costs=None)
    close = bars["close"].to_numpy()
    by_age = {int(r["impulse_age_s"]): r for r in e.iter_rows(named=True)}
    # 初動開始 = index 600。age 秒後の実勢終値で入る
    assert by_age[0]["entry_price"] == pytest.approx(close[600])
    assert by_age[7]["entry_price"] == pytest.approx(close[607])


def test_later_entry_captures_less_of_a_finished_move(params):
    """動きが終わった後に入るほど順行は減る（測定の向きが正しいこと）。"""
    bars = _bars_with_impulse(start=600, w=5, size=0.01, tail=0.0)
    starts = pl.DataFrame({"ts": [int(bars["ts"][600])], "direction": pl.Series([1], dtype=pl.Int8),
                           "impulse_bps": [100.0]})
    e = impulse.oracle_entries(bars, starts, window=5, ages=[0, 10],
                               horizons=[30], costs=None)
    by_age = {int(r["impulse_age_s"]): r for r in e.iter_rows(named=True)}
    # 動きが終わってから入ると順行はほぼ残っていない
    assert by_age[0]["mfe_30"] > by_age[10]["mfe_30"]


def test_ceiling_verdict_fails_when_move_is_smaller_than_cost(params):
    """コストに届かない天井では CEILING_BELOW_BAR を返すこと。"""
    entries = pl.DataFrame({
        "window_s": [5, 5], "impulse_age_s": [0, 7],
        "impulse_bps": [10.0, 10.0],
        "mfe_60": [0.01, 0.005],     # 1.0 bps / 0.5 bps
        "ret_60": [0.01, 0.005],
        "cost_pct": [0.14, 0.14],    # 14 bps
    })
    table = impulse.ceiling_table(entries, params)
    v = impulse.verdict(table)
    assert v["verdict"] == "CEILING_BELOW_BAR"
    assert v["best_mfe_cost_multiple"] < 3.0


def test_ceiling_verdict_passes_when_move_clears_the_bar(params):
    entries = pl.DataFrame({
        "window_s": [5], "impulse_age_s": [0],
        "impulse_bps": [100.0],
        "mfe_60": [0.60], "ret_60": [0.50], "cost_pct": [0.14],
    })
    v = impulse.verdict(impulse.ceiling_table(entries, params))
    assert v["verdict"] == "CEILING_CLEARS_BAR"


def test_cooldown_merges_one_move_into_one_impulse():
    bars = _bars_with_impulse(start=600, w=5, size=0.02)
    starts = impulse.label_impulse_starts(bars, window=5, sigma_mult=4.0,
                                          cooldown=60, ref_seconds=120)
    ts = np.sort(starts["ts"].to_numpy())
    if ts.size > 1:
        assert np.diff(ts).min() > 60


def test_verdict_flags_negative_net_even_when_mfe_bar_is_cleared(params):
    """MFE 基準を超えても純リターンが負なら、そう明示すること。

    MFE は「最良の瞬間に降りられたら取れた幅」なので、基準を超えていても
    実際の決済が負ということは普通に起きる。ここを取り違えると
    「天井を超えた=勝てる」と誤読する。
    """
    entries = pl.DataFrame({
        "window_s": [5, 5], "impulse_age_s": [0, 0],
        "impulse_bps": [50.0, 50.0],
        "mfe_60": [0.90, 0.90],      # 90 bps: コスト 20bps の 4.5 倍 → MFE 基準は通る
        "ret_60": [-0.10, -0.10],    # しかし決済は負
        "cost_pct": [0.20, 0.20],
    })
    v = impulse.verdict(impulse.ceiling_table(entries, params))
    assert v["verdict"] == "CEILING_CLEARS_BAR"
    assert v["net_positive_anywhere"] is False
    assert "負" in v["net_note"]
