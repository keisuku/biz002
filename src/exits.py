"""§4.4 フローベース出口のシミュレーション。

比較する出口ルール:
    E1  1 秒あたり trade_count が発火時ピークの X% を割った瞬間
    E2  ofi の符号が発火方向と逆に反転した瞬間
    E3  直近 N 秒の実現ボラが発火時の X% を割った瞬間
    E4  固定時間決済（§4.2 のベースライン）
    E5  E1 OR E2

**価格ベースのトレーリングストップは実装しない**（価格が下がってから反応するため
構造的に遅れる）。安全弁として全ルールにハードストップ（発火時価格から
−K × 発火時の 1 秒レンジ）と最大保有時間を併設する。

約定価格の仮定:
* エントリ … 発火した秒の終値 + スリッページ（不利方向）
* フロー出口 … その秒の終値 + スリッページ（不利方向）
* ハードストップ … ストップ価格からギャップ分だけ不利側へ滑る。ただしその秒の
  安値/高値より悪くはならない範囲でクリップする（§4.5 の実測で置き換え可能）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import polars as pl

from .config import Params

StopFillEstimator = Callable[[str, int, float, int], float | None]
"""(symbol, trigger_ts, stop_price, direction) -> 実効約定価格 / None"""


@dataclass(frozen=True)
class ExitRule:
    rule_id: str
    kind: str          # e1 | e2 | e3 | e4 | e5
    param: float | int


def build_rules(params: Params, fixed_horizon: int) -> list[ExitRule]:
    ex = params["exits"]
    rules = [ExitRule(f"E1_{int(f*100)}", "e1", f) for f in ex["e1_fractions"]]
    rules += [ExitRule("E2", "e2", 0.0)]
    rules += [ExitRule(f"E3_{int(f*100)}", "e3", f) for f in ex["e3_fractions"]]
    rules += [ExitRule(f"E4_{fixed_horizon}s", "e4", fixed_horizon)]
    rules += [ExitRule(f"E5_{int(f*100)}", "e5", f) for f in ex["e1_fractions"]]
    return rules


def _short_rv(seconds: pl.DataFrame, window: int) -> np.ndarray:
    return (
        seconds.select(
            (pl.col("close").log() - pl.col("close").log().shift(1))
            .rolling_std(window_size=window, min_samples=max(window // 2, 2))
            .alias("rv")
        )["rv"]
        .fill_null(float("nan"))
        .to_numpy()
    )


def simulate(events: pl.DataFrame, seconds: pl.DataFrame, params: Params,
             fixed_horizon: int, stop_fill: StopFillEstimator | None = None,
             cost_multiplier: float = 1.0) -> pl.DataFrame:
    """イベント × 出口ルールの長い表を返す（1 行 = 1 イベント 1 ルール）。"""
    ex = params["exits"]
    max_hold = int(ex["max_hold_seconds"])
    min_hold = int(ex["min_hold_seconds"])
    peak_lb = int(ex["peak_lookback_seconds"])
    k_stop = float(ex["hard_stop_k"])
    e3_win = int(ex["e3_window_seconds"])
    rules = build_rules(params, fixed_horizon)

    if events.height == 0:
        return pl.DataFrame(schema={
            "symbol": pl.String, "event_ts": pl.Int64, "direction": pl.Int8, "rule_id": pl.String,
            "exit_ts": pl.Int64, "hold_seconds": pl.Int64, "exit_reason": pl.String,
            "gross_pct": pl.Float64, "net_pct": pl.Float64,
        })

    ts_arr = seconds["ts"].to_numpy()
    ts0 = int(ts_arr[0])
    n = len(ts_arr)
    high = seconds["high"].to_numpy()
    low = seconds["low"].to_numpy()
    close = seconds["close"].to_numpy()
    tc = seconds["trade_count"].to_numpy().astype(np.float64)
    ofi = seconds["ofi"].to_numpy()
    gap = seconds["gap_max_bps"].to_numpy()
    rv_s = _short_rv(seconds, e3_win)

    ev_ts = events["event_ts"].to_numpy()
    direction = events["direction"].to_numpy().astype(np.int64)
    entry_px = events["entry_price"].to_numpy().astype(np.float64)
    slip_pct = (events["slippage_est_pct"].to_numpy().astype(np.float64)
                if "slippage_est_pct" in events.columns else np.zeros(len(ev_ts)))
    fee_pct = (events["fee_pct"].to_numpy().astype(np.float64)
               if "fee_pct" in events.columns else np.zeros(len(ev_ts)))
    range_bps = events["range_bps_1s"].to_numpy().astype(np.float64)
    symbols = events["symbol"].to_list()

    rows: list[dict] = []
    for j in range(len(ev_ts)):
        i = int(ev_ts[j] - ts0)
        if i < 0 or i + 1 >= n:
            continue
        d = int(direction[j])
        e = float(entry_px[j])
        slip = float(slip_pct[j]) * cost_multiplier
        fee = float(fee_pct[j]) * cost_multiplier
        lo_i = max(i - peak_lb + 1, 0)
        peak_tc = float(tc[lo_i: i + 1].max())
        peak_rv = float(rv_s[i]) if not math.isnan(rv_s[i]) else 0.0
        stop_dist = k_stop * float(range_bps[j]) / 1e4 * e
        stop_px = e - d * stop_dist

        end = min(i + max_hold, n - 1)
        if end <= i:
            continue
        sl = slice(i + 1, end + 1)
        offs = np.arange(1, end - i + 1)
        adverse = (low[sl] <= stop_px) if d > 0 else (high[sl] >= stop_px)
        stop_idx = int(np.argmax(adverse)) if adverse.any() else -1

        for rule in rules:
            cond = _rule_condition(rule, tc[sl], ofi[sl], rv_s[sl], peak_tc, peak_rv, d)
            if rule.kind == "e4":
                h = int(rule.param)
                rule_idx = h - 1 if (i + h) <= end else -1
            else:
                eligible = cond & (offs >= min_hold)
                rule_idx = int(np.argmax(eligible)) if eligible.any() else -1

            reason = "max_hold"
            exit_off = end - i
            fill = float(close[end])
            if stop_idx >= 0 and (rule_idx < 0 or stop_idx <= rule_idx):
                reason = "hard_stop"
                exit_off = stop_idx + 1
                t = i + exit_off
                fill = _stop_fill_price(stop_px, d, float(gap[t]), float(low[t]), float(high[t]))
                if stop_fill is not None:
                    est = stop_fill(symbols[j], int(ts_arr[t]), stop_px, d)
                    if est is not None:
                        fill = float(est)
            elif rule_idx >= 0:
                reason = rule.rule_id
                exit_off = rule_idx + 1
                fill = float(close[i + exit_off])

            gross = d * (fill - e) / e * 100.0
            net = gross - fee - 2.0 * slip
            rows.append({
                "symbol": symbols[j],
                "event_ts": int(ev_ts[j]),
                "direction": d,
                "rule_id": rule.rule_id,
                "exit_ts": int(ts_arr[i + exit_off]),
                "hold_seconds": int(exit_off),
                "exit_reason": reason,
                "gross_pct": gross,
                "net_pct": net,
            })
    return pl.DataFrame(rows)


def _rule_condition(rule: ExitRule, tc: np.ndarray, ofi: np.ndarray, rv: np.ndarray,
                    peak_tc: float, peak_rv: float, d: int) -> np.ndarray:
    if rule.kind == "e1":
        return tc < float(rule.param) * peak_tc
    if rule.kind == "e2":
        return (ofi * d) < 0
    if rule.kind == "e3":
        if peak_rv <= 0:
            return np.zeros(len(rv), dtype=bool)
        return np.nan_to_num(rv, nan=np.inf) < float(rule.param) * peak_rv
    if rule.kind == "e5":
        return (tc < float(rule.param) * peak_tc) | ((ofi * d) < 0)
    return np.zeros(len(tc), dtype=bool)


def _stop_fill_price(stop_px: float, d: int, gap_bps: float, low: float, high: float) -> float:
    """ストップの滑りを、その秒に実在した価格レンジを超えない範囲で見積もる。"""
    slip_px = stop_px * gap_bps / 1e4
    if d > 0:
        return max(low, stop_px - slip_px)
    return min(high, stop_px + slip_px)


def compare_to_baseline(sim: pl.DataFrame, baseline_rule_id: str) -> pl.DataFrame:
    """§4.4「E4（固定時間）に対して有意に勝てない出口ルールは採用しない」の検定。

    同一イベント上での対応のある差を見る（イベント選択の違いによる交絡を避ける）。
    """
    from scipy import stats as _st

    if sim.height == 0 or baseline_rule_id not in sim["rule_id"].unique().to_list():
        return pl.DataFrame(schema={"rule_id": pl.String, "mean_diff_pct": pl.Float64})
    base = sim.filter(pl.col("rule_id") == baseline_rule_id).select(
        "symbol", "event_ts", pl.col("net_pct").alias("base_net")
    )
    rows = []
    for rid in sim["rule_id"].unique().to_list():
        if rid == baseline_rule_id:
            continue
        j = sim.filter(pl.col("rule_id") == rid).join(base, on=["symbol", "event_ts"], how="inner")
        if j.height < 2:
            continue
        diff = (j["net_pct"] - j["base_net"]).to_numpy()
        t, p = _st.ttest_rel(j["net_pct"].to_numpy(), j["base_net"].to_numpy())
        rows.append({
            "rule_id": rid,
            "baseline": baseline_rule_id,
            "n_paired": j.height,
            "mean_diff_pct": float(diff.mean()),
            "t_stat_paired": float(t),
            "p_value": float(p),
            "beats_baseline_p05": bool(p < 0.05 and diff.mean() > 0),
        })
    return pl.DataFrame(rows).sort("mean_diff_pct", descending=True)


def summarize(sim: pl.DataFrame, min_wins: int) -> pl.DataFrame:
    """§4.4 のルール比較表: 勝率・平均損益・最大損失・保有時間分布。"""
    if sim.height == 0:
        return pl.DataFrame(schema={"rule_id": pl.String, "n": pl.Int64})
    return (
        sim.group_by("rule_id")
        .agg(
            pl.len().alias("n"),
            (pl.col("net_pct") > 0).mean().alias("win_rate"),
            (pl.col("net_pct") > 0).sum().alias("n_wins"),
            pl.col("net_pct").mean().alias("mean_net_pct"),
            pl.col("net_pct").median().alias("median_net_pct"),
            pl.col("net_pct").std().alias("sd_net_pct"),
            pl.col("net_pct").min().alias("worst_net_pct"),
            pl.col("net_pct").sum().alias("total_net_pct"),
            pl.col("gross_pct").mean().alias("mean_gross_pct"),
            pl.col("hold_seconds").median().alias("median_hold_s"),
            pl.col("hold_seconds").quantile(0.9).alias("p90_hold_s"),
            (pl.col("exit_reason") == "hard_stop").mean().alias("stop_rate"),
            (pl.col("exit_reason") == "max_hold").mean().alias("timeout_rate"),
        )
        .with_columns(
            (pl.col("mean_net_pct") / pl.col("sd_net_pct") * pl.col("n").sqrt()).alias("t_stat"),
            (pl.col("n_wins") >= min_wins).alias("enough_wins"),
        )
        .sort("mean_net_pct", descending=True)
    )
