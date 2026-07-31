"""§3.4 結果指標（MFE / MAE / ret）と執行コスト推定。

単位の約束（混同すると全ての判定が壊れるので厳守）:
* `mfe_H` / `mae_H` / `ret_H` … **パーセント (%)**。方向調整済み（順行が正）。
* `*_bps` で終わる列 … ベーシスポイント (1bps = 0.01%)。
* `fee_pct` / `slippage_pct` / `cost_pct` … パーセント (%)。

エントリは「発火した秒の終値」で約定すると仮定し、順行/逆行の測定は
**次の秒から**行う。発火した秒自身の高値/安値を成果に含めるのは
「もう見えている値動きで儲かったことにする」ルックアヘッドになるため。
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .config import Params


def _index_map(seconds: pl.DataFrame) -> tuple[np.ndarray, int]:
    ts = seconds["ts"].to_numpy()
    if ts.size and (ts[-1] - ts[0] + 1) != ts.size:
        raise ValueError("seconds frame must be a dense 1s grid")
    return ts, int(ts[0]) if ts.size else 0


def add_outcomes(events: pl.DataFrame, seconds: pl.DataFrame, params: Params) -> pl.DataFrame:
    """各イベントに MFE/MAE/ret を付与する。前方データが足りない分は null。"""
    horizons = list(params.get_path("outcomes.horizons"))
    if events.height == 0:
        cols = []
        for h in horizons:
            cols += [pl.lit(None, dtype=pl.Float64).alias(f"{p}_{h}") for p in ("mfe", "mae", "ret")]
        return events.with_columns(cols) if cols else events

    ts_arr, ts0 = _index_map(seconds)
    high = seconds["high"].to_numpy()
    low = seconds["low"].to_numpy()
    close = seconds["close"].to_numpy()
    n = len(ts_arr)

    ev_ts = events["event_ts"].to_numpy()
    direction = events["direction"].to_numpy().astype(np.int64)
    entry = events["entry_price"].to_numpy().astype(np.float64)
    idx = ev_ts - ts0

    out: dict[str, np.ndarray] = {}
    for h in horizons:
        mfe = np.full(len(ev_ts), np.nan)
        mae = np.full(len(ev_ts), np.nan)
        ret = np.full(len(ev_ts), np.nan)
        for j in range(len(ev_ts)):
            i = int(idx[j])
            end = i + h
            if i < 0 or end >= n:
                continue
            hi = high[i + 1: end + 1].max()
            lo = low[i + 1: end + 1].min()
            e = entry[j]
            d = direction[j]
            if d > 0:
                fav, adv = (hi - e) / e * 100.0, (lo - e) / e * 100.0
            else:
                fav, adv = (e - lo) / e * 100.0, (e - hi) / e * 100.0
            # 定義どおり MFE >= 0, MAE <= 0（一度も順行/逆行しなければ 0）
            mfe[j] = max(fav, 0.0)
            mae[j] = min(adv, 0.0)
            ret[j] = d * (close[end] - e) / e * 100.0
        out[f"mfe_{h}"] = mfe
        out[f"mae_{h}"] = mae
        out[f"ret_{h}"] = ret
    return events.with_columns([pl.Series(k, v) for k, v in out.items()])


def add_costs(events: pl.DataFrame, params: Params, multiplier: float = 1.0) -> pl.DataFrame:
    """執行コスト推定（§4.1 / §6.4）。

    spread_at_entry_bps: bookTicker があればそれを使い、無ければ「凪局面の約定間
    価格ギャップの平均」を代理変数として使う（§2.3）。
    slippage: 半スプレッド + 発火時のギャップ + 発火秒のレンジ、の加重和。
    multiplier はコスト感度分析（1.0 / 1.5 / 2.0 倍）用。
    """
    c = params["costs"]
    s = c["slippage"]
    fee_pct = float(c["taker_fee_bps"]) / 100.0 * int(c["round_trip_sides"]) * float(c["fee_multiplier"])
    # 再計算（コスト感度分析）に備えて既存の出力列は落としておく
    events = events.drop([col for col in
                          ("spread_at_entry_bps", "slippage_est_pct", "fee_pct", "cost_pct")
                          if col in events.columns])

    if events.height == 0:
        return events.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("spread_at_entry_bps"),
            pl.lit(None, dtype=pl.Float64).alias("slippage_est_pct"),
            pl.lit(fee_pct, dtype=pl.Float64).alias("fee_pct"),
            pl.lit(None, dtype=pl.Float64).alias("cost_pct"),
        )

    spread = pl.col("spread_bps") if "spread_bps" in events.columns else pl.col("gap_mean_bps_ref")
    ev = events.with_columns(spread.fill_null(0.0).alias("spread_at_entry_bps"))
    slip_bps = (
        float(s["half_spread_weight"]) * pl.col("spread_at_entry_bps") / 2.0
        + float(s["gap_weight"]) * pl.col("gap_max_bps_w").fill_null(0.0)
        + float(s["impact_weight"]) * pl.col("range_bps_1s").fill_null(0.0)
    )
    slip_bps = pl.max_horizontal(slip_bps, pl.lit(float(s["min_bps"])))
    ev = ev.with_columns((slip_bps / 100.0 * multiplier).alias("slippage_est_pct"))
    # 往復（エントリ + 決済）でスリッページを 2 回払う
    ev = ev.with_columns(
        pl.lit(fee_pct * multiplier).alias("fee_pct"),
    ).with_columns(
        (pl.col("fee_pct") + 2.0 * pl.col("slippage_est_pct")).alias("cost_pct")
    )
    return ev


def net_return(events: pl.DataFrame, ret_col: str) -> pl.Expr:
    return pl.col(ret_col) - pl.col("cost_pct")
