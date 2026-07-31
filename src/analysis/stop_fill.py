"""§4.5 損失側の実測（省略不可）。

急変の直後は逆指値が想定価格で約定しない。1 秒バーの安値/高値だけで損失を測ると
楽観的になり、期待値の符号が反転しうる。そこで **aggTrades を辿って**、ストップ発動
時刻以降に実際に約定した価格系列から実効約定価格を推定する。

推定モデル:
1. 発動秒以降の約定を時刻順に走査し、最初にストップ価格を跨いだ約定を「発動」とみなす
2. 発動から `exits.stop_latency_ms` の間に約定した価格を数量加重平均する
   （成行転換の執行が完了するまでの実勢価格）
3. 得られた実効価格と、意図したストップ価格との差が「実損の上乗せ分」
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import polars as pl

from ..config import Params
from ..data.download import raw_path


class TickStopFill:
    """生 aggTrades から実効ストップ約定価格を推定する。exits.simulate に渡して使う。"""

    def __init__(self, params: Params, dataset_dir: str = "aggTrades"):
        self.params = params
        self.dataset_dir = dataset_dir
        self.latency_ms = int(params.get_path("exits.stop_latency_ms"))
        self._cache: dict[tuple[str, str], pl.DataFrame | None] = {}

    def _day_frame(self, symbol: str, ts: int) -> pl.DataFrame | None:
        day = datetime.fromtimestamp(ts, tz=timezone.utc).date()
        key = (symbol, day.isoformat())
        if key not in self._cache:
            p = raw_path(self.params, self.dataset_dir, symbol, day)
            self._cache[key] = pl.read_parquet(p) if p.exists() else None
            if len(self._cache) > 4:  # メモリ節約: 直近数日のみ保持
                for k in list(self._cache)[:-4]:
                    self._cache.pop(k, None)
        return self._cache[key]

    def __call__(self, symbol: str, trigger_ts: int, stop_price: float, direction: int) -> float | None:
        df = self._day_frame(symbol, trigger_ts)
        if df is None or df.height == 0:
            return None
        lo = trigger_ts * 1000
        hi = lo + 1000 + self.latency_ms * 4
        w = df.filter((pl.col("ts_ms") >= lo) & (pl.col("ts_ms") <= hi))
        if w.height == 0:
            return None
        px = w["price"].to_numpy()
        qty = w["quantity"].to_numpy()
        tms = w["ts_ms"].to_numpy()
        through = (px <= stop_price) if direction > 0 else (px >= stop_price)
        if not through.any():
            return None
        i = int(np.argmax(through))
        window = tms <= tms[i] + self.latency_ms
        window &= np.arange(len(px)) >= i
        q = qty[window]
        if q.sum() <= 0:
            return float(px[i])
        return float((px[window] * q).sum() / q.sum())


def realized_stop_slippage(sim: pl.DataFrame, events: pl.DataFrame, params: Params) -> pl.DataFrame:
    """ハードストップで抜けた負けトレードについて、意図した幅と実損の差を集計する。"""
    k = float(params.get_path("exits.hard_stop_k"))
    stops = sim.filter(pl.col("exit_reason") == "hard_stop")
    if stops.height == 0:
        return pl.DataFrame(schema={"rule_id": pl.String, "n_stops": pl.Int64})
    ev = events.select("event_ts", "direction", "entry_price", "range_bps_1s")
    j = stops.join(ev, on=["event_ts", "direction"], how="left")
    j = j.with_columns((-k * pl.col("range_bps_1s") / 100.0).alias("intended_loss_pct"))
    j = j.with_columns((pl.col("gross_pct") - pl.col("intended_loss_pct")).alias("stop_slippage_pct"))
    return (
        j.group_by("rule_id")
        .agg(
            pl.len().alias("n_stops"),
            pl.col("intended_loss_pct").mean().alias("intended_loss_mean_pct"),
            pl.col("gross_pct").mean().alias("realized_loss_mean_pct"),
            pl.col("stop_slippage_pct").mean().alias("stop_slippage_mean_pct"),
            pl.col("stop_slippage_pct").quantile(0.10).alias("stop_slippage_p10_pct"),
            pl.col("gross_pct").min().alias("worst_realized_pct"),
        )
        .sort("rule_id")
    )
