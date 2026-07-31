"""§3.3 発火イベントの抽出。

    発火 = is_calm
        AND velocity(k) >= SIGMA_THRESHOLD
        AND tick_ratio  >= TICK_THRESHOLD
        AND ofi_ratio   >= OFI_THRESHOLD
        AND impact_ratio>= IMPACT_THRESHOLD

同一方向の連続発火はクールダウン（既定 60 秒）で 1 イベントに統合する。
データ欠損区間（data_ok=False）は発火対象外。
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .config import Params

CONTEXT_COLUMNS = [
    "close", "volume", "trade_count", "range_bps_1s", "gap_max_bps_w", "gap_mean_bps_ref",
    "rv_calm_prev", "calm_threshold", "tick_count_w", "tick_ref", "tick_ratio",
    "ofi_w", "vol_w", "ofi_ratio", "impact", "impact_ref", "impact_ratio",
    "active_ratio_1h",
]


def fire_mask(feats: pl.DataFrame, params: Params) -> pl.Expr:
    th = params["thresholds"]
    k = int(params.get_path("features.velocity_primary_k"))
    return (
        pl.col("is_calm")
        & pl.col("data_ok")
        & (pl.col(f"velocity_{k}") >= float(th["sigma"]))
        & (pl.col("tick_ratio") >= float(th["tick"]))
        & (pl.col("ofi_ratio") >= float(th["ofi"]))
        & (pl.col("impact_ratio") >= float(th["impact"]))
    ).fill_null(False)


def apply_cooldown(ts: np.ndarray, direction: np.ndarray, cooldown: int, scope: str) -> np.ndarray:
    """クールダウン内の同一（方向 or 銘柄）発火を落とすブールマスクを返す。"""
    keep = np.zeros(len(ts), dtype=bool)
    last: dict[int, int] = {}
    for i in range(len(ts)):
        key = int(direction[i]) if scope == "direction" else 0
        prev = last.get(key)
        if prev is None or ts[i] - prev > cooldown:
            keep[i] = True
            last[key] = int(ts[i])
        else:
            # 統合されたイベントの終端を延ばさない（60 秒の静穏で 1 件と数える定義）
            pass
    return keep


def extract_events(feats: pl.DataFrame, params: Params, symbol: str,
                   region: tuple[int, int] | None = None) -> pl.DataFrame:
    """特徴量フレームから発火イベントを取り出す。

    region=(start_ts, end_ts) を渡すと、その範囲の発火のみを返す（ウォームアップ除外用）。
    クールダウンは region 手前の発火も考慮できるよう、マスク適用前に全体で走らせる。
    """
    k = int(params.get_path("features.velocity_primary_k"))
    ks = list(params.get_path("features.velocity_ks"))
    cooldown = int(params.get_path("events.cooldown_seconds"))
    scope = str(params.get_path("events.cooldown_scope"))

    hits = feats.filter(fire_mask(feats, params)).with_columns(
        pl.when(pl.col(f"ret_{k}s") > 0).then(1).otherwise(-1).cast(pl.Int8).alias("direction")
    )
    hits = hits.filter(pl.col(f"ret_{k}s") != 0)
    if hits.height == 0:
        return _empty_events(params)

    keep = apply_cooldown(hits["ts"].to_numpy(), hits["direction"].to_numpy(), cooldown, scope)
    hits = hits.filter(pl.Series(keep))
    if region is not None:
        hits = hits.filter((pl.col("ts") >= region[0]) & (pl.col("ts") < region[1]))
    if hits.height == 0:
        return _empty_events(params)

    cols = ["ts", "direction"] + CONTEXT_COLUMNS
    for kk in ks:
        cols += [f"ret_{kk}s", f"sigma_ref_{kk}s", f"velocity_{kk}"]
    cols = [c for c in dict.fromkeys(cols) if c in hits.columns]
    out = hits.select(cols).rename({"ts": "event_ts", "close": "entry_price"})
    out = out.with_columns(
        pl.lit(symbol).alias("symbol"),
        (pl.col("ofi_w").sign().cast(pl.Int8) == pl.col("direction")).alias("ofi_dir_agree"),
    )
    return out


def _empty_events(params: Params) -> pl.DataFrame:
    ks = list(params.get_path("features.velocity_ks"))
    schema: dict[str, pl.DataType] = {"event_ts": pl.Int64, "direction": pl.Int8}
    for c in CONTEXT_COLUMNS:
        schema[c if c != "close" else "entry_price"] = pl.Float64
    schema["trade_count"] = pl.Int64
    for kk in ks:
        schema[f"ret_{kk}s"] = pl.Float64
        schema[f"sigma_ref_{kk}s"] = pl.Float64
        schema[f"velocity_{kk}"] = pl.Float64
    schema["symbol"] = pl.String
    schema["ofi_dir_agree"] = pl.Boolean
    return pl.DataFrame(schema=schema)


def add_time_context(events: pl.DataFrame) -> pl.DataFrame:
    """§3.4 のコンテキスト列（時刻・曜日）を付与する。JST = UTC+9。"""
    if events.height == 0:
        return events.with_columns(
            pl.lit(None, dtype=pl.Int8).alias("hour_utc"),
            pl.lit(None, dtype=pl.Int8).alias("hour_jst"),
            pl.lit(None, dtype=pl.Int8).alias("dow_jst"),
            pl.lit(None, dtype=pl.Boolean).alias("is_weekend"),
        )
    dt_utc = pl.from_epoch(pl.col("event_ts"), time_unit="s")
    dt_jst = pl.from_epoch(pl.col("event_ts") + 9 * 3600, time_unit="s")
    return events.with_columns(
        dt_utc.dt.hour().cast(pl.Int8).alias("hour_utc"),
        dt_jst.dt.hour().cast(pl.Int8).alias("hour_jst"),
        dt_jst.dt.weekday().cast(pl.Int8).alias("dow_jst"),  # 1=Mon .. 7=Sun
        (dt_jst.dt.weekday() >= 6).alias("is_weekend"),
    )
