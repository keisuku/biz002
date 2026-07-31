"""§3.2 特徴量。

**すべての特徴量は「発火判定時点までの情報だけ」で計算される。**
実装上の規約:

* 使ってよい時系列操作は `rolling_*`（後ろ向き窓）と `shift(+n)`（過去方向）のみ。
  `shift(-n)`・`reverse()`・`rolling_*(center=True)` は禁止。
* 参照統計（sigma_ref / tick_ref / impact_ref / 凪分位点）は、発火そのものを
  含まないように必ず窓の分だけ `shift` する。含めると発火が自分の基準値を押し上げ、
  閾値が過小評価される（=見かけ上イベントが減り、質も歪む）。
* この規約が守られていることは tests/test_lookahead.py が全特徴量に対して機械的に検査する。
"""

from __future__ import annotations

import math

import polars as pl

from .config import Params

# 生成される特徴量列（ルックアヘッド検査はこの一覧を総なめする）
FEATURE_COLUMNS = [
    "r1",
    "rv_calm",
    "rv_calm_prev",
    "calm_threshold",
    "is_calm",
    "tick_count_w",
    "tick_ref",
    "tick_ratio",
    "ofi_w",
    "vol_w",
    "ofi_ratio",
    "impact",
    "impact_ref",
    "impact_ratio",
    "gap_max_bps_w",
    "gap_mean_bps_ref",
    "active_ratio_1h",
    "has_long_gap_1h",
    "range_bps_1s",
    "data_ok",
]


def velocity_cols(ks) -> list[str]:
    out = []
    for k in ks:
        out += [f"ret_{k}s", f"sigma_ref_{k}s", f"velocity_{k}"]
    return out


def _log_close(col: str = "close") -> pl.Expr:
    return pl.col(col).log()


def realized_vol(window: int, min_ratio: float = 1.0) -> pl.Expr:
    """rv(w) = 1 秒対数リターンの標準偏差 × sqrt(w)。"""
    ms = max(int(window * min_ratio), 2)
    return pl.col("r1").rolling_std(window_size=window, min_samples=ms) * math.sqrt(window)


def minute_rv_series(bars: pl.DataFrame, windows, sample_seconds: int) -> pl.DataFrame:
    """凪参照用に rv(w) を分足サンプルへ落とす。

    7 日分の分位点を秒単位の rolling_quantile で出すのは計算量の無駄なので、
    分足にサンプルしてから分位点を取る（推定量として等価）。
    """
    df = bars.select("ts", "close").with_columns(
        (_log_close() - _log_close().shift(1)).alias("r1")
    )
    exprs = [realized_vol(w).alias(f"rv_{w}") for w in windows]
    df = df.with_columns(exprs)
    return df.filter(pl.col("ts") % sample_seconds == 0).select(["ts"] + [f"rv_{w}" for w in windows])


def calm_threshold_series(minute_rv: pl.DataFrame, window: int, percentile: float,
                          ref_days: int, sample_seconds: int) -> pl.DataFrame:
    """過去 ref_days 日の rv(window) 分布の下位 percentile 点を、各サンプル時刻について返す。

    `shift(1)` により、時刻 t の閾値は t より**前**のサンプルのみから作られる。
    """
    n = int(ref_days * 24 * 3600 / sample_seconds)
    min_samples = min(n, max(int(86_400 / sample_seconds), 10))  # 最低 1 日分は必要
    col = f"rv_{window}"
    return minute_rv.select(
        "ts",
        pl.col(col)
        .rolling_quantile(quantile=percentile / 100.0, interpolation="linear",
                          window_size=n, min_samples=min_samples)
        .shift(1)
        .alias("calm_threshold"),
    )


def compute_features(bars: pl.DataFrame, calm_thresholds: pl.DataFrame | None,
                     params: Params) -> pl.DataFrame:
    """密な 1 秒バー（ウォームアップ込み）に §3.2 の特徴量を付与する。"""
    core = compute_core_features(bars, params)
    return add_calm_columns(core, calm_thresholds, int(params.get_path("thresholds.calm_window")),
                            int(params.get_path("features.calm_exclude_seconds")))


def compute_core_features(bars: pl.DataFrame, params: Params) -> pl.DataFrame:
    """凪判定以外の特徴量。凪の窓・分位点を変えても再計算不要な部分。"""
    f = params["features"]
    ks = list(f["velocity_ks"])
    tw = int(f["tick_window_seconds"])
    ow = int(f["ofi_window_seconds"])
    iw = int(f["impact_window_seconds"])
    sigma_ref_n = int(f["sigma_ref_seconds"])
    tick_ref_n = int(f["tick_ref_seconds"])
    impact_ref_n = int(f["impact_ref_seconds"])
    max_gap = int(params.get_path("bars.max_fill_gap_seconds"))
    min_active = float(params.get_path("bars.min_active_second_ratio"))

    df = bars.sort("ts").with_columns(
        (_log_close() - _log_close().shift(1)).alias("r1"),
        ((pl.col("high") - pl.col("low")) / pl.col("close") * 1e4).alias("range_bps_1s"),
    )

    # --- 速度 σ ---------------------------------------------------------------------
    vel_exprs: list[pl.Expr] = []
    for k in ks:
        vel_exprs.append((_log_close() - _log_close().shift(k)).alias(f"ret_{k}s"))
    df = df.with_columns(vel_exprs)
    ref_exprs: list[pl.Expr] = []
    for k in ks:
        # 参照窓は現在の k 秒リターンと重ならないよう k だけ過去にずらす
        ref_exprs.append(
            pl.col(f"ret_{k}s")
            .rolling_std(window_size=sigma_ref_n, min_samples=max(sigma_ref_n // 2, 2))
            .shift(k)
            .alias(f"sigma_ref_{k}s")
        )
    df = df.with_columns(ref_exprs)
    df = df.with_columns(
        [
            pl.when(pl.col(f"sigma_ref_{k}s") > 0)
            .then(pl.col(f"ret_{k}s").abs() / pl.col(f"sigma_ref_{k}s"))
            .otherwise(None)
            .alias(f"velocity_{k}")
            for k in ks
        ]
    )

    # --- ティック密度 ----------------------------------------------------------------
    min_ref_tc = float(f["min_ref_trade_count"])
    df = df.with_columns(
        pl.col("trade_count").rolling_sum(window_size=tw, min_samples=tw).alias("tick_count_w")
    )
    df = df.with_columns(
        pl.col("tick_count_w")
        .rolling_median(window_size=tick_ref_n, min_samples=max(tick_ref_n // 2, 2))
        .shift(tw)
        .alias("tick_ref")
    )
    df = df.with_columns(
        pl.when(pl.col("tick_ref") >= min_ref_tc)
        .then(pl.col("tick_count_w") / pl.col("tick_ref"))
        .otherwise(None)
        .alias("tick_ratio")
    )

    # --- フローの一方向性 -------------------------------------------------------------
    min_ref_vol = float(f["min_ref_volume"])
    df = df.with_columns(
        pl.col("ofi").rolling_sum(window_size=ow, min_samples=ow).alias("ofi_w"),
        pl.col("volume").rolling_sum(window_size=ow, min_samples=ow).alias("vol_w"),
    )
    df = df.with_columns(
        pl.when(pl.col("vol_w") > min_ref_vol)
        .then(pl.col("ofi_w").abs() / pl.col("vol_w"))
        .otherwise(None)
        .alias("ofi_ratio")
    )

    # --- 価格インパクト（板消失の代理変数） --------------------------------------------
    df = df.with_columns(
        pl.col("volume").rolling_sum(window_size=iw, min_samples=iw).alias("_vol_iw"),
        (pl.col("close") / pl.col("close").shift(iw) - 1.0).abs().alias("_ret_iw_abs"),
    )
    df = df.with_columns(
        pl.when(pl.col("_vol_iw") > min_ref_vol)
        .then(pl.col("_ret_iw_abs") / pl.col("_vol_iw"))
        .otherwise(None)
        .alias("impact")
    )
    df = df.with_columns(
        pl.col("impact")
        .rolling_median(window_size=impact_ref_n, min_samples=max(impact_ref_n // 2, 2))
        .shift(iw)
        .alias("impact_ref")
    )
    df = df.with_columns(
        pl.when(pl.col("impact_ref") > 0)
        .then(pl.col("impact") / pl.col("impact_ref"))
        .otherwise(None)
        .alias("impact_ratio")
    )

    # --- スプレッド代理 / 執行コスト推定用 ---------------------------------------------
    df = df.with_columns(
        pl.col("gap_max_bps").rolling_max(window_size=ow, min_samples=1).alias("gap_max_bps_w"),
        pl.col("gap_mean_bps")
        .rolling_mean(window_size=sigma_ref_n, min_samples=max(sigma_ref_n // 2, 2))
        .shift(ow)
        .alias("gap_mean_bps_ref"),
    )

    # --- データ品質 ------------------------------------------------------------------
    df = df.with_columns(
        (1.0 - pl.col("is_filled").cast(pl.Float64))
        .rolling_mean(window_size=sigma_ref_n, min_samples=max(sigma_ref_n // 2, 2))
        .alias("active_ratio_1h"),
        pl.col("is_filled")
        .cast(pl.Int64)
        .rolling_sum(window_size=max_gap, min_samples=max_gap)
        .rolling_max(window_size=sigma_ref_n, min_samples=1)
        .alias("_max_empty_run_proxy"),
    )
    df = df.with_columns(
        (pl.col("_max_empty_run_proxy") >= max_gap).fill_null(True).alias("has_long_gap_1h")
    )

    df = df.with_columns(
        (
            (pl.col("active_ratio_1h") >= min_active)
            & ~pl.col("has_long_gap_1h")
            & ~pl.col("is_filled")
        )
        .fill_null(False)
        .alias("data_ok")
    )
    return df.drop("_vol_iw", "_ret_iw_abs", "_max_empty_run_proxy")


def add_calm_columns(df: pl.DataFrame, calm_thresholds: pl.DataFrame | None,
                     calm_window: int, calm_exclude: int) -> pl.DataFrame:
    """凪判定（§3.2 の is_calm）。窓幅・分位点の探索ではここだけ再計算すればよい。

    `rv_calm_prev` は発火分（calm_exclude 秒）を除いた直前の実現ボラ。
    `calm_threshold` は「その時刻より前」のデータだけから作られた分位点
    （calm_threshold_series 側で shift(1) 済み）。
    """
    out = df.with_columns(realized_vol(calm_window).alias("rv_calm"))
    out = out.with_columns(pl.col("rv_calm").shift(calm_exclude).alias("rv_calm_prev"))
    if "calm_threshold" in out.columns:
        out = out.drop("calm_threshold")
    if calm_thresholds is not None and calm_thresholds.height:
        out = out.join_asof(
            calm_thresholds.sort("ts"), on="ts", strategy="backward", allow_exact_matches=True
        )
    else:
        out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias("calm_threshold"))
    return out.with_columns(
        (
            pl.col("rv_calm_prev").is_not_null()
            & pl.col("calm_threshold").is_not_null()
            & (pl.col("rv_calm_prev") <= pl.col("calm_threshold"))
        ).alias("is_calm")
    )


def feature_column_names(params: Params) -> list[str]:
    return FEATURE_COLUMNS + velocity_cols(params.get_path("features.velocity_ks"))
