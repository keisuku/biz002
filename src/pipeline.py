"""Phase 1 → Phase 2 のオーケストレーション。

チャンク分割の考え方:
* 特徴量は「助走（warmup）付きのチャンク」で計算し、発火の抽出は助走部分を除いた
  区間に限定する。こうするとチャンク境界で参照統計が欠けることによる取りこぼしが出ない。
* 結果指標（MFE/MAE）と出口シミュレーションには前方データが要るので、チャンク末尾に
  最大ホライズン + 最大保有時間ぶんの延長を読む。
* チャンク幅は結果に影響しない（メモリ制御のみ）。テストで境界不変性を検査している。
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from . import events as ev_mod
from . import exits as exits_mod
from . import features as feat
from . import outcomes as out_mod
from .config import Params, resolve_dir
from .data.to_seconds import day_start_ts, load_seconds, SECONDS_PER_DAY

log = logging.getLogger(__name__)


def day_list(start: date, end: date) -> list[date]:
    out, d = [], start
    while d <= end:
        out.append(d)
        d += timedelta(days=1)
    return out


def forward_seconds(params: Params) -> int:
    return int(max(params.get_path("outcomes.horizons")) + params.get_path("exits.max_hold_seconds") + 5)


# --------------------------------------------------------------------------------------
# 凪参照（分足 rv）
# --------------------------------------------------------------------------------------
def refs_path(params: Params, symbol: str) -> Path:
    return resolve_dir(params, "refs_dir") / f"{symbol}_minute_rv.parquet"


def build_refs(params: Params, symbol: str, days: list[date]) -> pl.DataFrame:
    """凪判定に使う分足 rv 系列を全期間ぶん作る（日跨ぎの助走込み）。"""
    windows = sorted({int(params.get_path("thresholds.calm_window")),
                      *[int(w) for w in params.get_path("grid.calm_window")]})
    sample = int(params.get_path("features.calm_ref_sample_seconds"))
    frames = []
    for d in days:
        t0 = day_start_ts(d)
        warm = max(windows) + sample
        sec = load_seconds(params, symbol, t0 - warm, t0 + SECONDS_PER_DAY)
        if sec.height == 0:
            continue
        mr = feat.minute_rv_series(sec, windows, sample)
        frames.append(mr.filter(pl.col("ts") >= t0))
    if not frames:
        return pl.DataFrame(schema={"ts": pl.Int64, **{f"rv_{w}": pl.Float64 for w in windows}})
    out = pl.concat(frames).unique(subset=["ts"], keep="last").sort("ts")
    p = refs_path(params, symbol)
    p.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(p)
    return out


def load_refs(params: Params, symbol: str) -> pl.DataFrame:
    p = refs_path(params, symbol)
    if not p.exists():
        raise FileNotFoundError(f"missing calm reference series: {p} (run `refs` first)")
    return pl.read_parquet(p)


# --------------------------------------------------------------------------------------
# 発火台帳
# --------------------------------------------------------------------------------------
def _chunks(days: list[date], chunk_days: int) -> list[tuple[date, date]]:
    return [(days[i], days[min(i + chunk_days, len(days)) - 1])
            for i in range(0, len(days), chunk_days)]


def _contiguous_core_slice(
    seconds: pl.DataFrame, core_start_ts: int, core_end_ts: int
) -> pl.DataFrame:
    """Keep only the contiguous span containing the requested core interval.

    A sparse research download can contain another selected segment just inside
    the requested warmup/forward range, separated by an undownloaded day.
    Rolling across that gap would be invalid.  Gaps inside the core remain a
    hard error; gaps outside it bound the usable context.
    """
    if seconds.height == 0:
        return seconds
    ts = seconds["ts"].to_numpy()
    core_left = int(np.searchsorted(ts, core_start_ts, side="left"))
    core_right = int(np.searchsorted(ts, core_end_ts, side="left")) - 1
    if core_left >= len(ts) or core_right < core_left:
        return seconds.head(0)
    if ts[core_left] != core_start_ts or ts[core_right] != core_end_ts - 1:
        raise ValueError("seconds frame does not fully cover the core interval")
    gaps = np.flatnonzero(np.diff(ts) != 1)
    if np.any((gaps >= core_left) & (gaps < core_right)):
        raise ValueError("seconds frame has a gap inside the core interval")
    left_gaps = gaps[gaps < core_left]
    right_gaps = gaps[gaps >= core_right]
    left = int(left_gaps[-1] + 1) if left_gaps.size else 0
    right = int(right_gaps[0] + 1) if right_gaps.size else len(ts)
    return seconds.slice(left, right - left)


def build_events(params: Params, symbol: str, days: list[date],
                 minute_rv: pl.DataFrame | None = None,
                 with_exits: bool = True,
                 stop_fill: exits_mod.StopFillEstimator | None = None,
                 cost_multiplier: float = 1.0) -> tuple[pl.DataFrame, pl.DataFrame]:
    """発火台帳（イベント × 特徴量 × 結果指標 × コスト）と出口シミュレーション結果を返す。"""
    if minute_rv is None:
        minute_rv = load_refs(params, symbol)
    calm_window = int(params.get_path("thresholds.calm_window"))
    pct = float(params.get_path("thresholds.calm_percentile"))
    ref_days = int(params.get_path("features.calm_ref_days"))
    sample = int(params.get_path("features.calm_ref_sample_seconds"))
    calm_excl = int(params.get_path("features.calm_exclude_seconds"))
    warm = int(params.get_path("validation.warmup_seconds"))
    fwd = forward_seconds(params)
    fixed_h = int(params.get_path("validation.sweep_horizon"))

    thresholds = feat.calm_threshold_series(minute_rv, calm_window, pct, ref_days, sample)

    ev_frames, sim_frames = [], []
    for c0, c1 in _chunks(days, int(params.get_path("validation.chunk_days"))):
        t0, t1 = day_start_ts(c0), day_start_ts(c1) + SECONDS_PER_DAY
        sec = load_seconds(params, symbol, t0 - warm, t1 + fwd)
        sec = _contiguous_core_slice(sec, t0, t1)
        if sec.height == 0:
            continue
        core = feat.compute_core_features(sec, params)
        feats = feat.add_calm_columns(core, thresholds, calm_window, calm_excl)
        e = ev_mod.extract_events(feats, params, symbol, region=(t0, t1))
        if e.height == 0:
            continue
        e = ev_mod.add_time_context(e)
        e = out_mod.add_costs(e, params, multiplier=cost_multiplier)
        e = out_mod.add_outcomes(e, sec, params)
        ev_frames.append(e)
        if with_exits:
            sim_frames.append(exits_mod.simulate(e, sec, params, fixed_h, stop_fill=stop_fill,
                                                 cost_multiplier=cost_multiplier))
    ledger = pl.concat(ev_frames, how="diagonal_relaxed") if ev_frames else ev_mod._empty_events(params)
    # extract_events is called per memory chunk. Apply cooldown once more over
    # the assembled ledger so a chunk boundary cannot create a duplicate event.
    if ledger.height:
        ledger = ledger.sort("event_ts")
        keep = ev_mod.apply_cooldown(
            ledger["event_ts"].to_numpy(),
            ledger["direction"].to_numpy(),
            int(params.get_path("events.cooldown_seconds")),
            str(params.get_path("events.cooldown_scope")),
        )
        ledger = ledger.filter(pl.Series(keep))
    sim = pl.concat(sim_frames, how="diagonal_relaxed") if sim_frames else pl.DataFrame()
    if sim.height and ledger.height:
        sim = sim.join(
            ledger.select("symbol", "event_ts", "direction"),
            on=["symbol", "event_ts", "direction"],
            how="semi",
        )
    return ledger.sort("event_ts"), sim


def build_funnel(params: Params, symbol: str, days: list[date],
                 minute_rv: pl.DataFrame | None = None) -> dict:
    """発火条件のファネル診断（§3.3 のどの条件が律速かを見る）。件数は変えない。"""
    from .analysis import funnel as funnel_mod

    if minute_rv is None:
        minute_rv = load_refs(params, symbol)
    calm_window = int(params.get_path("thresholds.calm_window"))
    pct = float(params.get_path("thresholds.calm_percentile"))
    ref_days = int(params.get_path("features.calm_ref_days"))
    sample = int(params.get_path("features.calm_ref_sample_seconds"))
    calm_excl = int(params.get_path("features.calm_exclude_seconds"))
    warm = int(params.get_path("validation.warmup_seconds"))
    thresholds = feat.calm_threshold_series(minute_rv, calm_window, pct, ref_days, sample)

    chunks = []
    for c0, c1 in _chunks(days, int(params.get_path("validation.chunk_days"))):
        t0, t1 = day_start_ts(c0), day_start_ts(c1) + SECONDS_PER_DAY
        sec = load_seconds(params, symbol, t0 - warm, t1)
        if sec.height == 0:
            continue
        core = feat.compute_core_features(sec, params)
        feats = feat.add_calm_columns(core, thresholds, calm_window, calm_excl)
        feats = feats.filter((pl.col("ts") >= t0) & (pl.col("ts") < t1))
        if feats.height:
            chunks.append(funnel_mod.count_chunk(feats, params))
    return funnel_mod.merge_chunks(chunks)


def build_impulse_scan(params: Params, symbol: str, days: list[date]) -> pl.DataFrame:
    """初動エントリの天井測定（src/analysis/impulse.py）。検出器は使わない。

    コストは発火台帳と**同じモデル**で見積もる（比較可能にするため）。そのため
    core features を計算して gap / range 列をエントリ秒から引く。
    """
    from .analysis import impulse as imp

    cfg = params["impulse_scan"]
    warm = int(params.get_path("validation.warmup_seconds"))
    fwd = int(max(cfg["horizons"]) + max(cfg["entry_ages"]) + max(cfg["windows"]) + 5)
    frames = []
    for c0, c1 in _chunks(days, int(params.get_path("validation.chunk_days"))):
        t0, t1 = day_start_ts(c0), day_start_ts(c1) + SECONDS_PER_DAY
        sec = load_seconds(params, symbol, t0 - warm, t1 + fwd)
        sec = _contiguous_core_slice(sec, t0, t1)
        if sec.height == 0:
            continue
        core = feat.compute_core_features(sec, params)
        costs = out_mod.add_costs(
            core.select("ts", "gap_mean_bps_ref", "gap_max_bps_w", "range_bps_1s"), params
        ).select("ts", "cost_pct")
        for w in cfg["windows"]:
            for mult in cfg["sigma_mults"]:
                starts = imp.label_impulse_starts(
                    sec, int(w), float(mult), int(cfg["cooldown_seconds"]),
                    int(params.get_path("features.sigma_ref_seconds")),
                )
                starts = starts.filter((pl.col("ts") >= t0) & (pl.col("ts") < t1))
                if starts.height == 0:
                    continue
                e = imp.oracle_entries(sec, starts, int(w), list(cfg["entry_ages"]),
                                       list(cfg["horizons"]), costs)
                if e.height:
                    frames.append(e.with_columns(pl.lit(float(mult)).alias("sigma_mult"),
                                                 pl.lit(symbol).alias("symbol")))
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def build_trend_rides(params: Params, symbol: str, days: list[date],
                      minute_rv: pl.DataFrame | None = None) -> pl.DataFrame:
    """「特大の動きに乗ってストップを置いて放置」の全トレードを返す。

    凪条件は使わない。初動のラベルは impulse.label_impulse_starts（値動きだけで決まる）。
    """
    from .analysis import impulse as imp
    from .analysis import trend_ride as ride

    cfg = params["trend_ride"]
    scan = params["impulse_scan"]
    warm = int(cfg["warmup_seconds"])
    holds = [int(h) for h in cfg["max_hold_seconds"]]
    fwd = max(holds) + int(cfg["entry_age_seconds"]) + 5

    vol = None
    if minute_rv is None:
        try:
            minute_rv = load_refs(params, symbol)
        except FileNotFoundError:
            minute_rv = None
    if minute_rv is not None:
        rv_col = f"rv_{int(params.get_path('thresholds.calm_window'))}"
        if rv_col in minute_rv.columns:
            vol = ride.vol_regime_series(
                minute_rv, rv_col, int(cfg["vol_regime_ref_days"]),
                int(params.get_path("features.calm_ref_sample_seconds")),
            )

    frames = []
    for c0, c1 in _chunks(days, int(params.get_path("validation.chunk_days"))):
        t0, t1 = day_start_ts(c0), day_start_ts(c1) + SECONDS_PER_DAY
        sec = load_seconds(params, symbol, t0 - warm, t1 + fwd)
        sec = _contiguous_core_slice(sec, t0, t1)
        if sec.height == 0:
            continue
        core = feat.compute_core_features(sec, params)
        costs = out_mod.add_costs(
            core.select("ts", "gap_mean_bps_ref", "gap_max_bps_w", "range_bps_1s"), params
        ).select("ts", "cost_pct")
        ctx = ride.trend_context(sec, int(cfg["trend_lookback_seconds"]),
                                 int(cfg["trend_ref_seconds"]))
        if vol is not None:
            ctx = ctx.join_asof(vol.sort("ts"), on="ts", strategy="backward")
        else:
            ctx = ctx.with_columns(pl.lit(None, dtype=pl.Float64).alias("vol_z"))

        for w in scan["windows"]:
            for mult in scan["sigma_mults"]:
                starts = imp.label_impulse_starts(
                    sec, int(w), float(mult), int(scan["cooldown_seconds"]),
                    int(params.get_path("features.sigma_ref_seconds")),
                )
                starts = starts.filter((pl.col("ts") >= t0) & (pl.col("ts") < t1))
                if starts.height == 0:
                    continue
                starts = starts.with_columns(pl.lit(symbol).alias("symbol"))
                for hold in holds:
                    r = ride.simulate_rides(starts, sec, params, hold, costs, ctx)
                    if r.height:
                        frames.append(r.with_columns(pl.lit(int(w)).alias("window_s"),
                                                     pl.lit(float(mult)).alias("sigma_mult")))
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def simulate_exits_for_ledger(params: Params, symbol: str, ledger: pl.DataFrame,
                              fixed_horizon: int,
                              stop_fill: exits_mod.StopFillEstimator | None = None,
                              cost_multiplier: float = 1.0) -> pl.DataFrame:
    """既存の台帳に対して、指定した固定時間ベースラインで出口ルールを回し直す。

    §4.2 で素の保有時間 H が決まってから §4.4 の比較を行うため、台帳生成とは分離する。
    """
    if ledger.height == 0:
        return exits_mod.simulate(ledger, ledger, params, fixed_horizon)
    fwd = forward_seconds(params)
    ts = ledger["event_ts"].to_numpy()
    frames = []
    chunk = int(params.get_path("validation.chunk_days")) * SECONDS_PER_DAY
    lo = int(ts.min()) // SECONDS_PER_DAY * SECONDS_PER_DAY
    hi = int(ts.max()) + 1
    while lo < hi:
        end = min(lo + chunk, hi)
        sub = ledger.filter((pl.col("event_ts") >= lo) & (pl.col("event_ts") < end))
        if sub.height:
            sec = load_seconds(params, symbol, lo, end + fwd)
            if sec.height:
                frames.append(exits_mod.simulate(sub, sec, params, fixed_horizon,
                                                 stop_fill=stop_fill,
                                                 cost_multiplier=cost_multiplier))
        lo = end
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def ledger_path(params: Params, symbol: str) -> Path:
    return resolve_dir(params, "events_dir") / f"{symbol}_events.parquet"


def save_ledger(params: Params, symbol: str, ledger: pl.DataFrame, sim: pl.DataFrame) -> None:
    p = ledger_path(params, symbol)
    p.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_parquet(p)
    if sim is not None and sim.height:
        sim.write_parquet(p.with_name(f"{symbol}_exits.parquet"))


# --------------------------------------------------------------------------------------
# パラメータ探索（§3.3 の格子 / §6 のウォークフォワード用の集計）
# --------------------------------------------------------------------------------------
def grid_combinations(params: Params) -> list[dict]:
    g = params["grid"]
    combos = []
    for cw in g["calm_window"]:
        for cp in g["calm_percentile"]:
            for s in g["sigma"]:
                for t in g["tick"]:
                    for o in g["ofi"]:
                        for im in g["impact"]:
                            combos.append({"calm_window": int(cw), "calm_percentile": float(cp),
                                           "sigma": float(s), "tick": float(t),
                                           "ofi": float(o), "impact": float(im)})
    return combos


PARAM_COLS = ["calm_window", "calm_percentile", "sigma", "tick", "ofi", "impact"]


def sweep(params: Params, symbol: str, days: list[date], folds: list[tuple[int, int]],
          minute_rv: pl.DataFrame | None = None,
          keep_returns_for: dict | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """格子全点 × fold の集計表を作る。

    評価は固定時間決済（validation.sweep_horizon 秒）の手数料・スリッページ控除後リターン。
    出口ルールの選択は別問題として §4.4 で扱う（ここで一緒に探索すると試行回数が爆発する）。

    keep_returns_for に組み合わせを 1 つ渡すと、その組み合わせのイベント別リターンも返す
    （DSR / ブートストラップ用）。
    """
    if minute_rv is None:
        minute_rv = load_refs(params, symbol)
    h = int(params.get_path("validation.sweep_horizon"))
    ref_days = int(params.get_path("features.calm_ref_days"))
    sample = int(params.get_path("features.calm_ref_sample_seconds"))
    calm_excl = int(params.get_path("features.calm_exclude_seconds"))
    warm = int(params.get_path("validation.warmup_seconds"))
    fwd = forward_seconds(params)
    combos = grid_combinations(params)

    # 凪の設定ごとに特徴量を作り直す。閾値だけの違いは同じ特徴量から評価できる。
    calm_variants = sorted({(c["calm_window"], c["calm_percentile"]) for c in combos})
    rows: list[dict] = []
    kept: list[pl.DataFrame] = []

    for c0, c1 in _chunks(days, int(params.get_path("validation.chunk_days"))):
        t0, t1 = day_start_ts(c0), day_start_ts(c1) + SECONDS_PER_DAY
        sec = load_seconds(params, symbol, t0 - warm, t1 + fwd)
        if sec.height == 0:
            continue
        core = feat.compute_core_features(sec, params)
        for cw, cp in calm_variants:
            thr = feat.calm_threshold_series(minute_rv, cw, cp, ref_days, sample)
            feats = feat.add_calm_columns(core, thr, cw, calm_excl)
            for combo in combos:
                if combo["calm_window"] != cw or combo["calm_percentile"] != cp:
                    continue
                p2 = params.with_overrides({
                    "thresholds.calm_window": cw, "thresholds.calm_percentile": cp,
                    "thresholds.sigma": combo["sigma"], "thresholds.tick": combo["tick"],
                    "thresholds.ofi": combo["ofi"], "thresholds.impact": combo["impact"],
                })
                e = ev_mod.extract_events(feats, p2, symbol, region=(t0, t1))
                if e.height == 0:
                    continue
                e = out_mod.add_costs(e, p2)
                e = out_mod.add_outcomes(e, sec, p2)
                d = e.select(
                    pl.col("event_ts"),
                    (pl.col(f"ret_{h}") - pl.col("cost_pct")).alias("net_pct"),
                ).drop_nulls()
                if d.height == 0:
                    continue
                rows.append({**combo, "chunk": c0.isoformat(), "_frame": d})
                if keep_returns_for and all(combo[k] == keep_returns_for[k] for k in PARAM_COLS):
                    kept.append(d)

    if not rows:
        return pl.DataFrame(), pl.DataFrame()

    agg_rows = []
    for r in rows:
        d: pl.DataFrame = r.pop("_frame")
        ts = d["event_ts"].to_numpy()
        fold_ids = _assign_folds(ts, folds)
        for fold in np.unique(fold_ids):
            sub = d.filter(pl.Series(fold_ids == fold))["net_pct"]
            agg_rows.append({**r, "fold": int(fold), "n": sub.len(),
                             "n_wins": int((sub > 0).sum()), "sum_net": float(sub.sum()),
                             "sum_sq": float((sub**2).sum())})
    raw = pl.DataFrame(agg_rows)
    per_fold = _finalize(raw.group_by(PARAM_COLS + ["fold"]).agg(
        pl.col("n").sum(), pl.col("n_wins").sum(), pl.col("sum_net").sum(), pl.col("sum_sq").sum()))
    overall = _finalize(raw.group_by(PARAM_COLS).agg(
        pl.col("n").sum(), pl.col("n_wins").sum(), pl.col("sum_net").sum(), pl.col("sum_sq").sum()
    )).with_columns(pl.lit(-1).alias("fold"))
    kept_df = pl.concat(kept) if kept else pl.DataFrame()
    return pl.concat([per_fold, overall], how="diagonal_relaxed"), kept_df


def _assign_folds(ts: np.ndarray, folds: list[tuple[int, int]]) -> np.ndarray:
    out = np.full(len(ts), -1, dtype=np.int64)
    for i, (lo, hi) in enumerate(folds):
        out[(ts >= lo) & (ts < hi)] = i
    return out


def _finalize(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        (pl.col("sum_net") / pl.col("n")).alias("net_mean_pct"),
        (pl.col("n_wins") / pl.col("n")).alias("win_rate"),
    ).with_columns(
        (
            ((pl.col("sum_sq") - pl.col("n") * pl.col("net_mean_pct") ** 2)
             / pl.max_horizontal(pl.col("n") - 1, pl.lit(1))).sqrt()
        ).alias("sd_net_pct")
    ).with_columns(
        pl.when(pl.col("sd_net_pct") > 0)
        .then(pl.col("net_mean_pct") / pl.col("sd_net_pct") * pl.col("n").sqrt())
        .otherwise(None)
        .alias("t_stat")
    )
