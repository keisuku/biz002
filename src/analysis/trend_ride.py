"""「特大の動きに乗って、ストップを置いて放置する」手法の測定。

指示書のスキャル前提とは**別の手法**として扱う。違いは次の 3 点で、いずれも
判定結果を左右する:

| | 指示書のスキャル | この測定 |
|---|---|---|
| 保有 | 60 秒 / 勢いが緩んだら降りる | 損切りに掛かるまで放置（分〜時間） |
| 指標 | MFE の**中央値** | **R 倍率の平均と右の裾** |
| 前提 | 凪（過去 7 日で最低ボラ） | 凪条件を使わない（荒れた局面で効く手法） |

## なぜ中央値を使わないか

この手法の損益は「多くは −1R で損切り、たまに +5R や +10R」という形になる。
中央値はその「たまに大きく」を定義上捨てるので、**中央値で判定すると
トレンドフォローは構造的に必ず不合格になる**。見るべきは平均 R、合計 R、
そして +3R 以上が何割あるか。

## 裁量部分の扱い

「そろそろ調整が来るだろうから逆方向に乗る」という判断は検出器に直接
コピーできない。代わりに、直前トレンドとの関係を計算可能な形にして層別する:

* `trend_z` … 直前 1 時間のリターンを、その長さのリターンの標準偏差で割ったもの。
  絶対値が大きいほど「すでに伸び切っている」
* `trend_align` … 初動の向きが直前トレンドと同じか逆か
* `vol_z` … 実現ボラが過去 7 日の水準からどれだけ離れているか（暴落期かどうか）

順張りと逆張りは期待値が全く違う可能性が高いので、必ず分けて集計する。
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ..config import Params


def vol_regime_series(minute_rv: pl.DataFrame, rv_col: str, ref_days: int,
                      sample_seconds: int) -> pl.DataFrame:
    """実現ボラの水準を、過去 ref_days 日の分布からの乖離（z）で表す。

    暴落期かどうかの代理変数。`shift(1)` で当該時点を含めない。
    """
    n = int(ref_days * 24 * 3600 / sample_seconds)
    ms = min(n, max(int(86_400 / sample_seconds), 10))
    med = pl.col(rv_col).rolling_median(window_size=n, min_samples=ms).shift(1)
    sd = pl.col(rv_col).rolling_std(window_size=n, min_samples=ms).shift(1)
    return minute_rv.select(
        "ts",
        pl.when(sd > 0).then((pl.col(rv_col) - med) / sd).otherwise(None).alias("vol_z"),
    )


def trend_context(seconds: pl.DataFrame, lookback: int, ref_seconds: int) -> pl.DataFrame:
    """直前トレンドの向きと伸び具合。すべて後ろ向きの窓のみ。"""
    logc = pl.col("close").log()
    trail = logc - logc.shift(lookback)
    sd = trail.rolling_std(window_size=ref_seconds,
                           min_samples=max(ref_seconds // 4, 2)).shift(lookback)
    return seconds.select(
        "ts",
        (trail * 1e4).alias("trend_ret_bps"),
        pl.when(sd > 0).then(trail / sd).otherwise(None).alias("trend_z"),
    )


def simulate_rides(events: pl.DataFrame, seconds: pl.DataFrame, params: Params,
                   max_hold: int, costs: pl.DataFrame | None = None,
                   context: pl.DataFrame | None = None) -> pl.DataFrame:
    """初動から entry_age 秒後に乗り、ストップを置いて放置した場合の結果。

    出口は「ストップに掛かる」か「最大保有時間に達する」かのみ。
    勢いが緩んだかどうかは**見ない**（それがこの手法の定義）。
    """
    cfg = params["trend_ride"]
    age = int(cfg["entry_age_seconds"])
    stop_k = float(cfg["stop_k"])
    min_stop = float(cfg["min_stop_bps"])
    if events.height == 0 or seconds.height == 0:
        return pl.DataFrame(schema={"symbol": pl.String, "impulse_ts": pl.Int64,
                                    "direction": pl.Int8, "max_hold_s": pl.Int64,
                                    "r_multiple": pl.Float64})

    ts = seconds["ts"].to_numpy()
    if ts[-1] - ts[0] + 1 != len(ts):
        raise ValueError("seconds frame must be a dense 1s grid")
    ts0 = int(ts[0])
    n = len(ts)
    high, low, close = (seconds[c].to_numpy() for c in ("high", "low", "close"))
    gap = seconds["gap_max_bps"].to_numpy()

    cost_by_ts = dict(zip(costs["ts"].to_list(), costs["cost_pct"].to_list())) if costs is not None else {}
    ctx = {}
    if context is not None and context.height:
        ctx = {int(r["ts"]): r for r in context.iter_rows(named=True)}

    rows: list[dict] = []
    for t0, direction, impulse_bps, symbol in zip(
        events["ts"].to_numpy(), events["direction"].to_numpy(),
        events["impulse_bps"].to_numpy(),
        events["symbol"].to_list() if "symbol" in events.columns else ["?"] * events.height,
    ):
        i = int(t0) + age - ts0
        if i < 0 or i + 1 >= n:
            continue
        entry = float(close[i])
        if not np.isfinite(entry) or entry <= 0:
            continue
        d = int(direction)
        stop_dist = max(stop_k * float(impulse_bps), min_stop) / 1e4 * entry
        if stop_dist <= 0:
            continue
        stop_px = entry - d * stop_dist

        end = min(i + max_hold, n - 1)
        if end <= i:
            continue
        sl = slice(i + 1, end + 1)
        adverse = (low[sl] <= stop_px) if d > 0 else (high[sl] >= stop_px)
        if adverse.any():
            off = int(np.argmax(adverse)) + 1
            t = i + off
            slip_px = stop_px * float(gap[t]) / 1e4
            fill = max(low[t], stop_px - slip_px) if d > 0 else min(high[t], stop_px + slip_px)
            reason = "stop"
        else:
            off = end - i
            fill = float(close[end])
            reason = "max_hold"

        fav = (float(np.nanmax(high[sl])) - entry) if d > 0 else (entry - float(np.nanmin(low[sl])))
        cost_pct = cost_by_ts.get(int(ts[i]), 0.0)
        cost_px = (cost_pct or 0.0) / 100.0 * entry
        c = ctx.get(int(ts[i]), {})
        trend_z = c.get("trend_z")
        rows.append({
            "symbol": symbol,
            "impulse_ts": int(t0),
            "direction": d,
            "max_hold_s": max_hold,
            "entry_price": entry,
            "stop_price": stop_px,
            "stop_dist_bps": stop_dist / entry * 1e4,
            "impulse_bps": float(impulse_bps),
            "exit_reason": reason,
            "hold_seconds": int(off),
            "gross_r": d * (float(fill) - entry) / stop_dist,
            "r_multiple": (d * (float(fill) - entry) - cost_px) / stop_dist,
            "mfe_r": max(fav, 0.0) / stop_dist,
            "cost_r": cost_px / stop_dist,
            "trend_z": trend_z,
            "vol_z": c.get("vol_z"),
            "trend_align": (None if trend_z is None
                            else bool(np.sign(trend_z) == d)),
        })
    # Long real-data segments can have more than Polars' default inference
    # sample of leading rows without trend context, followed by finite trend_z
    # values.  Infer from all rows so the column is Float64 rather than Null.
    return pl.DataFrame(rows, infer_schema_length=None)


def random_entries(seconds: pl.DataFrame, template: pl.DataFrame, seed: int,
                   multiplier: int, region: tuple[int, int] | None = None) -> pl.DataFrame:
    """**対照群**: 同じ期間・同じ方向分布・同じストップ幅分布で、時刻だけランダムに置く。

    これが必要な理由:

    荒れた日（=多くは暴落日）を選んで、その中でショートを取れば、初動の検出に
    何の価値が無くてもプラスになる。単に下げ相場でショートを持っていただけだからだ。
    **「シグナルに価値がある」と言うためには、同じ局面のランダムな時刻に対して
    勝っていなければならない。** 局面そのものの効果を差し引くのがこの対照群。

    方向とストップ幅は実シグナルから復元抽出するので、両者の違いは
    **「いつ入るか」だけ**になる。
    """
    if template.height == 0 or seconds.height == 0:
        return _empty_starts_like(template)
    ts = seconds["ts"].to_numpy()
    ok = ~seconds["is_filled"].to_numpy() if "is_filled" in seconds.columns else np.ones(len(ts), bool)
    lo, hi = (region if region is not None else (int(ts[0]), int(ts[-1]) + 1))
    usable = ts[ok & (ts >= lo) & (ts < hi)]
    if usable.size == 0:
        return _empty_starts_like(template)

    rng = np.random.default_rng(seed)
    n = template.height * max(int(multiplier), 1)
    picks = rng.choice(usable, size=n, replace=True)
    idx = rng.integers(0, template.height, size=n)
    return pl.DataFrame({
        "ts": np.sort(picks),
        "direction": template["direction"].to_numpy()[idx].astype(np.int8),
        "impulse_bps": template["impulse_bps"].to_numpy()[idx],
        "symbol": [template["symbol"][0] if "symbol" in template.columns else "?"] * n,
    })


def _empty_starts_like(template: pl.DataFrame) -> pl.DataFrame:
    return pl.DataFrame(schema={"ts": pl.Int64, "direction": pl.Int8,
                                "impulse_bps": pl.Float64, "symbol": pl.String})


def robustness(rides: pl.DataFrame, block_seconds: int = 259_200) -> pl.DataFrame:
    """1 つの窓（既定 3 日）を取り除いたときに平均 R がどうなるか。

    「最も寄与した窓を 1 つ抜くと平均がゼロ以下になる」なら、実効サンプル数は
    ほぼ 1 であり、統計的な根拠にならない。今回それが起きたので常設の指標にする。
    """
    if rides.height == 0:
        return pl.DataFrame(schema={"n_blocks": pl.Int64})
    r = rides.with_columns((pl.col("impulse_ts") // block_seconds).alias("_block"))
    total = float(r["r_multiple"].sum())
    n = r.height
    rows = []
    for blk, sub in r.group_by("_block"):
        k = sub.height
        if n - k <= 0:
            continue
        rows.append({
            "block": int(blk[0] if isinstance(blk, tuple) else blk),
            "n_in_block": k,
            "block_total_r": float(sub["r_multiple"].sum()),
            "mean_r_without_block": (total - float(sub["r_multiple"].sum())) / (n - k),
        })
    if not rows:
        return pl.DataFrame(schema={"n_blocks": pl.Int64})
    out = pl.DataFrame(rows).sort("block_total_r", descending=True)
    return out.with_columns(
        pl.lit(total / n).alias("mean_r_all"),
        pl.lit(out.height).alias("n_blocks"),
    )


def robustness_verdict(rob: pl.DataFrame) -> dict:
    """最大寄与ブロックを抜いた後も平均 R が正かどうか。"""
    if rob.height == 0:
        return {"verdict": "NO_DATA"}
    worst = rob.row(0, named=True)          # 寄与最大のブロックを抜いた行
    survives = worst["mean_r_without_block"] > 0
    return {
        "verdict": "SURVIVES_LEAVE_ONE_OUT" if survives else "DEPENDS_ON_ONE_BLOCK",
        "note": ("最大寄与ブロックを抜いても平均 R は正。"
                 if survives else
                 "**最大寄与ブロックを抜くと平均 R が正でなくなる。**"
                 "実効サンプル数がほぼ 1 であり、偶然と区別できない。"),
        "mean_r_all": worst["mean_r_all"],
        "mean_r_without_top_block": worst["mean_r_without_block"],
        "top_block_total_r": worst["block_total_r"],
        "n_blocks": worst["n_blocks"],
    }


def signal_vs_random(rides: pl.DataFrame, params: Params,
                     by: list[str] | None = None) -> pl.DataFrame:
    """シグナル群と対照群（ランダム時刻）の比較表。

    差が無ければ、勝っていたのは**局面**であってシグナルではない。
    """
    if rides.height == 0 or "source" not in rides.columns:
        return pl.DataFrame(schema={"source": pl.String, "n": pl.UInt32})
    group = (by or ["max_hold_s"]) + ["source"]
    return summarize(rides, params, by=group)


def summarize(rides: pl.DataFrame, params: Params,
              by: list[str] | None = None) -> pl.DataFrame:
    """R 倍率の集計。中央値ではなく平均・合計・右の裾を見る。"""
    if rides.height == 0:
        return pl.DataFrame(schema={"n": pl.UInt32})
    thresholds = [float(t) for t in params.get_path("trend_ride.r_thresholds")]
    group = by or ["max_hold_s"]
    aggs = [
        pl.len().alias("n"),
        (pl.col("r_multiple") > 0).mean().alias("win_rate"),
        pl.col("r_multiple").mean().alias("mean_r"),
        pl.col("r_multiple").median().alias("median_r"),
        pl.col("r_multiple").sum().alias("total_r"),
        pl.col("r_multiple").max().alias("max_r"),
        pl.col("r_multiple").min().alias("min_r"),
        pl.col("mfe_r").mean().alias("mean_mfe_r"),
        pl.col("cost_r").mean().alias("mean_cost_r"),
        pl.col("hold_seconds").median().alias("median_hold_s"),
        (pl.col("exit_reason") == "stop").mean().alias("stop_rate"),
        pl.col("r_multiple").filter(pl.col("r_multiple") > 0).sum().alias("_gross_win"),
        pl.col("r_multiple").filter(pl.col("r_multiple") < 0).sum().alias("_gross_loss"),
    ]
    aggs += [(pl.col("r_multiple") >= t).mean().alias(f"p_ge_{int(t)}r") for t in thresholds]
    out = rides.group_by(group).agg(aggs)
    return out.with_columns(
        pl.when(pl.col("_gross_loss") < 0)
        .then(pl.col("_gross_win") / pl.col("_gross_loss").abs())
        .otherwise(None)
        .alias("profit_factor")
    ).drop("_gross_win", "_gross_loss").sort(group)


def stratify(rides: pl.DataFrame, params: Params) -> pl.DataFrame:
    """順張り / 逆張り、ボラ局面別の集計。

    A ride table can contain several pre-registered ``window_s`` /
    ``sigma_mult`` definitions.  They must remain separate: pooling them would
    count the same market move several times and make the stratum sample size
    and expectancy uninterpretable.
    """
    if rides.height == 0:
        return pl.DataFrame(schema={"stratum": pl.String, "n": pl.UInt32})
    edges = [float(x) for x in params.get_path("trend_ride.trend_z_buckets")]
    r = rides.with_columns(
        pl.when(pl.col("trend_align").is_null()).then(pl.lit("unknown"))
        .when(pl.col("trend_align")).then(pl.lit("with_trend"))
        .otherwise(pl.lit("counter_trend")).alias("align"),
        pl.when(pl.col("vol_z").is_null()).then(pl.lit("unknown"))
        .when(pl.col("vol_z") >= edges[-1]).then(pl.lit("vol_extreme"))
        .when(pl.col("vol_z") >= edges[-2]).then(pl.lit("vol_high"))
        .otherwise(pl.lit("vol_normal")).alias("vol_bucket"),
        pl.when(pl.col("direction") > 0).then(pl.lit("long")).otherwise(pl.lit("short")).alias("side"),
    )
    frames = []
    signal_cols = [
        c for c in ("window_s", "sigma_mult") if c in r.columns
    ]
    for cols, name in ((["align"], "align"), (["vol_bucket"], "vol"), (["side"], "side"),
                       (["align", "vol_bucket"], "align_x_vol")):
        s = summarize(r, params, by=signal_cols + cols + ["max_hold_s"])
        parts: list[pl.Expr] = [pl.lit(f"{name}:")]
        for j, c in enumerate(cols):
            if j:
                parts.append(pl.lit("|"))
            parts.append(pl.col(c).cast(pl.String))
        label = pl.concat_str(parts, separator="")
        frames.append(s.with_columns(label.alias("stratum")).drop(cols))
    return pl.concat(frames, how="diagonal_relaxed").sort(["stratum", "max_hold_s"])


def verdict(summary: pl.DataFrame, params: Params) -> dict:
    """この手法の言葉での生死判定: 平均 R がコスト控除後に正か。"""
    if summary.height == 0:
        return {"verdict": "NO_TRADES"}
    best = summary.sort("mean_r", descending=True, nulls_last=True).row(0, named=True)
    min_n = int(params.get_path("validation.min_wins"))
    enough = int(best["n"] * best["win_rate"]) >= min_n
    return {
        "verdict": ("POSITIVE_EXPECTANCY" if best["mean_r"] > 0 else "NEGATIVE_EXPECTANCY"),
        "note": (
            "R 倍率の平均が正。ただしサンプル数と局面の偏りを必ず確認すること。"
            if best["mean_r"] > 0 else
            "R 倍率の平均が負。この定義・この期間では成立していない。"
        ),
        "best_max_hold_s": best["max_hold_s"],
        "n": best["n"],
        "win_rate": best["win_rate"],
        "mean_r": best["mean_r"],
        "median_r": best["median_r"],
        "total_r": best["total_r"],
        "profit_factor": best.get("profit_factor"),
        "max_r": best["max_r"],
        "mean_cost_r": best["mean_cost_r"],
        "enough_wins": enough,
        "min_wins": min_n,
    }
