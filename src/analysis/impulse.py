"""初動エントリの**上限値（天井）**測定。

## これは何であって、何ではないか

**これはバックテストではない。戦略でもない。** 初動の開始時刻を「未来を見て」
ラベルしているので、そのままでは実行不可能である。測っているのは次の一点だけ:

> 完璧な予言者が「今この秒が初動の開始だ」と教えてくれたとして、
> そこから w + Δ 秒後に**実際に市場にあった価格**で入ったら、
> コストを引いた後に何 bps 残るか。

用途は「これ以上速くしても無駄かどうか」の判定である。

* 天井がコストを下回る → **どんな検出器を作っても勝てない。そこで終わり。**
* 天井がコストを上回る → 勝てる証拠にはならない。実時間の検出器がその天井の
  どこまで近づけるか、という次の問いに進んでよい、というだけ。

## 生き残りバイアスを入れない工夫

初動を「結果的に大きく動いた動き」で選ぶと、勝ち馬だけを拾って天井が無意味に高くなる。
そこでラベルは **初動の最初の w 秒の値動きだけ**で決める。w 秒より後の情報は
一切使わない。こうすると「w 秒窓の完璧な検出器が t0+w 時点で知り得たこと」に
ちょうど一致する、**きつい（=意味のある）上限**になる。

エントリ価格・出口価格は実データの実勢価格であり、そこに予言は入っていない。
未来を使っているのは「どの秒を選ぶか」の一点だけである。
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ..config import Params


def label_impulse_starts(seconds: pl.DataFrame, window: int, sigma_mult: float,
                         cooldown: int, ref_seconds: int = 3600,
                         onset_mult: float = 3.0) -> pl.DataFrame:
    """初動の**開始秒**をラベルする（未来参照。実時間では不可能）。

    2 段階で決める:

    1. 候補: 秒 t から前方 window 秒の値動きが、過去 ref_seconds における同窓幅
       リターンの標準偏差の sigma_mult 倍以上
    2. 開始: 候補の先頭から前方 window 秒を走査し、**1 秒リターンが有意に
       （onset_mult × 1 秒ボラ以上）動き、かつ向きが初動と一致する最初の秒**を
       初動の開始とする。向きの一致を要求しないと、ノイズの秒を開始と誤認して
       実際の動き出しより早く入ったことになり、天井が過大に出る。

    2 が要る理由: 前方 window 秒を見る以上、候補フラグは実際の動き出しより最大
    window-1 秒早く立つ。そのままだと「動く前に入った」ことになり、天井が
    不当に高く出る。ここを外すと測定全体が意味を失う。
    """
    logc = seconds["close"].log()
    fwd = logc.shift(-window) - logc                       # ← 未来参照（意図的）
    past = logc - logc.shift(window)
    sigma = past.rolling_std(window_size=ref_seconds,
                             min_samples=max(ref_seconds // 2, 2)).shift(window)
    r1 = logc - logc.shift(1)
    sigma1 = r1.rolling_std(window_size=ref_seconds,
                            min_samples=max(ref_seconds // 2, 2)).shift(1)

    df = pl.DataFrame({
        "ts": seconds["ts"], "fwd": fwd, "sigma": sigma,
        "r1": r1, "sigma1": sigma1, "is_filled": seconds["is_filled"],
    }).with_columns(
        (
            (pl.col("sigma") > 0)
            & (pl.col("fwd").abs() >= sigma_mult * pl.col("sigma"))
            & ~pl.col("is_filled")
        ).fill_null(False).alias("cand")
    ).with_columns(
        (pl.col("cand") & ~pl.col("cand").shift(1, fill_value=False)).alias("cand_start"),
        (
            (pl.col("sigma1") > 0)
            & (pl.col("r1").abs() >= onset_mult * pl.col("sigma1"))
        ).fill_null(False).alias("moving"),
    )

    idx = np.flatnonzero(df["cand_start"].to_numpy())
    if idx.size == 0:
        return _empty_starts()
    moving = df["moving"].to_numpy()
    ts = df["ts"].to_numpy()
    logc_np = logc.to_numpy()
    r1_np = np.nan_to_num(df["r1"].to_numpy(), nan=0.0)
    fwd_np = np.nan_to_num(df["fwd"].to_numpy(), nan=0.0)
    n = len(ts)

    onset: list[int] = []
    for c in idx:
        # 候補の先頭から window 秒以内で、初動と同じ向きに有意に動いた最初の秒
        want = 1.0 if fwd_np[c] > 0 else -1.0
        hit = c
        for j in range(c, min(c + window + 1, n)):
            if moving[j] and np.sign(r1_np[j]) == want:
                hit = j
                break
        else:
            hit = c
        onset.append(int(hit))

    onset_arr = np.array(sorted(set(onset)), dtype=np.int64)
    keep = _cooldown(ts[onset_arr], cooldown)
    onset_arr = onset_arr[keep]
    if onset_arr.size == 0:
        return _empty_starts()

    # 初動の大きさは「開始から window 秒」で測る。その先の値動きは一切使わない
    # （結果的に伸びた動きだけを拾うと、勝ち馬だけを選んだ天井になる）
    ends = np.minimum(onset_arr + window, n - 1)
    move = logc_np[ends] - logc_np[onset_arr]
    return pl.DataFrame({
        "ts": ts[onset_arr],
        "direction": np.where(move > 0, 1, -1).astype(np.int8),
        "impulse_bps": np.abs(move) * 1e4,
    })


def _empty_starts() -> pl.DataFrame:
    return pl.DataFrame(schema={"ts": pl.Int64, "direction": pl.Int8,
                                "impulse_bps": pl.Float64})


def _cooldown(ts: np.ndarray, cooldown: int) -> np.ndarray:
    keep = np.zeros(len(ts), dtype=bool)
    last = -(10**18)
    for i, t in enumerate(ts):
        if t - last > cooldown:
            keep[i] = True
            last = int(t)
    return keep


def oracle_entries(seconds: pl.DataFrame, starts: pl.DataFrame, window: int,
                   ages: list[int], horizons: list[int],
                   costs: pl.DataFrame | None = None) -> pl.DataFrame:
    """初動開始から `age` 秒後の**実勢終値**で入った場合の成果を測る。

    `age`（初動年齢）は次の合計として読む:

        age = 検出窓 + 通知遅延 + 人間の反応

    * age=0 は物理的に不可能な理想値。**どんな検出器も超えられない天井**。
    * w 秒窓の検出器を使うなら age >= w。10 秒窓なら age >= 10。
    * 「初動から 7 秒で入る」は age=7 の行を見ればよい。

    未来を使っているのは開始秒の特定だけで、価格はすべて実データの実勢値である。
    """
    if starts.height == 0 or seconds.height == 0:
        return pl.DataFrame(schema={"window_s": pl.Int64, "delay_s": pl.Int64})
    ts = seconds["ts"].to_numpy()
    if ts[-1] - ts[0] + 1 != len(ts):
        raise ValueError("seconds frame must be a dense 1s grid")
    ts0 = int(ts[0])
    high = seconds["high"].to_numpy()
    low = seconds["low"].to_numpy()
    close = seconds["close"].to_numpy()
    n = len(ts)

    cost_by_ts = {}
    if costs is not None and costs.height:
        cost_by_ts = dict(zip(costs["ts"].to_list(), costs["cost_pct"].to_list()))

    rows: list[dict] = []
    for t0, direction, impulse_bps in zip(starts["ts"].to_numpy(),
                                          starts["direction"].to_numpy(),
                                          starts["impulse_bps"].to_numpy()):
        for age in ages:
            i = int(t0) + age - ts0
            if i < 0 or i >= n:
                continue
            entry = float(close[i])
            if not np.isfinite(entry) or entry <= 0:
                continue
            row = {
                "impulse_ts": int(t0),
                "window_s": window,
                "impulse_age_s": age,
                "direction": int(direction),
                "impulse_bps": float(impulse_bps),
                "entry_price": entry,
                "cost_pct": cost_by_ts.get(int(ts[i]), float("nan")),
            }
            for h in horizons:
                end = i + h
                if end >= n:
                    row[f"mfe_{h}"] = None
                    row[f"ret_{h}"] = None
                    continue
                hi = float(np.nanmax(high[i + 1: end + 1]))
                lo = float(np.nanmin(low[i + 1: end + 1]))
                if direction > 0:
                    fav = (hi - entry) / entry * 100.0
                else:
                    fav = (entry - lo) / entry * 100.0
                row[f"mfe_{h}"] = max(fav, 0.0)
                row[f"ret_{h}"] = int(direction) * (float(close[end]) - entry) / entry * 100.0
            rows.append(row)
    return pl.DataFrame(rows)


def ceiling_table(entries: pl.DataFrame, params: Params) -> pl.DataFrame:
    """(検出窓 × 反応遅延) ごとの天井。コストと比べられる形で出す。"""
    if entries.height == 0:
        return pl.DataFrame(schema={"window_s": pl.Int64, "impulse_age_s": pl.Int64})
    h = int(params.get_path("viability.horizon"))
    required = float(params.get_path("viability.mfe_multiple_required"))
    mfe, ret = f"mfe_{h}", f"ret_{h}"
    if mfe not in entries.columns:
        raise KeyError(f"horizon {h}s not measured; got {entries.columns}")
    out = (
        entries.drop_nulls([mfe, ret])
        .group_by(["window_s", "sigma_mult", "impulse_age_s"]
                  if "sigma_mult" in entries.columns else ["window_s", "impulse_age_s"])
        .agg(
            pl.len().alias("n"),
            pl.col("impulse_bps").median().alias("median_impulse_bps"),
            (pl.col(mfe) * 100.0).median().alias("median_mfe_bps"),
            (pl.col(mfe) * 100.0).quantile(0.75).alias("p75_mfe_bps"),
            (pl.col(ret) * 100.0).median().alias("median_ret_bps"),
            (pl.col("cost_pct") * 100.0).median().alias("median_cost_bps"),
            ((pl.col(ret) - pl.col("cost_pct")) > 0).mean().alias("win_rate_after_cost"),
            ((pl.col(ret) - pl.col("cost_pct")) * 100.0).mean().alias("mean_net_bps"),
        )
        .sort(["window_s", "impulse_age_s"])
    )
    return out.with_columns(
        (pl.col("median_mfe_bps") / pl.col("median_cost_bps")).alias("mfe_cost_multiple"),
        pl.lit(required).alias("required_multiple"),
    ).with_columns(
        (pl.col("mfe_cost_multiple") > required).alias("ceiling_clears_bar")
    )


def verdict(table: pl.DataFrame) -> dict:
    """天井がどこかで基準を超えるか。超えなければ検出器の速度改善は無意味。

    §4.1 の基準は MFE ベースであり、**必要条件であって十分条件ではない**。
    MFE は「最良の瞬間に降りられたら取れた幅」なので、MFE が基準を超えていても
    実際の決済リターンが負ということは普通に起きる。誤読を防ぐため、
    純リターンが正になる組み合わせが存在するかどうかも併記する。
    """
    if table.height == 0:
        return {"verdict": "NO_IMPULSES", "best": None}
    best = table.sort("mfe_cost_multiple", descending=True, nulls_last=True).row(0, named=True)
    clears = bool(table["ceiling_clears_bar"].any())
    net_positive = bool((table["mean_net_bps"] > 0).any())
    best_net = table.sort("mean_net_bps", descending=True, nulls_last=True).row(0, named=True)
    return {
        "verdict": "CEILING_CLEARS_BAR" if clears else "CEILING_BELOW_BAR",
        "meaning": (
            "MFE 基準を超えた組み合わせがある。**勝てる証拠ではない**（MFE は最良の"
            "瞬間に降りられた場合の幅）。実時間検出器を検討してよい、というだけ。"
            if clears else
            "完璧な予言者でも MFE がコスト基準に届かない。検出器を速くしても勝てない。"
        ),
        "net_positive_anywhere": net_positive,
        "net_note": (
            "固定時間決済の純リターンが正になる (窓, 年齢) が存在する。"
            if net_positive else
            "**どの (窓, 年齢) でも固定時間決済の純リターンは負**。"
            "MFE 基準を超えていても、実際に取り切れていない。"
        ),
        "best_net_impulse_age_s": best_net["impulse_age_s"],
        "best_net_bps": best_net["mean_net_bps"],
        "best_window_s": best["window_s"],
        "best_impulse_age_s": best["impulse_age_s"],
        "best_median_mfe_bps": best["median_mfe_bps"],
        "median_cost_bps": best["median_cost_bps"],
        "best_mfe_cost_multiple": best["mfe_cost_multiple"],
        "required_multiple": best["required_multiple"],
        "n": best["n"],
    }
