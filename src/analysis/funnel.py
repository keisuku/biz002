"""発火条件のファネル診断。

「イベントが 58 件と 1 件」のような差が出たとき、**4 条件のどれが効いて落ちているか**が
分からないと、次に何を疑えばよいか決められない。閾値を触って件数を合わせにいくのは
最悪の対処（結果を見てからの調整 = 試行回数の水増し）なので、その前にここを見る。

出すもの:

1. 各条件を**単独で**通過した秒数（他の条件は無視）
2. 各条件**だけを外した**ときに全条件を通過する秒数
   → これが「その条件がどれだけ効いているか」の直接の指標。
     ある条件を外した途端に件数が跳ね上がるなら、そこが律速。
3. 凪かつデータ健全な秒における各比率の分位点と、**設定した閾値が何パーセンタイルに
   当たるか**
   → 「impact 閾値 3.0 は BTC では上位 0.8%、ETH では上位 0.03% に当たる」のように、
     同じ数字が銘柄によって全く違う厳しさになっていることを可視化する。

この診断は閾値を変更しない。読むためだけのもの。
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ..config import Params

# 条件の評価順（累積 AND はこの順に適用する）
CONDITIONS = ("calm", "velocity", "tick", "ofi", "impact")


def condition_masks(feats: pl.DataFrame, params: Params) -> dict[str, pl.Series]:
    """§3.3 の各条件を個別のブール列として取り出す。"""
    th = params["thresholds"]
    k = int(params.get_path("features.velocity_primary_k"))
    return {
        "calm": feats["is_calm"].fill_null(False),
        "velocity": (feats[f"velocity_{k}"] >= float(th["sigma"])).fill_null(False),
        "tick": (feats["tick_ratio"] >= float(th["tick"])).fill_null(False),
        "ofi": (feats["ofi_ratio"] >= float(th["ofi"])).fill_null(False),
        "impact": (feats["impact_ratio"] >= float(th["impact"])).fill_null(False),
    }


RATIO_COLUMNS = {
    "velocity": "velocity_{k}",
    "tick": "tick_ratio",
    "ofi": "ofi_ratio",
    "impact": "impact_ratio",
}


def _ratio_column(name: str, params: Params) -> str:
    k = int(params.get_path("features.velocity_primary_k"))
    return RATIO_COLUMNS[name].format(k=k)


def count_chunk(feats: pl.DataFrame, params: Params, sample_rows: int = 50_000,
                seed: int = 0) -> dict:
    """1 チャンクぶんの集計。チャンクをまたいで足し合わせられる形で返す。"""
    ok = feats["data_ok"].fill_null(False)
    masks = condition_masks(feats, params)
    n_ok = int(ok.sum())

    alone = {c: int((masks[c] & ok).sum()) for c in CONDITIONS}

    all_mask = ok.clone()
    for c in CONDITIONS:
        all_mask = all_mask & masks[c]
    n_all = int(all_mask.sum())

    # 各条件「だけ」を外したときの通過数
    without = {}
    for skip in CONDITIONS:
        m = ok.clone()
        for c in CONDITIONS:
            if c != skip:
                m = m & masks[c]
        without[skip] = int(m.sum())

    # 累積 AND（条件を順に足していったときの残り数）
    cumulative = {}
    m = ok.clone()
    for c in CONDITIONS:
        m = m & masks[c]
        cumulative[c] = int(m.sum())

    # 凪かつ健全な秒の比率分布（分位点推定用に間引いて保持）
    calm_ok = ok & masks["calm"]
    cols = [_ratio_column(c, params) for c in ("velocity", "tick", "ofi", "impact")]
    sub = feats.filter(calm_ok).select([c for c in cols if c in feats.columns])
    if sub.height > sample_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(sub.height, sample_rows, replace=False)
        sub = sub[np.sort(idx)]

    return {
        "n_data_ok": n_ok,
        "n_calm_ok": int(calm_ok.sum()),
        "alone": alone,
        "without": without,
        "cumulative": cumulative,
        "n_all": n_all,
        "sample": sub,
    }


def merge_chunks(chunks: list[dict]) -> dict:
    if not chunks:
        return {"n_data_ok": 0, "n_calm_ok": 0, "n_all": 0, "alone": {}, "without": {},
                "cumulative": {}, "sample": pl.DataFrame()}
    out = {
        "n_data_ok": sum(c["n_data_ok"] for c in chunks),
        "n_calm_ok": sum(c["n_calm_ok"] for c in chunks),
        "n_all": sum(c["n_all"] for c in chunks),
        "alone": {k: sum(c["alone"].get(k, 0) for c in chunks) for k in CONDITIONS},
        "without": {k: sum(c["without"].get(k, 0) for c in chunks) for k in CONDITIONS},
        "cumulative": {k: sum(c["cumulative"].get(k, 0) for c in chunks) for k in CONDITIONS},
    }
    frames = [c["sample"] for c in chunks if c["sample"].height]
    out["sample"] = pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()
    return out


def funnel_table(merged: dict, params: Params) -> pl.DataFrame:
    th = params["thresholds"]
    thresholds = {"calm": float(th["calm_percentile"]), "velocity": float(th["sigma"]),
                  "tick": float(th["tick"]), "ofi": float(th["ofi"]),
                  "impact": float(th["impact"])}
    n_ok = max(merged["n_data_ok"], 1)
    n_all = merged["n_all"]
    rows = []
    for c in CONDITIONS:
        without = merged["without"].get(c, 0)
        rows.append({
            "condition": c,
            "threshold": thresholds[c],
            "pass_alone": merged["alone"].get(c, 0),
            "pass_alone_rate": merged["alone"].get(c, 0) / n_ok,
            "cumulative_pass": merged["cumulative"].get(c, 0),
            "pass_if_this_removed": without,
            # この条件を外すと何倍に増えるか。大きいほどここが律速。
            "binding_factor": (without / n_all) if n_all else float("inf") if without else 1.0,
        })
    return pl.DataFrame(rows).sort("binding_factor", descending=True)


def ratio_distribution(merged: dict, params: Params) -> pl.DataFrame:
    """凪かつ健全な秒における各比率の分位点と、設定閾値の立ち位置。"""
    sample: pl.DataFrame = merged["sample"]
    th = params["thresholds"]
    thresholds = {"velocity": float(th["sigma"]), "tick": float(th["tick"]),
                  "ofi": float(th["ofi"]), "impact": float(th["impact"])}
    rows = []
    for name, thr in thresholds.items():
        col = _ratio_column(name, params)
        if sample.height == 0 or col not in sample.columns:
            continue
        s = sample[col].drop_nulls()
        if s.len() == 0:
            continue
        rows.append({
            "ratio": name,
            "threshold": thr,
            "n_sampled": s.len(),
            "p50": float(s.quantile(0.50)),
            "p90": float(s.quantile(0.90)),
            "p99": float(s.quantile(0.99)),
            "p999": float(s.quantile(0.999)),
            "max": float(s.max()),
            # 閾値が分布の何パーセンタイルに当たるか（100 に近いほど厳しい）
            "threshold_percentile": float((s < thr).mean() * 100.0),
        })
    if not rows:
        return pl.DataFrame(schema={
            "ratio": pl.String, "threshold": pl.Float64, "n_sampled": pl.UInt32,
            "p50": pl.Float64, "p90": pl.Float64, "p99": pl.Float64, "p999": pl.Float64,
            "max": pl.Float64, "threshold_percentile": pl.Float64,
        })
    return pl.DataFrame(rows).sort("threshold_percentile", descending=True)
