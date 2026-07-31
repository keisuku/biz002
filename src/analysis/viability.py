"""§4.1 生死判定 と §4.2 保有時間の最適分布。

判定基準（指示書のまま。ここを緩めてはいけない）:
    mfe_60 の中位値 > 3 × (往復手数料 + 推定スリッページ) でなければ、
    この手法はこの検出器では成立しない。

不通過の場合はパラメータ探索に進む前に必ず報告する。探索を続けると過剰適合する。
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from ..config import Params


def mfe_distribution(events: pl.DataFrame, horizons) -> pl.DataFrame:
    rows = []
    for h in horizons:
        col = f"mfe_{h}"
        if col not in events.columns:
            continue
        s = events[col].drop_nulls()
        if s.len() == 0:
            continue
        rows.append({
            "horizon_s": h,
            "n": s.len(),
            "mfe_median_pct": float(s.median()),
            "mfe_p75_pct": float(s.quantile(0.75)),
            "mfe_p90_pct": float(s.quantile(0.90)),
            "mae_median_pct": float(events[f"mae_{h}"].drop_nulls().median() or 0.0),
        })
    return pl.DataFrame(rows)


def horizon_expectancy(events: pl.DataFrame, params: Params) -> pl.DataFrame:
    """§4.2 「H 秒後決済」の期待値。素の保有時間を決めるための表。"""
    horizons = list(params.get_path("outcomes.horizons"))
    rows = []
    for h in horizons:
        col = f"ret_{h}"
        if col not in events.columns:
            continue
        d = events.select(
            pl.col(col).alias("gross"),
            (pl.col(col) - pl.col("cost_pct")).alias("net"),
        ).drop_nulls()
        if d.height == 0:
            continue
        net = d["net"]
        sd = float(net.std()) if net.len() > 1 else float("nan")
        rows.append({
            "horizon_s": h,
            "n": net.len(),
            "gross_mean_pct": float(d["gross"].mean()),
            "net_mean_pct": float(net.mean()),
            "net_median_pct": float(net.median()),
            "win_rate": float((net > 0).mean()),
            "n_wins": int((net > 0).sum()),
            "sd_pct": sd,
            "t_stat": float(net.mean() / sd * (net.len() ** 0.5)) if sd and sd > 0 else float("nan"),
        })
    return pl.DataFrame(rows).sort("net_mean_pct", descending=True)


def viability_check(events: pl.DataFrame, params: Params) -> dict:
    h = int(params.get_path("viability.horizon"))
    required = float(params.get_path("viability.mfe_multiple_required"))
    col = f"mfe_{h}"
    n = events.height
    if n == 0 or col not in events.columns or events[col].drop_nulls().len() == 0:
        return {"verdict": "NO_EVENTS", "n_events": n, "horizon_s": h,
                "mfe_median_pct": None, "cost_pct": None, "required_multiple": required,
                "actual_multiple": None}
    mfe_med = float(events[col].drop_nulls().median())
    cost_med = float(events["cost_pct"].drop_nulls().median())
    fee = float(events["fee_pct"].drop_nulls().median())
    slip = float(events["slippage_est_pct"].drop_nulls().median())
    ratio = mfe_med / cost_med if cost_med > 0 else float("inf")
    return {
        "verdict": "PASS" if ratio > required else "FAIL",
        "n_events": n,
        "horizon_s": h,
        "mfe_median_pct": mfe_med,
        "mfe_p75_pct": float(events[col].drop_nulls().quantile(0.75)),
        "mfe_p90_pct": float(events[col].drop_nulls().quantile(0.90)),
        "fee_pct": fee,
        "slippage_est_pct_one_way": slip,
        "cost_pct": cost_med,
        "required_multiple": required,
        "actual_multiple": ratio,
    }


def cost_sensitivity(events_by_multiplier: dict[float, pl.DataFrame], params: Params) -> pl.DataFrame:
    """§6.4 コストを 1.5 / 2 倍にしても正の期待値が残るか。"""
    rows = []
    for m, ev in sorted(events_by_multiplier.items()):
        tbl = horizon_expectancy(ev, params)
        if tbl.height == 0:
            continue
        best = tbl.row(0, named=True)
        rows.append({
            "cost_multiplier": m,
            "best_horizon_s": best["horizon_s"],
            "net_mean_pct": best["net_mean_pct"],
            "win_rate": best["win_rate"],
            "n": best["n"],
            "positive": best["net_mean_pct"] > 0,
        })
    return pl.DataFrame(rows)


def write_report(result: dict, tables: dict[str, pl.DataFrame], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "viability.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    for name, tbl in tables.items():
        tbl.write_csv(out_dir / f"{name}.csv")
