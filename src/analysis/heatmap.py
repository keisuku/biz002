"""§4.3 時間帯分析。

必ず 2 枚作る:
    1. 発火頻度      hour_jst × 曜日 のイベント件数
    2. 実効期待値    hour_jst × 曜日 の（最適 H 決済リターン − 手数料 − スリッページ）

**1 枚目だけで判断してはいけない。** 板の薄い時間帯（JST 早朝・週末）は発火頻度と
値幅が大きく出るが、スリッページで消える可能性が高い。2 枚目が結論。

出力は CSV（常時）と PNG（matplotlib がある場合）。対話的な UI は作らない。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from ..config import Params

DOW_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _pivot(events: pl.DataFrame, value: pl.Expr, name: str) -> pl.DataFrame:
    grid = pl.DataFrame(
        {"hour_jst": np.repeat(np.arange(24), 7), "dow_jst": np.tile(np.arange(1, 8), 24)}
    ).with_columns(pl.col("hour_jst").cast(pl.Int8), pl.col("dow_jst").cast(pl.Int8))
    if events.height == 0:
        return grid.with_columns(pl.lit(None, dtype=pl.Float64).alias(name))
    agg = events.group_by(["hour_jst", "dow_jst"]).agg(value.alias(name))
    return grid.join(agg, on=["hour_jst", "dow_jst"], how="left").sort(["hour_jst", "dow_jst"])


def frequency_table(events: pl.DataFrame) -> pl.DataFrame:
    return _pivot(events, pl.len().cast(pl.Float64), "n_events").with_columns(
        pl.col("n_events").fill_null(0.0)
    )


def expectancy_table(events: pl.DataFrame, ret_col: str) -> pl.DataFrame:
    """実効期待値 = ret − 手数料 − スリッページ（cost_pct に両方含まれる）。"""
    net = (pl.col(ret_col) - pl.col("cost_pct")).mean()
    tbl = _pivot(events, net, "net_mean_pct")
    counts = frequency_table(events)
    return tbl.join(counts, on=["hour_jst", "dow_jst"], how="left")


def to_matrix(tbl: pl.DataFrame, value_col: str) -> np.ndarray:
    m = np.full((24, 7), np.nan)
    for row in tbl.iter_rows(named=True):
        v = row[value_col]
        m[int(row["hour_jst"]), int(row["dow_jst"]) - 1] = np.nan if v is None else float(v)
    return m


def save_heatmap_png(matrix: np.ndarray, title: str, path: Path, cmap: str,
                     center_zero: bool = False) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001 - 描画は任意。CSV が本体。
        return False
    fig, ax = plt.subplots(figsize=(6, 9))
    kw = {}
    if center_zero:
        lim = np.nanmax(np.abs(matrix)) if np.isfinite(matrix).any() else 1.0
        kw = {"vmin": -lim, "vmax": lim}
    im = ax.imshow(matrix, aspect="auto", origin="lower", cmap=cmap, **kw)
    ax.set_xticks(range(7), DOW_LABELS)
    ax.set_yticks(range(0, 24, 2), [f"{h:02d}" for h in range(0, 24, 2)])
    ax.set_xlabel("day of week (JST)")
    ax.set_ylabel("hour (JST)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def session_split(events: pl.DataFrame, params: Params, ret_col: str) -> pl.DataFrame:
    sessions = params.get_path("analysis.sessions")
    rows = []
    for name, (lo, hi) in sessions.items():
        sub = events.filter((pl.col("hour_utc") >= lo) & (pl.col("hour_utc") < hi))
        rows.append(_stats_row(sub, ret_col, {"bucket": f"session_{name}"}))
    return pl.DataFrame(rows)


def _minute_of_day_utc(col: str = "event_ts") -> pl.Expr:
    return (pl.col(col) % 86400) // 60


def funding_window_split(events: pl.DataFrame, params: Params, ret_col: str) -> pl.DataFrame:
    hours = params.get_path("analysis.funding_hours_utc")
    win = int(params.get_path("analysis.funding_window_minutes"))
    mod = _minute_of_day_utc()
    near = None
    for h in hours:
        center = h * 60
        d = ((mod - center + 720) % 1440) - 720  # -720..719 の符号付き距離
        cond = d.abs() <= win
        near = cond if near is None else (near | cond)
    rows = [
        _stats_row(events.filter(near), ret_col, {"bucket": "funding_pm30m"}),
        _stats_row(events.filter(~near), ret_col, {"bucket": "funding_other"}),
    ]
    return pl.DataFrame(rows)


def us_open_split(events: pl.DataFrame, params: Params, ret_col: str) -> pl.DataFrame:
    center = float(params.get_path("analysis.us_open_utc_hour")) * 60
    win = int(params.get_path("analysis.us_open_window_minutes"))
    d = ((_minute_of_day_utc() - center + 720) % 1440) - 720
    rows = [
        _stats_row(events.filter(d.abs() <= win), ret_col, {"bucket": "us_open_window"}),
        _stats_row(events.filter(d.abs() > win), ret_col, {"bucket": "us_open_other"}),
    ]
    return pl.DataFrame(rows)


def high_vol_windows(minute_rv: pl.DataFrame, rv_col: str, params: Params) -> np.ndarray:
    """外部カレンダーの代替: ボラ上位 p% の分足時刻を返す（§4.3）。"""
    p = float(params.get_path("analysis.high_vol_percentile"))
    s = minute_rv[rv_col].drop_nulls()
    if s.len() == 0:
        return np.array([], dtype=np.int64)
    thr = float(s.quantile(p / 100.0))
    return minute_rv.filter(pl.col(rv_col) >= thr)["ts"].to_numpy()


def high_vol_split(events: pl.DataFrame, hv_ts: np.ndarray, params: Params,
                   ret_col: str) -> pl.DataFrame:
    win = int(params.get_path("analysis.high_vol_window_minutes")) * 60
    if events.height == 0 or hv_ts.size == 0:
        return pl.DataFrame([_stats_row(events, ret_col, {"bucket": "high_vol_window"}),
                             _stats_row(events, ret_col, {"bucket": "high_vol_other"})])
    ev = events["event_ts"].to_numpy()
    hv = np.sort(hv_ts)
    idx = np.searchsorted(hv, ev)
    left = np.clip(idx - 1, 0, len(hv) - 1)
    right = np.clip(idx, 0, len(hv) - 1)
    dist = np.minimum(np.abs(ev - hv[left]), np.abs(ev - hv[right]))
    mask = dist <= win
    return pl.DataFrame([
        _stats_row(events.filter(pl.Series(mask)), ret_col, {"bucket": "high_vol_window"}),
        _stats_row(events.filter(pl.Series(~mask)), ret_col, {"bucket": "high_vol_other"}),
    ])


def _stats_row(sub: pl.DataFrame, ret_col: str, extra: dict) -> dict:
    row = dict(extra)
    if sub.height == 0 or ret_col not in sub.columns:
        row.update({"n": 0, "net_mean_pct": None, "win_rate": None, "gross_mean_pct": None})
        return row
    net = (sub[ret_col] - sub["cost_pct"]).drop_nulls()
    row.update({
        "n": net.len(),
        "gross_mean_pct": float(sub[ret_col].drop_nulls().mean() or 0.0) if net.len() else None,
        "net_mean_pct": float(net.mean()) if net.len() else None,
        "win_rate": float((net > 0).mean()) if net.len() else None,
    })
    return row


def build_all(events: pl.DataFrame, params: Params, ret_col: str, out_dir: Path,
              minute_rv: pl.DataFrame | None = None, rv_col: str | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    freq = frequency_table(events)
    exp = expectancy_table(events, ret_col)
    freq.write_csv(out_dir / "heatmap_frequency.csv")
    exp.write_csv(out_dir / "heatmap_expectancy.csv")
    png1 = save_heatmap_png(to_matrix(freq, "n_events"), "Ignition count (JST)",
                            out_dir / "heatmap_frequency.png", "viridis")
    png2 = save_heatmap_png(to_matrix(exp, "net_mean_pct"),
                            f"Net expectancy % after costs ({ret_col}, JST)",
                            out_dir / "heatmap_expectancy.png", "RdBu_r", center_zero=True)

    strata = [session_split(events, params, ret_col),
              funding_window_split(events, params, ret_col),
              us_open_split(events, params, ret_col)]
    if minute_rv is not None and rv_col:
        strata.append(high_vol_split(events, high_vol_windows(minute_rv, rv_col, params),
                                     params, ret_col))
    strat = pl.concat(strata, how="diagonal_relaxed")
    strat.write_csv(out_dir / "strata.csv")
    return {"png_written": bool(png1 and png2), "strata_rows": strat.height}
