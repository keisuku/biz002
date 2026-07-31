"""コマンドライン入口。

Phase 1:
    python -m src.cli discover      --symbol BTCUSDT
    python -m src.cli download      --symbol BTCUSDT --start 2023-08-01 --end 2026-07-29
    python -m src.cli bars          --symbol BTCUSDT --start ... --end ...
    python -m src.cli verify-flags  --symbol BTCUSDT --start ... --end ...
Phase 2:
    python -m src.cli refs          --symbol BTCUSDT --start ... --end ...
    python -m src.cli events        --symbol BTCUSDT --start ... --end ...
    python -m src.cli funnel        --symbol BTCUSDT --start ... --end ...   # 発火条件の律速診断
    python -m src.cli impulse-scan  --symbol BTCUSDT --start ... --end ...   # 初動エントリの天井（上限値）
    python -m src.cli ride          --symbol BTCUSDT --start ... --end ...   # 乗って放置する手法を R 倍率で測る
    python -m src.cli analyze       --symbol BTCUSDT            # §4.1〜4.5
    python -m src.cli latency       --symbol BTCUSDT            # 手動執行の 0〜20 秒遅延
    python -m src.cli validate      --symbol BTCUSDT --start ... --end ...   # §6
検証用:
    python -m src.cli synth         --symbol SYNTH --start ... --end ... --profile strong|null|realistic
    python -m src.cli trials        --symbol BTCUSDT --n 12 --note "手動でしきい値を調整した回数"
"""

from __future__ import annotations

import argparse
import json
import logging

from datetime import date, datetime
from pathlib import Path

import numpy as np
import polars as pl

from . import exits as exits_mod
from . import outcomes as out_mod
from . import pipeline as pipe
from .analysis import heatmap, latency, stop_fill, validation, viability
from .config import load_params, load_symbols, resolve_dir
from .data import download as dl
from .data import synthetic, to_seconds, verify_flags

log = logging.getLogger("cli")


def _date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _report_dir(params, symbol: str) -> Path:
    p = resolve_dir(params, "reports_dir") / symbol
    p.mkdir(parents=True, exist_ok=True)
    return p


def _trials_log(params, symbol: str) -> validation.TrialsLog:
    """試行回数ログは銘柄ごとに持つ（§6.2 / §9）。手動調整もここに加算する。"""
    p = Path(params.get_path("validation.trials_log"))
    return validation.TrialsLog(p if p.is_absolute() else _report_dir(params, symbol) / p)


def _days(params, args) -> list[date]:
    start = args.start or _date(params.get_path("data.start_date"))
    end = args.end or _date(params.get_path("data.end_date"))
    return pipe.day_list(start, end)


# --------------------------------------------------------------------------------------
def cmd_discover(args, params) -> None:
    layout = dl.discover_layout(params, args.symbol)
    out = resolve_dir(params, "reports_dir")
    out.mkdir(parents=True, exist_ok=True)
    (out / "data_layout.json").write_text(layout.to_json(), encoding="utf-8")
    print(layout.to_json())


def cmd_download(args, params) -> None:
    layout = dl.discover_layout(params, args.symbol)
    (resolve_dir(params, "reports_dir")).mkdir(parents=True, exist_ok=True)
    (resolve_dir(params, "reports_dir") / "data_layout.json").write_text(layout.to_json(), encoding="utf-8")
    syms = load_symbols()
    datasets = args.datasets or (syms["datasets"]["required"] + syms["datasets"]["recommended"])
    days = _days(params, args)
    for wanted in datasets:
        ds = dl.resolve_dataset_name(layout, wanted)
        interval = syms["klines_interval"] if ds.lower().startswith("kline") else None
        res = dl.fetch_range(params, layout, ds, [args.symbol], days[0], days[-1], interval)
        path = _report_dir(params, args.symbol) / f"download_{ds}.csv"
        res.write_csv(path)
        missing = res.filter(pl.col("status") == "missing")
        print(f"[{ds}] ok={res.filter(pl.col('status') == 'ok').height} "
              f"cached={res.filter(pl.col('status') == 'cached').height} missing={missing.height} -> {path}")


def cmd_bars(args, params) -> None:
    rows = [to_seconds.build_day(params, args.symbol, d, overwrite=args.overwrite)
            for d in _days(params, args)]
    rep = pl.DataFrame(rows)
    out = _report_dir(params, args.symbol)
    rep.write_csv(out / "bars_build_report.csv")
    missing = rep.filter(pl.col("status") == "missing_raw")
    gaps = rep.filter(pl.col("long_gaps") > 0) if "long_gaps" in rep.columns else rep.head(0)
    summary = {
        "symbol": args.symbol,
        "days": rep.height,
        "days_ok": int((rep["status"] == "ok").sum()),
        "days_missing_raw": missing.height,
        "missing_dates": missing["date"].to_list(),
        "days_with_long_gaps": gaps.height,
        "total_gap_seconds": int(rep["gap_seconds"].sum()) if "gap_seconds" in rep.columns else 0,
    }
    (out / "bars_missing_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def cmd_verify_flags(args, params) -> None:
    days = _days(params, args)
    frames = [pl.read_parquet(to_seconds.bars_path(params, args.symbol, d))
              for d in days if to_seconds.bars_path(params, args.symbol, d).exists()]
    if not frames:
        raise SystemExit("no bars found; run `bars` first")
    bars = pl.concat(frames).sort("ts")
    res = verify_flags.check_bars(bars)
    res["symbol"] = args.symbol
    res["days"] = len(frames)
    verify_flags.write_report(res, _report_dir(params, args.symbol))
    print(json.dumps(res, indent=2))


def cmd_refs(args, params) -> None:
    mr = pipe.build_refs(params, args.symbol, _days(params, args))
    print(f"minute rv rows: {mr.height} -> {pipe.refs_path(params, args.symbol)}")


def cmd_events(args, params) -> None:
    ledger, _ = pipe.build_events(params, args.symbol, _days(params, args), with_exits=False)
    pipe.save_ledger(params, args.symbol, ledger, pl.DataFrame())
    out = _report_dir(params, args.symbol)
    summary = {
        "symbol": args.symbol,
        "n_events": ledger.height,
        "thresholds": dict(params["thresholds"]),
        "first_event": int(ledger["event_ts"].min()) if ledger.height else None,
        "last_event": int(ledger["event_ts"].max()) if ledger.height else None,
        "long_events": int((ledger["direction"] == 1).sum()) if ledger.height else 0,
        "short_events": int((ledger["direction"] == -1).sum()) if ledger.height else 0,
        "ledger_path": str(pipe.ledger_path(params, args.symbol)),
    }
    (out / "events_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def cmd_funnel(args, params) -> None:
    """発火条件のどれが律速かを見る診断。閾値は一切変更しない（§9 の試行回数に数えない）。"""
    from .analysis import funnel as funnel_mod

    symbol = args.symbol
    merged = pipe.build_funnel(params, symbol, _days(params, args))
    table = funnel_mod.funnel_table(merged, params)
    dist = funnel_mod.ratio_distribution(merged, params)
    out = _report_dir(params, symbol)
    table.write_csv(out / "funnel.csv")
    dist.write_csv(out / "funnel_ratio_distribution.csv")
    summary = {
        "symbol": symbol,
        "seconds_evaluated": merged["n_data_ok"],
        "seconds_calm": merged["n_calm_ok"],
        "seconds_passing_all_conditions": merged["n_all"],
        "thresholds": dict(params["thresholds"]),
    }
    (out / "funnel_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n[条件別] binding_factor が大きいほど、その条件が件数を絞っている")
    print(table)
    print("\n[凪の秒における比率分布] threshold_percentile が 100 に近いほど厳しい閾値")
    print(dist)


def cmd_impulse_scan(args, params) -> None:
    """初動エントリの天井を測る。**これはバックテストではない**（開始時刻に未来を使う）。

    天井がコスト基準に届かなければ、検出器をいくら速くしても勝てない。
    その判定だけを目的にしている。
    """
    from .analysis import impulse as imp

    symbol = args.symbol
    entries = pipe.build_impulse_scan(params, symbol, _days(params, args))
    out = _report_dir(params, symbol)
    if entries.height == 0:
        print("初動が 1 件もラベルされなかった。期間かデータを確認すること。")
        return
    table = imp.ceiling_table(entries, params)
    v = imp.verdict(table)
    table.write_csv(out / "impulse_ceiling.csv")
    (out / "impulse_ceiling_verdict.json").write_text(
        json.dumps(v, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print("※ 初動の開始時刻に未来を使った**上限値**であり、実行可能な戦略ではない。")
    print(json.dumps(v, indent=2, ensure_ascii=False, default=str))
    compact = table.select(
        "window_s", "sigma_mult", "impulse_age_s", "n",
        pl.col("median_impulse_bps").round(2),
        pl.col("median_mfe_bps").round(2),
        pl.col("median_cost_bps").round(2),
        pl.col("mfe_cost_multiple").round(2),
        pl.col("mean_net_bps").round(2),
        pl.col("win_rate_after_cost").round(3),
        "ceiling_clears_bar",
    ).sort(["window_s", "impulse_age_s"])
    print("\n初動年齢 = 検出窓 + 通知遅延 + 人間の反応。age=0 は不可能な理想値（天井）。")
    with pl.Config(tbl_rows=60, tbl_cols=15, fmt_str_lengths=30):
        print(compact)


def cmd_ride(args, params) -> None:
    """特大の動きに乗ってストップを置き放置する手法を、R 倍率で測る。

    指示書のスキャル前提（60 秒・MFE 中央値・凪条件）とは別の手法として扱う。
    """
    from .analysis import trend_ride as ride

    symbol = args.symbol
    rides = pipe.build_trend_rides(params, symbol, _days(params, args),
                                   benchmark=not args.no_benchmark)
    out = _report_dir(params, symbol)
    if rides.height == 0:
        print("初動が 1 件もラベルされなかった。期間かデータを確認すること。")
        return
    by = ["window_s", "sigma_mult", "max_hold_s"]
    summary = ride.summarize(rides, params, by=by)
    strata = ride.stratify(rides, params)
    v = ride.verdict(summary, params)

    rides.write_parquet(out / "trend_rides.parquet")
    summary.write_csv(out / "trend_ride_summary.csv")
    strata.write_csv(out / "trend_ride_strata.csv")
    (out / "trend_ride_verdict.json").write_text(
        json.dumps(v, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    # --- 決定的な 2 つの検査 -------------------------------------------------------
    sig = rides.filter(pl.col("source") == "signal") if "source" in rides.columns else rides
    rob = ride.robustness(sig, int(params.get_path("trend_ride.robustness_block_seconds")))
    rob_v = ride.robustness_verdict(rob)
    rob.write_csv(out / "trend_ride_robustness.csv")
    (out / "trend_ride_robustness.json").write_text(
        json.dumps(rob_v, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    if "source" in rides.columns:
        cmp_tbl = ride.signal_vs_random(rides, params, by=by)
        cmp_tbl.write_csv(out / "trend_ride_signal_vs_random.csv")
        print("=" * 72)
        print("【検査1】シグナル vs ランダム時刻（同じ局面・同じ方向・同じストップ幅）")
        print("  差が無ければ、勝っていたのは局面であってシグナルではない。")
        print("=" * 72)
        with pl.Config(tbl_rows=60, tbl_cols=12):
            print(cmp_tbl.select([c for c in ("window_s", "sigma_mult", "max_hold_s",
                                              "source", "n", "win_rate", "mean_r",
                                              "total_r", "p_ge_3r")
                                  if c in cmp_tbl.columns]))
    print("=" * 72)
    print("【検査2】最大寄与ブロックを 1 つ抜いたときの平均 R")
    print("  1 窓抜いてプラスが消えるなら、実効サンプル数はほぼ 1。")
    print("=" * 72)
    print(json.dumps(rob_v, indent=2, ensure_ascii=False, default=str))

    print("R 倍率 = 損切り幅 1 個ぶん。負け -1R、勝ちは伸びたぶん。")
    print("**中央値ではなく平均 R と右の裾（+3R 以上の割合）を見ること。**")
    print(json.dumps(v, indent=2, ensure_ascii=False, default=str))
    cols = ["window_s", "sigma_mult", "max_hold_s", "n", "win_rate", "mean_r",
            "median_r", "total_r", "profit_factor", "p_ge_3r", "max_r", "stop_rate"]
    with pl.Config(tbl_rows=60, tbl_cols=15):
        print(summary.select([c for c in cols if c in summary.columns]))
        print("\n[層別] 順張り/逆張り・ボラ局面・売買方向")
        scols = ["stratum", "max_hold_s", "n", "win_rate", "mean_r", "total_r",
                 "profit_factor", "p_ge_3r"]
        print(strata.select([c for c in scols if c in strata.columns]))


def cmd_analyze(args, params) -> None:
    symbol = args.symbol
    out = _report_dir(params, symbol)
    ledger = pl.read_parquet(pipe.ledger_path(params, symbol))

    # §4.1 生死判定 ------------------------------------------------------------------
    verdict = viability.viability_check(ledger, params)
    horizons = params.get_path("outcomes.horizons")
    mfe = viability.mfe_distribution(ledger, horizons)
    # §4.2 保有時間 ------------------------------------------------------------------
    hexp = viability.horizon_expectancy(ledger, params)
    best_h = int(hexp.row(0, named=True)["horizon_s"]) if hexp.height else int(
        params.get_path("validation.sweep_horizon"))
    verdict["best_fixed_horizon_s"] = best_h
    viability.write_report(verdict, {"mfe_distribution": mfe, "horizon_expectancy": hexp}, out)
    print(json.dumps(verdict, indent=2, ensure_ascii=False))

    if verdict["verdict"] != "PASS" and not args.force:
        print("\n生死判定を通過していない。§9 の指示どおりここで停止する。"
              "\nパラメータ探索を続けると過剰適合するため、続行するには --force を明示すること。")
        return

    # §4.3 時間帯 --------------------------------------------------------------------
    mr = None
    try:
        mr = pipe.load_refs(params, symbol)
    except FileNotFoundError:
        pass
    rv_col = f"rv_{int(params.get_path('thresholds.calm_window'))}"
    hm = heatmap.build_all(ledger, params, f"ret_{best_h}", out, mr, rv_col if mr is not None else None)
    print(f"heatmaps: {hm}")

    # §4.4 出口ルール ----------------------------------------------------------------
    sf = stop_fill.TickStopFill(params) if args.tick_stops else None
    sim = pipe.simulate_exits_for_ledger(params, symbol, ledger, best_h, stop_fill=sf)
    if sim.height:
        min_wins = int(params.get_path("validation.min_wins"))
        table = exits_mod.summarize(sim, min_wins)
        table.write_csv(out / "exit_rules.csv")
        baseline = f"E4_{best_h}s"
        cmp_tbl = exits_mod.compare_to_baseline(sim, baseline)
        cmp_tbl.write_csv(out / "exit_vs_baseline.csv")
        # §4.5 損失側の実測
        stop_fill.realized_stop_slippage(sim, ledger, params).write_csv(out / "stop_slippage.csv")
        sim.write_parquet(pipe.ledger_path(params, symbol).with_name(f"{symbol}_exits.parquet"))
        print(table.head(20))
        print(cmp_tbl.head(20))

    # §6.4 コスト感度 ----------------------------------------------------------------
    mults = params.get_path("costs.sensitivity_multipliers")
    by_mult = {float(m): out_mod.add_costs(ledger, params, multiplier=float(m)) for m in mults}
    viability.cost_sensitivity(by_mult, params).write_csv(out / "cost_sensitivity.csv")
    print(f"reports written to {out}")


def cmd_latency(args, params) -> None:
    """Measure the 7s-vs-20s manual execution problem without tuning signals."""
    symbol = args.symbol
    ledger = pl.read_parquet(pipe.ledger_path(params, symbol))
    if ledger.height == 0:
        raise SystemExit("event ledger is empty")
    delays = [int(v) for v in params.get_path("latency.reaction_delay_seconds")]
    horizon = int(params.get_path("latency.diagnostic_horizon_seconds"))
    start_ts = int(ledger["event_ts"].min())
    end_ts = int(ledger["event_ts"].max()) + max(delays) + horizon + 1
    seconds = to_seconds.load_seconds(params, symbol, start_ts, end_ts)
    rows = latency.simulate(ledger, seconds, params)
    summary = latency.summarize(rows)
    out = _report_dir(params, symbol)
    rows.write_parquet(out / "latency_events.parquet")
    summary.write_csv(out / "latency_summary.csv")
    print(summary)


def cmd_validate(args, params) -> None:
    symbol = args.symbol
    out = _report_dir(params, symbol)
    days = _days(params, args)
    t0 = to_seconds.day_start_ts(days[0])
    t1 = to_seconds.day_start_ts(days[-1]) + to_seconds.SECONDS_PER_DAY
    folds = validation.make_folds(t0, t1, int(params.get_path("validation.n_folds")))

    combos = pipe.grid_combinations(params)
    trials = _trials_log(params, symbol)
    trials.add(len(combos), f"grid sweep {symbol} {days[0]}..{days[-1]}")

    sweep_tbl, _ = pipe.sweep(params, symbol, days, folds)
    if sweep_tbl.height == 0:
        raise SystemExit("sweep produced no events")
    sweep_tbl.write_csv(out / "sweep.csv")

    overall = sweep_tbl.filter(pl.col("fold") == -1)
    per_fold = sweep_tbl.filter(pl.col("fold") >= 0)

    # §6.1 ウォークフォワード
    wf = validation.walk_forward(per_fold, pipe.PARAM_COLS, params)
    wf.write_csv(out / "walk_forward.csv")

    # §6.5 サンプル数ゲート → 生き残りから最良点
    min_wins = int(params.get_path("validation.min_wins"))
    gated = overall.filter(pl.col("n_wins") >= min_wins)
    metric = params.get_path("validation.selection_metric")
    if gated.height == 0:
        result = {"status": "NO_COMBO_WITH_ENOUGH_WINS", "min_wins": min_wins,
                  "n_trials": trials.total, "n_combos": len(combos)}
        (out / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return
    best = gated.sort(metric, descending=True).row(0, named=True)

    # §6.3 プラトー
    plateau = validation.plateau_check(overall, {k: list(params.get_path(f"grid.{k}"))
                                                 for k in pipe.PARAM_COLS
                                                 if k in ("calm_window", "calm_percentile",
                                                          "sigma", "tick", "ofi", "impact")},
                                       best, params)

    # §6.2 DSR / §6.6 ブロックブートストラップ（選ばれた組み合わせのイベント別リターン）
    key = {k: best[k] for k in pipe.PARAM_COLS}
    _, kept = pipe.sweep(params, symbol, days, folds, keep_returns_for=key)
    trials.add(1, f"re-run for selected combo returns {symbol}")
    dsr = boot = None
    if kept.height:
        r = kept["net_pct"].to_numpy()
        sr_var = float(np.nanvar(overall[metric].to_numpy() / overall["sd_net_pct"].to_numpy(), ddof=1))
        dsr = validation.deflated_sharpe(r, trials.total, sr_var)
        bs = params.get_path("validation.bootstrap")
        boot = validation.block_bootstrap_ci(
            r, validation.day_blocks(kept["event_ts"].to_numpy(), int(bs["block_seconds"])),
            int(bs["n_resamples"]), float(bs["ci"]))

    result = {
        "symbol": symbol,
        "period": [days[0].isoformat(), days[-1].isoformat()],
        "n_grid_combinations": len(combos),
        "n_trials_total_logged": trials.total,
        "selection_metric": metric,
        "selected": key,
        "selected_stats": {k: best[k] for k in ("n", "n_wins", "net_mean_pct", "win_rate", "t_stat")},
        "walk_forward_oos_mean": float(np.nanmean(
            [r for r in wf["oos_metric"].to_list() if r is not None])) if wf.height else None,
        "walk_forward_folds_positive": int(sum(
            1 for r in wf["oos_metric"].to_list() if r is not None and r > 0)),
        "sample_gate": validation.sample_gate(best["n_wins"], params),
        "plateau": plateau,
        "deflated_sharpe": dsr,
        "block_bootstrap": boot,
    }
    (out / "validation.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str),
                                         encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


def cmd_synth(args, params) -> None:
    cfg = synthetic.make_config(args.profile, args.seed)
    print(synthetic.PROFILE_DOC)
    truths = []
    for i, d in enumerate(_days(params, args)):
        trades, truth = synthetic.generate_day(d, cfg, seed_offset=i)
        p = dl.raw_path(params, "aggTrades", args.symbol, d)
        p.parent.mkdir(parents=True, exist_ok=True)
        trades.write_parquet(p)
        if truth.height:
            truths.append(truth.with_columns(pl.lit(d.isoformat()).alias("date")))
        print(f"{d}: {trades.height} trades, {truth.height} injected ignitions -> {p}")
    if truths:
        tp = resolve_dir(params, "raw_dir") / f"{args.symbol}_truth.parquet"
        pl.concat(truths).write_parquet(tp)
        print(f"truth ledger -> {tp}")


def cmd_trials(args, params) -> None:
    data = _trials_log(params, args.symbol).add(args.n, args.note)
    print(json.dumps(data, indent=2, ensure_ascii=False))


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(prog="momentum-ignition")
    ap.add_argument("--params", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common(p, with_dates=True):
        p.add_argument("--symbol", required=True)
        if with_dates:
            p.add_argument("--start", type=_date, default=None)
            p.add_argument("--end", type=_date, default=None)

    p = sub.add_parser("discover"); add_common(p, False); p.set_defaults(fn=cmd_discover)
    p = sub.add_parser("download"); add_common(p); p.add_argument("--datasets", nargs="*"); p.set_defaults(fn=cmd_download)
    p = sub.add_parser("bars"); add_common(p); p.add_argument("--overwrite", action="store_true"); p.set_defaults(fn=cmd_bars)
    p = sub.add_parser("verify-flags"); add_common(p); p.set_defaults(fn=cmd_verify_flags)
    p = sub.add_parser("refs"); add_common(p); p.set_defaults(fn=cmd_refs)
    p = sub.add_parser("events"); add_common(p); p.set_defaults(fn=cmd_events)
    p = sub.add_parser("funnel"); add_common(p); p.set_defaults(fn=cmd_funnel)
    p = sub.add_parser("impulse-scan"); add_common(p); p.set_defaults(fn=cmd_impulse_scan)
    p = sub.add_parser("ride"); add_common(p)
    p.add_argument("--no-benchmark", action="store_true",
                   help="ランダム時刻の対照群を作らない（非推奨）")
    p.set_defaults(fn=cmd_ride)
    p = sub.add_parser("analyze"); add_common(p)
    p.add_argument("--force", action="store_true", help="生死判定 FAIL でも続行する（推奨しない）")
    p.add_argument("--tick-stops", action="store_true", help="§4.5 の tick 実測でストップ約定を推定")
    p.set_defaults(fn=cmd_analyze)
    p = sub.add_parser("latency"); add_common(p, False); p.set_defaults(fn=cmd_latency)
    p = sub.add_parser("validate"); add_common(p); p.set_defaults(fn=cmd_validate)
    p = sub.add_parser("synth"); add_common(p)
    p.add_argument("--profile", default="realistic", choices=sorted(synthetic.PROFILES))
    p.add_argument("--seed", type=int, default=None)
    p.set_defaults(fn=cmd_synth)
    p = sub.add_parser("trials"); p.add_argument("--symbol", required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--note", required=True); p.set_defaults(fn=cmd_trials)

    args = ap.parse_args(argv)
    params = load_params(args.params)
    args.fn(args, params)


if __name__ == "__main__":
    main()
