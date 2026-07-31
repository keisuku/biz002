"""§6 検証の作法。違反したら結果は無効。

1. ウォークフォワード（時系列 9 分割・ランダム分割禁止）
2. 試行回数の記録と Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014)
3. プラトー確認（尖ったピークは棄却）
4. コスト感度（viability.cost_sensitivity）
5. サンプル数ゲート（勝ちトレード 30 件未満は結論を出さない）
6. ブロックブートストラップ信頼区間（自己相関があるため通常のブートストラップ不可）
7. ルックアヘッド検査（tests/test_lookahead.py）
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

from ..config import Params

EULER_GAMMA = 0.5772156649015329


# --------------------------------------------------------------------------------------
# 1. ウォークフォワード
# --------------------------------------------------------------------------------------
def make_folds(ts_min: int, ts_max: int, n_folds: int) -> list[tuple[int, int]]:
    """期間を時系列順に n_folds 等分する。シャッフルは絶対に行わない。"""
    if ts_max <= ts_min:
        raise ValueError("empty period")
    edges = np.linspace(ts_min, ts_max + 1, n_folds + 1).astype(np.int64)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n_folds)]


def assign_fold(ts: np.ndarray, folds: list[tuple[int, int]]) -> np.ndarray:
    out = np.full(len(ts), -1, dtype=np.int64)
    for i, (lo, hi) in enumerate(folds):
        out[(ts >= lo) & (ts < hi)] = i
    return out


def walk_forward(fold_stats: pl.DataFrame, param_cols: list[str], params: Params) -> pl.DataFrame:
    """訓練 fold で選んだパラメータを**凍結**して次 fold を評価する。

    fold_stats は 1 行 = (パラメータ組み合わせ × fold) の集計表で、
    少なくとも fold / n / n_wins / <selection_metric> 列を持つこと。
    """
    metric = str(params.get_path("validation.selection_metric"))
    min_wins = int(params.get_path("validation.min_wins"))
    n_folds = int(fold_stats["fold"].max() or 0) + 1
    rows = []
    for f in range(1, n_folds):
        train = fold_stats.filter(pl.col("fold") == f - 1).filter(pl.col("n_wins") >= min_wins)
        if train.height == 0:
            rows.append({"test_fold": f, "selected": None, "reason": "insufficient_train_wins",
                         "oos_metric": None, "oos_n": 0})
            continue
        best = train.sort(metric, descending=True).row(0, named=True)
        key = {c: best[c] for c in param_cols}
        test = fold_stats.filter(pl.col("fold") == f)
        for c, v in key.items():
            test = test.filter(pl.col(c) == v)
        if test.height == 0:
            rows.append({"test_fold": f, "selected": json.dumps(key), "reason": "no_test_rows",
                         "oos_metric": None, "oos_n": 0})
            continue
        t = test.row(0, named=True)
        rows.append({
            "test_fold": f,
            "selected": json.dumps(key),
            "reason": "ok",
            "train_metric": best[metric],
            "oos_metric": t[metric],
            "oos_n": t["n"],
            "oos_wins": t["n_wins"],
        })
    return pl.DataFrame(rows)


# --------------------------------------------------------------------------------------
# 2. 試行回数と Deflated Sharpe Ratio
# --------------------------------------------------------------------------------------
@dataclass
class TrialsLog:
    """探索した全組み合わせ数を永続化する。手動調整も必ずここに加算すること。"""

    path: Path

    def read(self) -> dict:
        if self.path.exists():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return {"total_trials": 0, "entries": []}

    def add(self, n: int, note: str) -> dict:
        data = self.read()
        data["total_trials"] = int(data.get("total_trials", 0)) + int(n)
        data["entries"].append({"n": int(n), "note": note})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return data

    @property
    def total(self) -> int:
        return int(self.read().get("total_trials", 0))


def sharpe(returns: np.ndarray) -> float:
    r = returns[np.isfinite(returns)]
    if r.size < 2:
        return float("nan")
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 0 else float("nan")


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """試行 N 回のもとで、真の SR=0 でも期待される最大 SR（SR0）。"""
    if n_trials < 2 or not math.isfinite(sr_variance) or sr_variance <= 0:
        return 0.0
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(sr_variance) * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)


def deflated_sharpe(returns: np.ndarray, n_trials: int, sr_variance: float) -> dict:
    """Bailey & Lopez de Prado (2014) の DSR。素の SR と必ず併記して報告する。"""
    r = returns[np.isfinite(returns)]
    t = r.size
    sr = sharpe(r)
    if t < 3 or not math.isfinite(sr):
        return {"sharpe": sr, "sr0": None, "dsr": None, "n_obs": int(t), "n_trials": int(n_trials)}
    skew = float(stats.skew(r, bias=False))
    kurt = float(stats.kurtosis(r, fisher=False, bias=False))
    sr0 = expected_max_sharpe(n_trials, sr_variance)
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    if denom <= 0:
        return {"sharpe": sr, "sr0": sr0, "dsr": None, "n_obs": int(t), "n_trials": int(n_trials),
                "note": "non-positive variance term"}
    z = (sr - sr0) * math.sqrt(t - 1) / math.sqrt(denom)
    return {
        "sharpe": sr,
        "sr0": sr0,
        "dsr": float(stats.norm.cdf(z)),
        "skew": skew,
        "kurtosis": kurt,
        "n_obs": int(t),
        "n_trials": int(n_trials),
    }


# --------------------------------------------------------------------------------------
# 3. プラトー確認
# --------------------------------------------------------------------------------------
def plateau_check(summary: pl.DataFrame, grid: dict[str, list], best: dict,
                  params: Params) -> dict:
    """最良点の格子近傍（各次元 ±1）が性能を保っているかを見る。"""
    metric = str(params.get_path("validation.plateau.metric"))
    min_frac = float(params.get_path("validation.plateau.min_neighbor_fraction"))
    ratio = float(params.get_path("validation.plateau.plateau_ratio"))
    best_val = float(best[metric])
    neighbors: list[dict] = []
    for name, values in grid.items():
        if name not in best:
            continue
        idx = values.index(best[name])
        for step in (-1, 1):
            j = idx + step
            if 0 <= j < len(values):
                key = {k: best[k] for k in grid if k in best}
                key[name] = values[j]
                neighbors.append(key)
    if not neighbors:
        return {"n_neighbors": 0, "passed": False, "reason": "no neighbors in grid"}
    ok = 0
    details = []
    for key in neighbors:
        sub = summary
        for c, v in key.items():
            sub = sub.filter(pl.col(c) == v)
        if sub.height == 0:
            details.append({"key": key, "metric": None, "ok": False})
            continue
        val = sub.row(0, named=True)[metric]
        good = val is not None and val > 0 and val >= ratio * best_val
        ok += int(good)
        details.append({"key": key, "metric": val, "ok": bool(good)})
    frac = ok / len(neighbors)
    return {
        "n_neighbors": len(neighbors),
        "n_ok": ok,
        "fraction_ok": frac,
        "required_fraction": min_frac,
        "passed": frac >= min_frac,
        "best_metric": best_val,
        "details": details,
    }


# --------------------------------------------------------------------------------------
# 6. ブロックブートストラップ
# --------------------------------------------------------------------------------------
def block_bootstrap_ci(values: np.ndarray, block_ids: np.ndarray, n_resamples: int,
                       ci: float, seed: int = 12345) -> dict:
    """時系列の自己相関を壊さないよう、ブロック（既定は 1 日）単位で再標本化する。"""
    v = np.asarray(values, dtype=float)
    b = np.asarray(block_ids)
    mask = np.isfinite(v)
    v, b = v[mask], b[mask]
    if v.size == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0, "n_blocks": 0}
    uniq = np.unique(b)
    groups = [v[b == u] for u in uniq]
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        pick = rng.integers(0, len(groups), len(groups))
        means[i] = np.concatenate([groups[p] for p in pick]).mean()
    alpha = (1.0 - ci) / 2.0
    return {
        "mean": float(v.mean()),
        "lo": float(np.quantile(means, alpha)),
        "hi": float(np.quantile(means, 1.0 - alpha)),
        "p_le_zero": float((means <= 0).mean()),
        "n": int(v.size),
        "n_blocks": int(len(groups)),
        "n_resamples": int(n_resamples),
    }


def day_blocks(ts: np.ndarray, block_seconds: int) -> np.ndarray:
    return (np.asarray(ts, dtype=np.int64) // int(block_seconds)).astype(np.int64)


# --------------------------------------------------------------------------------------
# 5. サンプル数ゲート
# --------------------------------------------------------------------------------------
def sample_gate(n_wins: int, params: Params) -> dict:
    need = int(params.get_path("validation.min_wins"))
    return {"n_wins": int(n_wins), "min_wins": need, "passed": int(n_wins) >= need}
