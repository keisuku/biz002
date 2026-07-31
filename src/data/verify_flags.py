"""Empirically verify the is_buyer_maker orientation from price/flow alignment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl


def check_bars(bars: pl.DataFrame) -> dict:
    usable = (
        bars.select(
            (pl.col("close").log() - pl.col("close").log().shift(1)).alias(
                "return"
            ),
            (pl.col("ofi") / pl.col("volume"))
            .replace([float("inf"), float("-inf")], None)
            .alias("flow"),
        )
        .drop_nulls()
        .filter(pl.col("flow").is_finite() & pl.col("return").is_finite())
    )
    if usable.height < 100:
        return {
            "verdict": "INSUFFICIENT_DATA",
            "n": usable.height,
            "correlation": None,
        }
    returns = usable["return"].to_numpy()
    flows = usable["flow"].to_numpy()
    correlation = float(np.corrcoef(returns, flows)[0, 1])
    tail = np.abs(returns) >= np.quantile(np.abs(returns), 0.99)
    tail_alignment = float(np.mean(np.sign(returns[tail]) == np.sign(flows[tail])))
    verdict = "PASS" if correlation > 0 and tail_alignment > 0.5 else "FAIL_CHECK_FLAG_POLARITY"
    return {
        "verdict": verdict,
        "n": usable.height,
        "correlation": correlation,
        "tail_alignment": tail_alignment,
        "interpretation": "is_buyer_maker=False means taker buy",
    }


def write_report(result: dict, output_dir: str | Path) -> None:
    destination = Path(output_dir) / "is_buyer_maker_verification.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

