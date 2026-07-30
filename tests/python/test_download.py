from __future__ import annotations

from datetime import date

import pytest

from momentum_ignition.data.download import archive_url, parse_checksum


def test_archive_url_uses_observed_binance_layout() -> None:
    assert archive_url("btcusdt", date(2026, 7, 23)) == (
        "https://data.binance.vision/data/futures/um/daily/aggTrades/"
        "BTCUSDT/BTCUSDT-aggTrades-2026-07-23.zip"
    )


def test_kline_archive_url_requires_and_uses_interval() -> None:
    assert archive_url("ETHUSDT", date(2026, 7, 23), "klines", "1m") == (
        "https://data.binance.vision/data/futures/um/daily/klines/"
        "ETHUSDT/1m/ETHUSDT-1m-2026-07-23.zip"
    )


def test_checksum_parser_accepts_sha256_format() -> None:
    digest = "a" * 64
    assert parse_checksum(
        f"{digest}  BTCUSDT-aggTrades-2026-07-23.zip\n".encode(),
        "BTCUSDT-aggTrades-2026-07-23.zip",
    ) == digest


def test_checksum_parser_rejects_wrong_filename() -> None:
    with pytest.raises(ValueError, match="expected"):
        parse_checksum(
            f"{'a' * 64}  other.zip\n".encode(),
            "BTCUSDT-aggTrades-2026-07-23.zip",
        )
