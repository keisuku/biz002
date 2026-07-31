"""Phase 1 のパーサ。Binance は配布 CSV の形式（ヘッダ有無・時刻粒度）を途中で
変えているため、決め打ちで壊れないことをテストで固定する。"""

from __future__ import annotations

import io
import zipfile

import numpy as np
import polars as pl
import pytest

from src.data import download as dl

BODY = (
    "1,100.5,0.5,10,11,{t0},true\n"
    "2,100.6,1.5,12,13,{t1},false\n"
)
HEADER_NEW = "agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker\n"
HEADER_OLD = "a,p,q,f,l,T,m\n"


def _csv(header: str, t0: int, t1: int) -> bytes:
    return (header + BODY.format(t0=t0, t1=t1)).encode()


@pytest.mark.parametrize("header", ["", HEADER_NEW, HEADER_OLD])
def test_parses_with_and_without_header(header):
    df = dl.parse_agg_trades(_csv(header, 1_700_000_000_000, 1_700_000_000_500))
    assert df.height == 2
    assert df["price"].to_list() == [100.5, 100.6]
    assert df["is_buyer_maker"].to_list() == [True, False]
    assert df["ts_ms"].to_list() == [1_700_000_000_000, 1_700_000_000_500]


def test_microsecond_timestamps_are_normalised():
    us = dl.parse_agg_trades(_csv("", 1_700_000_000_000_000, 1_700_000_000_500_000))
    assert us["ts_ms"].to_list() == [1_700_000_000_000, 1_700_000_000_500]


def test_second_timestamps_are_normalised():
    s = dl.parse_agg_trades(_csv("", 1_700_000_000, 1_700_000_001))
    assert s["ts_ms"].to_list() == [1_700_000_000_000, 1_700_000_001_000]


def test_numeric_maker_flag():
    csv = ("1,100.5,0.5,10,11,1700000000000,1\n"
           "2,100.6,1.5,12,13,1700000000500,0\n").encode()
    df = dl.parse_agg_trades(csv)
    assert df["is_buyer_maker"].to_list() == [True, False]


def test_extra_trailing_column_is_ignored():
    csv = ("1,100.5,0.5,10,11,1700000000000,true,true\n").encode()
    df = dl.parse_agg_trades(csv)
    assert df.height == 1 and df["is_buyer_maker"][0]


def test_unzip_single_member():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("x.csv", "hello")
    assert dl.unzip_single(buf.getvalue()) == b"hello"


def test_unzip_rejects_multiple_members():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("a.csv", "1")
        zf.writestr("b.csv", "2")
    with pytest.raises(dl.LayoutError):
        dl.unzip_single(buf.getvalue())


def test_checksum_algorithm_selected_by_length():
    import hashlib

    data = b"abc"
    assert dl._hash_of(data, 64) == hashlib.sha256(data).hexdigest()
    assert dl._hash_of(data, 32) == hashlib.md5(data).hexdigest()
    with pytest.raises(dl.LayoutError):
        dl._hash_of(data, 12)


def test_resolve_dataset_name_is_case_and_plural_tolerant():
    layout = dl.Layout(market_root="data/futures/um", frequency="daily",
                       datasets=["aggTrades", "klines", "metrics"])
    assert dl.resolve_dataset_name(layout, "aggtrades") == "aggTrades"
    assert dl.resolve_dataset_name(layout, "kline") == "klines"
    with pytest.raises(dl.LayoutError):
        dl.resolve_dataset_name(layout, "bookDepth")


def test_normalize_epoch_ms_handles_nanoseconds():
    s = pl.Series([1_700_000_000_000_000_000, 1_700_000_000_500_000_000])
    assert dl.normalize_epoch_ms(s).to_list() == [1_700_000_000_000, 1_700_000_000_500]
