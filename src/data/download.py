"""Binance public-archive discovery, download, verification, and parsing.

The archive layout is discovered through the bucket listing before files are
requested. Downloaded ZIP files are checksum-verified and converted to
Parquet, so all downstream code reads one stable schema.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import re
import time
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlencode

import polars as pl
import requests

from ..config import Params, resolve_dir

log = logging.getLogger(__name__)


class LayoutError(RuntimeError):
    """The public archive differs from the structure the pipeline can parse."""


@dataclass(frozen=True)
class Layout:
    market_root: str
    frequency: str
    datasets: list[str]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


def _request_bytes(url: str, timeout: float, attempts: int, backoff: float) -> bytes:
    headers = {"User-Agent": "momentum-ignition-research/1.0"}
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.content
        except requests.RequestException as exc:
            last = exc
            if attempt + 1 < attempts:
                time.sleep(backoff * (2**attempt))
    raise LayoutError(f"request failed after {attempts} attempts: {url}: {last}")


def _network_settings(params: Params) -> tuple[float, int, float]:
    return (
        float(params.get_path("data.request_timeout_seconds")),
        int(params.get_path("data.retry.attempts")),
        float(params.get_path("data.retry.backoff_seconds")),
    )


def _list_prefixes(params: Params, prefix: str) -> list[str]:
    timeout, attempts, backoff = _network_settings(params)
    query = urlencode({"delimiter": "/", "prefix": prefix})
    payload = _request_bytes(
        f"{params.get_path('data.list_url')}?{query}", timeout, attempts, backoff
    )
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise LayoutError(f"bucket listing was not XML for prefix {prefix!r}") from exc
    return [
        node.text or ""
        for node in root.findall(".//{*}CommonPrefixes/{*}Prefix")
        if node.text
    ]


def discover_layout(params: Params, symbol: str) -> Layout:
    """Read the real dataset directories instead of trusting remembered paths."""
    del symbol  # Dataset directories are market-wide; symbol validation happens per file.
    market_root = str(params.get_path("data.market_root")).strip("/")
    frequency = str(params.get_path("data.frequency")).strip("/")
    prefix = f"{market_root}/{frequency}/"
    prefixes = _list_prefixes(params, prefix)
    datasets = sorted({p.removeprefix(prefix).strip("/") for p in prefixes if p != prefix})
    if not datasets:
        raise LayoutError(f"no dataset directories found below {prefix!r}")
    return Layout(market_root=market_root, frequency=frequency, datasets=datasets)


def _canonical_dataset(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower()).removesuffix("s")


def resolve_dataset_name(layout: Layout, wanted: str) -> str:
    target = _canonical_dataset(wanted)
    for actual in layout.datasets:
        if _canonical_dataset(actual) == target:
            return actual
    raise LayoutError(
        f"dataset {wanted!r} not found; available={', '.join(layout.datasets)}"
    )


def _filename(dataset: str, symbol: str, day: date, interval: str | None) -> str:
    if dataset.lower() == "klines":
        if not interval:
            raise LayoutError("klines require an interval such as 1m")
        return f"{symbol}-{interval}-{day.isoformat()}.zip"
    return f"{symbol}-{dataset}-{day.isoformat()}.zip"


def _archive_key(
    layout: Layout, dataset: str, symbol: str, day: date, interval: str | None
) -> str:
    parts = [layout.market_root, layout.frequency, dataset, symbol]
    if dataset.lower() == "klines":
        if not interval:
            raise LayoutError("klines require an interval")
        parts.append(interval)
    parts.append(_filename(dataset, symbol, day, interval))
    return "/".join(parts)


def raw_path(
    params: Params,
    dataset: str,
    symbol: str,
    day: date,
    interval: str | None = None,
) -> Path:
    suffix = f"-{interval}" if interval else ""
    return (
        resolve_dir(params, "raw_dir")
        / dataset
        / symbol.upper()
        / f"{day:%Y}"
        / f"{day:%m}"
        / f"{symbol.upper()}-{dataset}{suffix}-{day.isoformat()}.parquet"
    )


def unzip_single(payload: bytes) -> bytes:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = [m for m in archive.infolist() if not m.is_dir()]
        if len(members) != 1:
            raise LayoutError(
                f"expected one file in archive, found {[m.filename for m in members]}"
            )
        return archive.read(members[0])


def _hash_of(payload: bytes, digest_length: int) -> str:
    if digest_length == 64:
        return hashlib.sha256(payload).hexdigest()
    if digest_length == 32:
        return hashlib.md5(payload).hexdigest()  # noqa: S324 - archive integrity only
    raise LayoutError(f"unsupported checksum digest length: {digest_length}")


def _expected_checksum(payload: bytes, filename: str) -> str:
    line = payload.decode("utf-8").strip().splitlines()[0]
    match = re.match(r"^([0-9a-fA-F]+)\s+\*?(.+)$", line)
    if not match:
        raise LayoutError(f"unsupported checksum file: {line!r}")
    digest, named = match.groups()
    if Path(named).name != filename:
        raise LayoutError(f"checksum names {named!r}; expected {filename!r}")
    return digest.lower()


def normalize_epoch_ms(values: pl.Series) -> pl.Series:
    """Normalize seconds, milliseconds, microseconds, or nanoseconds to ms."""
    cast = values.cast(pl.Int64, strict=False)
    finite = cast.drop_nulls()
    if len(finite) == 0:
        return cast
    magnitude = abs(int(finite.median()))
    if magnitude >= 100_000_000_000_000_000:
        return cast // 1_000_000
    if magnitude >= 100_000_000_000_000:
        return cast // 1_000
    if magnitude >= 100_000_000_000:
        return cast
    return cast * 1_000


def parse_agg_trades(payload: bytes) -> pl.DataFrame:
    """Parse both legacy/new headers and all epoch precisions used by Binance."""
    first = payload.splitlines()[0].decode("utf-8", errors="replace").lower()
    has_header = first.split(",", 1)[0] in {"a", "agg_trade_id", "aggtradeid"}
    frame = pl.read_csv(
        io.BytesIO(payload),
        has_header=has_header,
        infer_schema=False,
        truncate_ragged_lines=True,
        ignore_errors=False,
    )
    if frame.width < 7:
        raise LayoutError(f"aggTrades CSV has {frame.width} columns; expected at least 7")
    frame = frame.select(frame.columns[:7])
    frame.columns = [
        "agg_trade_id",
        "price",
        "quantity",
        "first_trade_id",
        "last_trade_id",
        "transact_time",
        "is_buyer_maker_raw",
    ]
    maker = (
        pl.col("is_buyer_maker_raw")
        .cast(pl.String)
        .str.to_lowercase()
        .is_in(["true", "1"])
        .alias("is_buyer_maker")
    )
    frame = frame.with_columns(
        pl.col("agg_trade_id").cast(pl.Int64, strict=False),
        pl.col("price").cast(pl.Float64, strict=False),
        pl.col("quantity").cast(pl.Float64, strict=False),
        pl.col("first_trade_id").cast(pl.Int64, strict=False),
        pl.col("last_trade_id").cast(pl.Int64, strict=False),
        maker,
    )
    frame = frame.with_columns(
        pl.Series("ts_ms", normalize_epoch_ms(frame["transact_time"]))
    )
    required = ["agg_trade_id", "ts_ms", "price", "quantity", "is_buyer_maker"]
    return frame.select(required + ["first_trade_id", "last_trade_id"]).drop_nulls(required)


def _parse_generic_csv(payload: bytes) -> pl.DataFrame:
    return pl.read_csv(io.BytesIO(payload), infer_schema_length=10_000)


def fetch_range(
    params: Params,
    layout: Layout,
    dataset: str,
    symbols: list[str],
    start: date,
    end: date,
    interval: str | None = None,
) -> pl.DataFrame:
    """Download a date range and return one manifest row per expected archive."""
    if end < start:
        raise ValueError("end must be on or after start")
    base_url = str(params.get_path("data.base_url")).rstrip("/")
    timeout, attempts, backoff = _network_settings(params)
    verify = bool(params.get_path("data.verify_checksum"))
    rows: list[dict] = []
    day = start
    while day <= end:
        for raw_symbol in symbols:
            symbol = raw_symbol.upper()
            filename = _filename(dataset, symbol, day, interval)
            key = _archive_key(layout, dataset, symbol, day, interval)
            url = f"{base_url}/{key}"
            destination = raw_path(params, dataset, symbol, day, interval)
            if destination.exists():
                rows.append(
                    {
                        "symbol": symbol,
                        "dataset": dataset,
                        "date": day.isoformat(),
                        "status": "cached",
                        "rows": pl.read_parquet(destination).height,
                        "path": str(destination),
                        "url": url,
                        "checksum_verified": True,
                    }
                )
                continue
            try:
                archive = _request_bytes(url, timeout, attempts, backoff)
                checksum_ok = False
                if verify:
                    checksum_payload = _request_bytes(
                        f"{url}.CHECKSUM", timeout, attempts, backoff
                    )
                    expected = _expected_checksum(checksum_payload, filename)
                    actual = _hash_of(archive, len(expected))
                    if actual != expected:
                        raise LayoutError(f"checksum mismatch for {url}")
                    checksum_ok = True
                csv_payload = unzip_single(archive)
                if dataset.lower() == "aggtrades":
                    parsed = parse_agg_trades(csv_payload)
                else:
                    parsed = _parse_generic_csv(csv_payload)
                destination.parent.mkdir(parents=True, exist_ok=True)
                parsed.write_parquet(destination)
                rows.append(
                    {
                        "symbol": symbol,
                        "dataset": dataset,
                        "date": day.isoformat(),
                        "status": "ok",
                        "rows": parsed.height,
                        "path": str(destination),
                        "url": url,
                        "checksum_verified": checksum_ok,
                    }
                )
            except (LayoutError, requests.RequestException) as exc:
                log.warning("download failed: %s", exc)
                rows.append(
                    {
                        "symbol": symbol,
                        "dataset": dataset,
                        "date": day.isoformat(),
                        "status": "missing",
                        "rows": 0,
                        "path": str(destination),
                        "url": url,
                        "checksum_verified": False,
                    }
                )
        day += timedelta(days=1)
    return pl.DataFrame(rows)
