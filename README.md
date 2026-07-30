# Momentum Ignition Research

An open, reproducible test of a simple trading observation:

> When price speed and trade intensity become unmistakably different from the
> preceding calm period, is there enough continuation left to trade after a
> human can actually react?

This repository is intentionally a research system, not a trading bot. It
contains no order placement and no live dashboard.

## Why latency is the central variable

The original observation involved entering roughly seven seconds after the
move became obvious. A notification workflow may push that to roughly twenty
seconds. The experiment therefore measures both delay after the signal and
total age from the start of the impulse.

Three strategy families share one execution and outcome model:

- immediate continuation;
- brief pause followed by renewed continuation;
- pullback followed by reclaim.

## Phase 0 quickstart

```bash
uv sync --extra dev

# Download seven complete UTC days. Dates are examples.
uv run mi-download \
  --symbol BTCUSDT \
  --start 2026-07-23 \
  --end 2026-07-29

uv run mi-download \
  --symbol BTCUSDT \
  --dataset klines \
  --interval 1m \
  --start 2026-07-23 \
  --end 2026-07-29

uv run mi-to-seconds \
  --symbol BTCUSDT \
  --start 2026-07-23 \
  --end 2026-07-29

uv run mi-reconcile \
  --symbol BTCUSDT \
  --start 2026-07-23 \
  --end 2026-07-29

uv run mi-run-experiment \
  --symbol BTCUSDT \
  --start 2026-07-23 \
  --end 2026-07-29
```

Raw downloads and generated research data remain local and are ignored by Git.

## Validation

```bash
uv run pytest
uv run ruff check src tests
npm test
npm run lint
```

See [docs/RESEARCH_CONTRACT.md](docs/RESEARCH_CONTRACT.md) before changing any
feature, signal, or cost assumption.
