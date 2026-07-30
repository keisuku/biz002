# Momentum Ignition Research — agent contract

This repository tests whether unusual short-lived crypto-futures momentum can
be detected early enough to remain tradable after human reaction delay and
execution costs.

## Non-negotiable rules

1. Never implement automatic order placement.
2. Never add a live trading dashboard or continuously updating chart UI.
3. Treat every strategy as a falsifiable hypothesis, not a product claim.
4. Phase 3 live notifications are forbidden until the Phase 2 viability gate
   has been reviewed and explicitly approved.
5. Never tune parameters silently. Record every evaluated combination.
6. All signal features must be computable with data available at `signal_ts`.
7. Simulated entry must occur after `signal_ts`; never enter at the signal
   bar's close.
8. Never commit raw market data, generated Parquet files, credentials, bot
   tokens, or `.env` files.

## Required checks

```bash
uv run pytest
uv run ruff check src tests
npm test
npm run lint
```

## Research gates

- Gate 0: data schema, checksums, aggregation, and reconciliation pass.
- Gate 1: fixed-parameter latency study is reported before any grid search.
- Gate 2: walk-forward out-of-sample validation passes before live work.
- Gate 3: shadow operation precedes real notifications.

