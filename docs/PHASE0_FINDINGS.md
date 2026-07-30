# Phase 0 findings — 2026-07-23 through 2026-07-29

## Scope

This is a seven-day engineering spike, not a strategy verdict. BTCUSDT and
ETHUSDT USDⓈ-M futures `aggTrades` and one-minute klines were downloaded from
Binance's public archive. Every downloaded ZIP passed the published SHA-256
checksum.

## Data quality

| Measure | BTCUSDT | ETHUSDT |
|---|---:|---:|
| Aggregate-trade rows | 6,333,155 | 6,212,878 |
| Estimated underlying trades | 17,750,020 | 28,489,675 |
| Aggregate-ID gaps | 0 | 0 |
| One-second timeline | 604,800 seconds after continuity fill | 604,800 seconds |
| Minimum minute-level taker-buy correlation to kline | 0.999978 | 0.999996 |
| `is_buyer_maker` orientation test | Passed | Passed |

Daily aggregate volume and taker-buy volume reconcile effectively exactly. A
small per-minute difference can occur because aggregate trades may cross minute
boundaries and because the current feed distinguishes normal and RPI-related
trades. The direction interpretation is nevertheless unambiguous: the intended
orientation is vastly closer to the official taker-buy totals than its inverse.

## Fixed initial detector

No parameters were tuned after seeing these results.

Round-trip cost assumption:

- taker fee: 5 bps per side;
- slippage allowance: 1.5 bps per side;
- total: 13 bps.

| Symbol | Base impulses | Median 60s MFE, earliest executable entry | Median 60s MFE, +10s reaction | Median net 60s return, earliest | Median net 60s return, +10s |
|---|---:|---:|---:|---:|---:|
| BTCUSDT | 146 | 1.02 bps | 0.59 bps | -12.73 bps | -12.98 bps |
| ETHUSDT | 72 | 1.30 bps | 0.61 bps | -13.08 bps | -13.00 bps |

The initial detector fails the preliminary oracle kill screen by a wide margin.
It is identifying movements that are statistically unusual relative to the
preceding hour but are much weaker than the visually obvious opportunities that
motivated the project.

This result must be reported before any threshold exploration. It does not show
that the user's observation is false. It shows that the first mathematical
translation does not yet isolate the same phenomenon.

## Next research decision

Do not build the live notifier. The next approved experiment should focus on
pre-registered alternative *shapes*, not arbitrary threshold hunting:

1. a larger discrete price displacement floor in basis points;
2. an explicit impulse–pause–reacceleration sequence;
3. a pullback-depth and reclaim sequence;
4. event rarity constrained before outcome inspection.

Every alternative must reuse the same executable-entry and cost model.

