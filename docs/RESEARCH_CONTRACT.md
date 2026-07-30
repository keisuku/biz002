# Research contract

## Question

Does an abrupt, unusual increase in price speed and aggressive trade flow
produce enough continuation to remain profitable after realistic notification,
human reaction, spread, fees, and slippage?

The observation that motivated this project is deliberately treated as one
hypothesis. The framework must also test later entries:

1. `instant_continuation`: enter as soon as the initial impulse is detected.
2. `pause_then_continue`: wait for the first brief pause, then enter only when
   directional flow restarts without materially giving back the impulse.
3. `pullback_reclaim`: wait for a measurable retracement from the impulse
   extreme, then enter when the original direction starts to reclaim.

New hypotheses must implement the same event and outcome contracts. They may
not rewrite the data or cost model to make themselves look better.

## Time contract

- `impulse_start_ts`: earliest second included in the triggering return window.
- `signal_ts`: end of the last fully observed second used by the detector.
- `decision_ts`: when the strategy is allowed to decide; initially equal to
  `signal_ts`.
- `entry_ts`: first executable second strictly after `decision_ts + delay`.
- `signal_age_s`: `signal_ts - impulse_start_ts`.
- `entry_age_s`: `entry_ts - impulse_start_ts`.

The primary latency comparison uses 0, 3, 7, 10, 15, and 20 seconds after the
signal. Results must also report total age from the beginning of the impulse.

## Data contract

Binance `aggTrades` is aggregate-trade data, not a one-row-per-fill tape.

- `aggtrade_count`: number of aggregate-trade rows.
- `trade_count_est`: sum of `last_trade_id - first_trade_id + 1`.
- `signed_taker_volume`: taker-buy volume minus taker-sell volume.

`signed_taker_volume` is not called true order-flow imbalance because the core
dataset does not contain full order-book changes.

Seconds without trades are present in the one-second series. OHLC is carried
forward from the prior close, while volume and count fields are zero.

## Viability contract

The first fixed-parameter result is reported before parameter exploration.

1. Oracle kill screen: executable-entry `mfe_60` median must exceed the
   round-trip cost estimate by at least 3×.
2. Realizable screen: fixed-horizon net returns are reported for every latency.
3. Passing the oracle screen does not establish viability.
4. Failure is a valid result. No tuning occurs until the fixed result has been
   reviewed.

## Publication contract

The public page is a static research report, not a signal terminal. It must
show status, methodology, latency sensitivity, and limitations. It must never
display live prices or create a reason to watch the screen continuously.

