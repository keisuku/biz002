from __future__ import annotations

from momentum_ignition.features import add_features
from momentum_ignition.hypotheses import generate_signals


def test_hypothesis_registry_can_emit_multiple_entry_timings(
    calm_then_impulse_bars,
    compact_config,
) -> None:
    features = add_features(calm_then_impulse_bars, compact_config)
    signals = generate_signals("BTCUSDT", features, compact_config)
    names = {signal.hypothesis for signal in signals}
    assert "instant_continuation" in names
    assert names <= {
        "instant_continuation",
        "pause_then_continue",
        "pullback_reclaim",
    }
    assert all(signal.signal_ts >= signal.impulse_start_ts for signal in signals)

