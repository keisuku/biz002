"""モメンタム・イグニッション検出器。

Phase 1: データ取得・整備      (src.data)
Phase 2: 発火台帳と統計分析    (src.features / src.events / src.outcomes / src.exits / src.analysis)
Phase 3: リアルタイム検出器    (src.live)  ← Phase 2 の生死判定を通過するまで実装しない
"""

__all__ = ["config"]
