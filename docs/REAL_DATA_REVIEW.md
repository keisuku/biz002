# 実データ投入・実装比較レビュー — 2026-07-31

## 結論

Binance 公開アーカイブへの接続、チェックサム検証、Parquet 化、1 秒バー生成、
売買方向検証、発火台帳、生死判定まで実データで完走した。

初期検出器は BTCUSDT で明確に FAIL。ETHUSDT はイベントが 1 件しかなく、
統計的な結論以前にサンプル不足である。結果を見た後の閾値調整は行っていない。

この実データは 7 日間の工学的スパイクであり、3 年バックテストではない。

## データ経路

- Source: `data.binance.vision`
- Market: USDⓈ-M futures
- Dataset: daily `aggTrades`
- Period: 2026-07-23 through 2026-07-29 UTC
- Symbols: BTCUSDT, ETHUSDT
- Integrity: all 14 ZIP archives passed the published checksum
- Dense bars: 604,800 rows per symbol
- Long missing gaps: zero

`is_buyer_maker=False` をテイカー買いとする解釈も実データで確認した。

| Symbol | return–flow correlation | Top-1% move direction alignment |
|---|---:|---:|
| BTCUSDT | 0.3612 | 96.4% |
| ETHUSDT | 0.3308 | 95.0% |

## 固定初期条件の生死判定

| Symbol | Events | Median MFE 60s | Median cost | MFE / cost | Verdict |
|---|---:|---:|---:|---:|---|
| BTCUSDT | 58 | 0.01127% | 0.13995% | 0.0805x | FAIL |
| ETHUSDT | 1 | 0.00054% | 0.13413% | 0.0040x | FAIL / insufficient sample |

必要条件は `MFE / cost > 3.0`。BTC は必要水準の約 1/37 である。

## 手動操作の遅延

現在の発火本体は 10 秒速度窓である。反応遅延 0 秒でも、初動開始からの最短年齢は
すでに 10 秒。ユーザーが体感した「開始から約 7 秒のエントリー」は、現在の定義では
構造上再現できない。

BTCUSDT:

| Signal後の反応遅延 | 初動からの最短年齢 | Median MFE 60s | Median net return 60s |
|---:|---:|---:|---:|
| 0s | 10s | 1.127 bps | -13.946 bps |
| 3s | 13s | 0.999 bps | -14.855 bps |
| 7s | 17s | 0.646 bps | -14.849 bps |
| 10s | 20s | 0.429 bps | -14.521 bps |
| 15s | 25s | 0.412 bps | -14.494 bps |
| 20s | 30s | 0.164 bps | -14.091 bps |

このサンプルでは、20 秒時点の問題以前に、最短実行時点でも値幅がコストに届いていない。

## Claude Code 版と Codex 版

### Claude Code 版が優れている点

- Phase 2 の範囲が広い。E1〜E5、ハードストップ、時間帯分析、コスト感度、
  ウォークフォワード、DSR、プラトー、ブロックブートストラップまで実装している。
- ルックアヘッド、チャンク境界、合成対照群を含むテスト設計が厚い。
- 生死判定 FAIL で後続分析を止める規律がコードになっている。

### Claude Code 版で見つかった問題

1. `.gitignore` の `data/` が `src/data/` にも一致し、Phase 1 のソース一式が
   GitHub に保存されていなかった。main は import error で 52 テストを開始できなかった。
2. 依存関係に上限がなく、NumPy 2.5.1 で Bus error を再現した。
3. シグナル時終値で即時約定する前提で、ユーザーにとって最重要の 7〜20 秒遅延がなかった。
4. 初速、停止後再加速、押し戻し後奪回を交換可能に比較する仮説レジストリがない。
5. ストップ実測側の raw-data パスとダウンローダの保存パスが一致していなかった。
6. 60 秒クールダウンがメモリチャンク境界でリセットされる可能性があった。

本ブランチでは 1、2、3、5、6 を修正し、テストを 54 件へ増やした。

### Codex 版が優れている点

- 最初から Binance 実データを取得し、初回の数字を公開した。
- 0 / 3 / 7 / 10 / 15 / 20 秒の反応遅延と、初動開始からの年齢を中心に置いた。
- `instant_continuation`、`pause_then_continue`、`pullback_reclaim` を同じ執行モデルで
  比較できる構造にした。
- 静的な公開研究ページがあり、ライブ画面を作らないという行動設計を守った。

### Codex 版の弱い点

- Phase 2 全体は Claude Code 版より狭く、E1〜E5、DSR、9 分割ウォークフォワードなどは
  未実装だった。
- 7 日間の固定条件スパイクで停止しており、3 年分の本検証ではない。

## 統合方針

土台は Claude Code 版、研究インターフェースは Codex 版がよい。

次の実験は結果を見て閾値だけを動かすのではなく、以下を事前登録した別仮説として扱う。

1. 5 秒窓で検知し、通知・操作を含めて初動開始から 7 秒へ入れる高速系
2. impulse → pause → reacceleration の形状系
3. pullback depth → reclaim の形状系
4. `aggTrade` メッセージ数と、`last_trade_id-first_trade_id+1` の内部約定推定数の比較

4 の定義差により、同じ 7 日でも ETH のイベント数は Claude Code 版 1 件、
Codex 版 72 件となった。結果を見た後に都合のよい方を選ばず、別試行として両方を記録する。
