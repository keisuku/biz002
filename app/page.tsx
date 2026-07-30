// Internal workspace sites can read the authenticated OpenAI user from the
// forwarded request headers:
//
// import { headers } from "next/headers";
//
// export default async function Home() {
//   const requestHeaders = await headers();
//   const email = requestHeaders.get("oai-authenticated-user-email");
//   const encodedFullName = requestHeaders.get("oai-authenticated-user-full-name");
//   const fullName =
//     encodedFullName &&
//     requestHeaders.get("oai-authenticated-user-full-name-encoding") ===
//       "percent-encoded-utf-8"
//       ? decodeURIComponent(encodedFullName)
//       : null;
//   const displayName = fullName ?? email;
//   // ...
// }

const hypotheses = [
  {
    id: "H1",
    title: "初速へ乗る",
    english: "Instant continuation",
    body: "異常な速度・約定密度・方向フローを確認した直後に入る。最も速いが、通知と操作の遅延に最も弱い。",
    tone: "cyan",
  },
  {
    id: "H2",
    title: "一瞬の間を待つ",
    english: "Pause, then continue",
    body: "最初の急伸後、値崩れしない短い停止を確認。流れが同方向へ再加速した瞬間を候補にする。",
    tone: "violet",
  },
  {
    id: "H3",
    title: "押し目から戻す",
    english: "Pullback reclaim",
    body: "初動の高値・安値から一定割合戻した後、元の方向へ奪回し始めた時点を候補にする。",
    tone: "amber",
  },
];

const gates = [
  ["Gate 0", "データの正しさ", "チェックサム、欠損、1秒集約、売買方向を検証", "PASSED"],
  ["Gate 1", "固定条件の生死判定", "初期定義はコストを超えず、調整前に停止", "STOPPED"],
  ["Gate 2", "完全な期間外検証", "ウォークフォワードとコスト2倍で再検証", "LOCKED"],
  ["Gate 3", "シャドー運用", "実通知の前に、取引せずリアルタイム差異を測る", "LOCKED"],
];

export default function Home() {
  return (
    <main>
      <header className="siteHeader">
        <a className="brand" href="#top" aria-label="Momentum Ignition Research ホーム">
          <span className="brandMark" aria-hidden="true">
            MI
          </span>
          <span>
            <strong>Momentum Ignition</strong>
            <small>OPEN RESEARCH</small>
          </span>
        </a>
        <nav aria-label="ページ内ナビゲーション">
          <a href="#question">問い</a>
          <a href="#result">初回結果</a>
          <a href="#hypotheses">仮説</a>
          <a href="#gates">検証工程</a>
          <a
            className="repoLink"
            href="https://github.com/keisuku/biz002"
            target="_blank"
            rel="noreferrer"
          >
            GitHub ↗
          </a>
        </nav>
      </header>

      <section className="hero" id="top">
        <div className="heroGlow" aria-hidden="true" />
        <div className="eyebrow">
          <span className="statusDot" />
          PHASE 0 · 初回固定条件を公開
        </div>
        <h1>
          その急変は、
          <br />
          <em>20秒後</em>にも取れるのか。
        </h1>
        <p className="heroLead">
          「今までと明らかに違う」値動きを検知しても、人間が通知を見て購入する頃には
          優位性が消えているかもしれない。感覚を否定も肯定もせず、秒単位で測る公開研究です。
        </p>
        <div className="heroActions">
          <a className="primaryButton" href="#question">
            検証設計を見る
          </a>
          <span>自動発注なし · ライブ画面なし · 結果は公開</span>
        </div>

        <div className="latencyPanel" aria-label="検証する遅延の比較">
          <div className="latencyIntro">
            <small>THE LATENCY PROBLEM</small>
            <strong>勝てた体感</strong>
            <span>初動から約7秒</span>
          </div>
          <div className="latencyTrack">
            <span className="trackLine" aria-hidden="true" />
            {[0, 3, 7, 10, 15, 20].map((second) => (
              <div
                className={`latencyPoint ${second === 7 ? "observed" : ""} ${
                  second === 20 ? "risk" : ""
                }`}
                key={second}
              >
                <i />
                <strong>{second}s</strong>
                <small>
                  {second === 7 ? "体感値" : second === 20 ? "通知運用想定" : "検証"}
                </small>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="questionSection" id="question">
        <div className="sectionLabel">01 · THE QUESTION</div>
        <div className="questionGrid">
          <h2>
            見つけたいのは
            <br />
            「急騰」ではない。
          </h2>
          <div>
            <p className="largeCopy">
              凪から初速が生まれ、短い停止を挟んでも、
              <strong>まだ注文できるだけの勢いが残っている局面</strong>です。
            </p>
            <p>
              速度、約定密度、成行方向、単位出来高あたりの価格変化を同時に測定。
              シグナル時点の価格ではなく、反応時間後に初めて約定できる価格から損益を計算します。
            </p>
          </div>
        </div>
        <div className="principles">
          <div>
            <span>01</span>
            <strong>未来価格で入らない</strong>
            <p>シグナル秒の終了後、次に実行可能な秒から計測。</p>
          </div>
          <div>
            <span>02</span>
            <strong>調整前を公開する</strong>
            <p>最初の固定条件が死んだら、数字を隠さず一度停止。</p>
          </div>
          <div>
            <span>03</span>
            <strong>画面を見続けない</strong>
            <p>公開ページは研究結果のみ。ライブ価格は表示しない。</p>
          </div>
        </div>
      </section>

      <section className="resultSection" id="result">
        <div className="sectionLabel">02 · FIRST FIXED RUN</div>
        <div className="resultHeading">
          <div>
            <span className="stopBadge">STOP · 初期定義は不成立</span>
            <h2>
              動いた。
              <br />
              ただし、弱すぎた。
            </h2>
          </div>
          <p>
            2026年7月23日〜29日のBTCUSDT・ETHUSDTで、結果を見る前に固定した条件を一度だけ実行。
            最も楽観的なMFEでも、想定往復コスト13bpを大幅に下回りました。
          </p>
        </div>
        <div className="resultGrid">
          <article>
            <small>BTCUSDT · 146 IMPULSES</small>
            <strong>1.02 <i>bp</i></strong>
            <span>最短実行の60秒MFE中央値</span>
            <p>+10秒遅延では 0.59bp</p>
          </article>
          <article>
            <small>ETHUSDT · 72 IMPULSES</small>
            <strong>1.30 <i>bp</i></strong>
            <span>最短実行の60秒MFE中央値</span>
            <p>+10秒遅延では 0.61bp</p>
          </article>
          <article className="costCard">
            <small>ROUND-TRIP ASSUMPTION</small>
            <strong>13.00 <i>bp</i></strong>
            <span>手数料10bp + 滑り3bp</span>
            <p>ライブ通知器は作らず停止</p>
          </article>
        </div>
        <p className="resultNote">
          この結果は体感した現象の否定ではありません。初期の数式が、目で見ていた
          「ガッと動く」局面よりはるかに弱い動きまで拾った、という診断です。
        </p>
      </section>

      <section className="hypothesisSection" id="hypotheses">
        <div className="sectionLabel">03 · HYPOTHESIS REGISTRY</div>
        <div className="sectionHeading">
          <h2>入口を一つに決めつけない。</h2>
          <p>
            同じデータ、同じコスト、同じ評価指標で比較するため、
            検出ロジックと執行シミュレーションを分離しています。
          </p>
        </div>
        <div className="hypothesisGrid">
          {hypotheses.map((hypothesis) => (
            <article className={`hypothesisCard ${hypothesis.tone}`} key={hypothesis.id}>
              <div className="cardTop">
                <span>{hypothesis.id}</span>
                <small>{hypothesis.english}</small>
              </div>
              <div className="microChart" aria-hidden="true">
                <i />
                <i />
                <i />
                <i />
                <i />
                <i />
                <i />
                <i />
              </div>
              <h3>{hypothesis.title}</h3>
              <p>{hypothesis.body}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="gateSection" id="gates">
        <div className="sectionLabel">04 · STOP RULES</div>
        <div className="sectionHeading">
          <h2>良い数字が出るまで弄らない。</h2>
          <p>
            研究の目的は仮説を守ることではなく、取れない仮説を早く捨てることです。
          </p>
        </div>
        <div className="gateList">
          {gates.map(([number, title, body, status]) => (
            <article key={number}>
              <span>{number}</span>
              <div>
                <strong>{title}</strong>
                <p>{body}</p>
              </div>
              <i className={status === "PASSED" ? "passedGate" : status === "STOPPED" ? "stoppedGate" : ""}>
                {status}
              </i>
            </article>
          ))}
        </div>
      </section>

      <section className="currentSection">
        <div>
          <div className="sectionLabel">CURRENT STATUS</div>
          <h2>初期定義では、勝てません。</h2>
        </div>
        <div className="currentCard">
          <span>PHASE 0 完了</span>
          <ul>
            <li>Binance日次データのチェックサム付き取得</li>
            <li>aggTradesと推定約定件数を分けた1秒集約</li>
            <li>未来データ混入を防ぐ特徴量テスト</li>
            <li>0〜20秒の反応遅延シミュレーション</li>
            <li>3種類の交換可能な仮説レジストリ</li>
          </ul>
          <p>
            次へ進むなら、値幅の絶対条件、初動–停止–再加速、押し戻し–奪回を
            事前登録した別実験にします。結果を見ながら閾値だけを弄ることはしません。
          </p>
        </div>
      </section>

      <footer>
        <div className="brand footerBrand">
          <span className="brandMark">MI</span>
          <span>
            <strong>Momentum Ignition</strong>
            <small>OPEN RESEARCH</small>
          </span>
        </div>
        <p>
          研究・教育目的。投資助言ではありません。自動発注機能は実装しません。
        </p>
        <a
          href="https://github.com/keisuku/biz002"
          target="_blank"
          rel="noreferrer"
        >
          Source on GitHub ↗
        </a>
      </footer>
    </main>
  );
}
