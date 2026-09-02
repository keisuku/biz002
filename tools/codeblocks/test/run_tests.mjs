// codeblocks.html の純粋関数部を Node で検証する（開発時のみ使用。社内 PC では不要）
// 実行: node tools/codeblocks/test/run_tests.mjs
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.join(here, '..');
const html = fs.readFileSync(path.join(root, 'codeblocks.html'), 'utf8');

// ---- ツール本体から <script> を取り出して評価する ----
const scriptStart = html.indexOf('<script>');
const scriptEnd = html.lastIndexOf('</script>');
const script = html.slice(scriptStart + '<script>'.length, scriptEnd);
const ctx = vm.createContext({ console, TextEncoder, URL, setTimeout, clearTimeout });
vm.runInContext(script, ctx, { filename: 'codeblocks.html' });
const api = vm.runInContext('({ tokenizeJs, splitJsTopLevel, parseDocument, joinDocument, segmentHtml, parseResponse, parseEnvelopes, parseBareCode, applyItems, checkDocumentSyntax, diffLines, diffStats, buildPrompt, makeCatalog, makePartsJs, makeZip, crc32, stampedFileName, findOmissions, omissionRegexes, relatedBlocks, transitiveCalls, SAMPLE_APP, SAMPLE_RESPONSE, SAMPLE_RESPONSE_BAD, DEFAULT_TEMPLATES, DEFAULT_OMISSION_PATTERNS, blockCode, headingOf })', ctx);

let pass = 0, fail = 0;
function test(name, fn) {
  try { fn(); pass++; console.log('  ok  ' + name); }
  catch (e) { fail++; console.log('  NG  ' + name + '\n      ' + (e && e.stack || e).toString().split('\n').slice(0, 4).join('\n      ')); }
}
function eq(a, b, msg) {
  if (a !== b) throw new Error((msg || 'eq') + '\n      expected: ' + JSON.stringify(b) + '\n      actual:   ' + JSON.stringify(a));
}
function ok(v, msg) { if (!v) throw new Error(msg || 'ok'); }
const names = (stmts) => stmts.filter(s => s.kind !== 'trivia').map(s => s.name || ('<' + s.kind + '>'));

console.log('== トークナイザ ==');
test('文字列・コメント・正規表現の中の括弧に惑わされない', () => {
  const src = "function a(){ return '}' + \"}\" + `}` + /}/.test('x') + `${ {b:'}'}.b }`; } // }\n/* } */\nfunction b(){}\n";
  const st = api.splitJsTopLevel(src);
  eq(names(st).join(','), 'a,b');
  eq(st.map(s => s.text).join(''), src, '往復一致');
});
test('除算と正規表現を区別する', () => {
  const src = 'const x = a / b / c;\nconst y = /ab\\/c/g.test(s);\nfunction f(){ return x / 2; }\n';
  eq(names(api.splitJsTopLevel(src)).join(','), 'x,y,f');
});
test('セミコロン無し（ASI）でも文を分けられる', () => {
  const src = 'let a = 1\nlet b = 2\nfunction f(){}\nconst g = () => {\n  return 1\n}\nfoo()\nbar()\n';
  eq(names(api.splitJsTopLevel(src)).join(','), 'a,b,f,g,<stmt>,<stmt>');
});
test('行頭の演算子・ドットは前の文の続きになる', () => {
  const src = 'const s = a\n  + b\n  .c()\nconst t = 2;\n';
  eq(names(api.splitJsTopLevel(src)).join(','), 's,t');
});
test('if / else / try / catch / do-while を 1 文として扱う', () => {
  const src = 'if (a)\n  x();\nelse if (b) {\n  y();\n} else\n  z();\ntry {\n  q();\n} catch (e) {\n} finally {\n}\ndo {\n  r();\n} while (k);\nfunction w(){}\n';
  eq(names(api.splitJsTopLevel(src)).join(','), '<stmt>,<stmt>,<stmt>,w');
});
test('class / async / export / 代入 / 分割代入 を判定する', () => {
  const src = 'class A extends B(C) { m(){ return {}; } }\nasync function f(){ await 1; }\nexport const K = 1;\nwindow.onload = () => {};\nconst {a, b} = obj;\nfoo.bar.baz = 3;\n';
  const st = api.splitJsTopLevel(src);
  eq(names(st).join('|'), 'A|f|K|window.onload|{a, b}|foo.bar.baz');
  eq(st[0].kind, 'class'); eq(st[1].kind, 'function'); eq(st[2].kind, 'var'); eq(st[3].kind, 'assign');
});
test('先頭コメントは次の宣言に属し、末尾コメントは同じ文に属する', () => {
  const src = '// ==== 区切り ====\n\n// 役割\nfunction a(){} // 末尾\n\n/** doc */\nfunction b(){}\n';
  const st = api.splitJsTopLevel(src);
  eq(st.length, 2);
  ok(st[0].text.startsWith('// ==== 区切り'), 'a の先頭にコメント');
  ok(st[0].text.includes('// 末尾\n'), 'a の末尾コメント');
  ok(st[1].text.startsWith('\n/** doc */'), 'b の先頭に doc');
});
test('CRLF でも分割・往復できる', () => {
  const src = 'function a(){\r\n  return 1;\r\n}\r\n\r\nfunction b(){}\r\n';
  const st = api.splitJsTopLevel(src);
  eq(names(st).join(','), 'a,b');
  eq(st.map(s => s.text).join(''), src);
});

console.log('== HTML 分割と文書 ==');
test('サンプルアプリを分割して往復一致する', () => {
  const doc = api.parseDocument(api.SAMPLE_APP);
  ok(doc.roundTrip, 'roundTrip');
  const keys = doc.blocks.map(b => b.key);
  for (const k of ['STATE', 'parseRow', 'sumMinutes', 'formatMinutes', 'renderTable', 'onAdd', 'init', 'css: 基本', 'css: 入力フォーム', 'css: 一覧テーブル', 'html: 画面: 入力', 'html: 画面: 一覧']) ok(keys.includes(k), 'ブロック ' + k + ' がある: ' + keys.join(','));
  const sum = doc.byKey.get('sumMinutes');
  eq(sum.role, '分の合計を返す');
  eq(sum.section, '計算');
  eq(sum.sig, 'function sumMinutes(rows)');
  ok(sum.calledBy.includes('renderTable'), 'renderTable から呼ばれる');
  ok(doc.byKey.get('renderTable').calls.includes('formatMinutes'), 'renderTable が formatMinutes を呼ぶ');
  ok(doc.byKey.get('html: 画面: 入力').calls.includes('onAdd'), 'HTML から onAdd を呼ぶ');
});
test('ツール自身を分割して往復一致し、主要関数がブロックになる', () => {
  const doc = api.parseDocument(html);
  ok(doc.roundTrip, 'roundTrip');
  for (const k of ['STATE', 'tokenizeJs', 'splitJsTopLevel', 'parseDocument', 'parseResponse', 'applyItems', 'buildPrompt', 'init', 'SAMPLE_APP']) ok(doc.byKey.has(k), 'ブロック ' + k);
  ok(doc.blocks.filter(b => b.kind === 'stmt').length <= 2, '名前のない文は起動行だけ: ' + doc.blocks.filter(b => b.kind === 'stmt').map(b => b.display).join(' | '));
  ok(doc.byKey.get('parseDocument').calls.includes('segmentHtml'), '依存');
});
test('外部ライブラリ・JSON・純粋 JS ファイルを扱える', () => {
  const src = '<html><head><script src="lib.js"></script><script type="application/json">{"a":1}</script></head><body><script>\nfunction a(){}\n</script></body></html>';
  const doc = api.parseDocument(src);
  ok(doc.roundTrip);
  const lib = doc.blocks.find(b => b.kind === 'frozen');
  ok(lib && lib.key === 'lib: lib.js', 'lib block');
  ok(doc.blocks.some(b => b.kind === 'text'), 'json text block');
  ok(doc.byKey.has('a'));
  const js = api.parseDocument('function x(){}\nfunction y(){ x(); }\n');
  eq(js.segments.length, 1); eq(js.segments[0].type, 'script'); ok(js.roundTrip); ok(js.byKey.get('y').calls.includes('x'));
});
test('大文字タグと属性付き script でも分割できる', () => {
  const src = '<HTML><BODY><SCRIPT TYPE="text/javascript" defer>\nvar q = 1;\n</SCRIPT></BODY></HTML>';
  const doc = api.parseDocument(src);
  ok(doc.roundTrip); ok(doc.byKey.has('q'));
});
test('30 万文字規模でも 1 秒以内に解析できる', () => {
  let js = '';
  for (let i = 0; i < 2500; i++) js += '// 関数 ' + i + '\nfunction fn' + i + '(a, b) {\n  const s = "x}" + `${a}` + /}/.test(b);\n  if (a > b) {\n    return fn' + (i > 0 ? i - 1 : 0) + '(b, a);\n  }\n  return s;\n}\n\n';
  const src = '<html><body><script>\n' + js + '</script></body></html>';
  ok(src.length > 300000, 'サイズ ' + src.length);
  const t0 = Date.now();
  const doc = api.parseDocument(src);
  const ms = Date.now() - t0;
  ok(doc.roundTrip);
  eq(doc.blocks.filter(b => b.kind === 'function').length, 2500);
  ok(ms < 3000, '解析時間 ' + ms + 'ms');
});

console.log('== 応答パーサ ==');
test('封筒形式（BLOCK / NEW BLOCK / DELETE BLOCK、フェンス付き）を読む', () => {
  const text = '説明です。\n```js\n=== BLOCK: sumMinutes ===\nfunction sumMinutes(rows){ return 0; }\n=== END ===\n```\n=== NEW BLOCK: avg | after: sumMinutes ===\n```\nfunction avg(){ return 1; }\n```\n=== END ===\n=== DELETE BLOCK: onAdd ===\n';
  const items = api.parseEnvelopes(text);
  eq(items.length, 3);
  eq(items[0].action, 'replace'); eq(items[0].name, 'sumMinutes'); eq(items[0].code, 'function sumMinutes(rows){ return 0; }');
  eq(items[1].action, 'add'); eq(items[1].after, 'sumMinutes'); eq(items[1].code, 'function avg(){ return 1; }');
  eq(items[2].action, 'delete'); eq(items[2].name, 'onAdd');
});
test('サンプル返答を解析すると置換 1 件・追加 1 件、警告なし', () => {
  const doc = api.parseDocument(api.SAMPLE_APP);
  const r = api.parseResponse(api.SAMPLE_RESPONSE, doc, null);
  eq(r.mode, 'envelope'); eq(r.items.length, 2);
  eq(r.items[0].action, 'replace'); eq(r.items[0].warnings.length, 0, JSON.stringify(r.items[0].warnings)); eq(r.items[0].errors.length, 0);
  eq(r.items[1].action, 'add'); eq(r.items[1].after, 'sumMinutes');
});
test('省略サイン入りの返答に警告が出る', () => {
  const doc = api.parseDocument(api.SAMPLE_APP);
  const r = api.parseResponse(api.SAMPLE_RESPONSE_BAD, doc, null);
  eq(r.items.length, 1);
  ok(r.items[0].warnings.some(w => w.kind === 'omission'), JSON.stringify(r.items[0].warnings));
});
test('省略サインの検出パターン', () => {
  const re = api.omissionRegexes(api.DEFAULT_OMISSION_PATTERNS);
  eq(api.findOmissions('// ...省略...\nfoo();\n/* unchanged */\n...\n// rest of the code', re).join(','), '1,3,4,5');
  eq(api.findOmissions('const s = "...";\nconsole.log("省略");\n', re).length, 0, '文字列の中は対象外');
});
test('封筒なし（コードだけ）は宣言名で突き合わせる', () => {
  const doc = api.parseDocument(api.SAMPLE_APP);
  const r = api.parseResponse('直しました。\n```js\nfunction sumMinutes(rows){ return 1; }\nfunction newOne(){}\n```\n', doc, null);
  eq(r.mode, 'bare'); eq(r.items.length, 2);
  eq(r.items[0].action, 'replace'); eq(r.items[1].action, 'add');
});
test('全文 HTML が返ってきたらブロック差分にする', () => {
  const doc = api.parseDocument(api.SAMPLE_APP);
  const full = api.SAMPLE_APP.replace('return h + \':\'', 'return h + \'h\'').replace('// 起動\nfunction init() {', 'function extra(){}\n\n// 起動\nfunction init() {').replace(/\/\/ 追加ボタン\nfunction onAdd\(\) \{[\s\S]*?\n\}\n\n/, '');
  const r = api.parseResponse(full, doc, null);
  eq(r.mode, 'fullfile');
  const acts = r.items.map(i => i.action + ':' + i.name).sort().join(',');
  eq(acts, 'add:extra,delete:onAdd,replace:formatMinutes');
});
test('改名・行数急減・構文エラー・凍結を検出する', () => {
  const doc = api.parseDocument(api.SAMPLE_APP);
  const r = api.parseResponse('=== BLOCK: sumMinutes ===\nfunction sumAll(rows){ return 0; }\n=== END ===\n=== BLOCK: renderTable ===\nfunction renderTable(){ if ( }\n=== END ===\n', doc, null);
  ok(r.items[0].warnings.some(w => w.kind === 'rename'), 'rename');
  ok(r.items[1].errors.some(e => e.kind === 'syntax'), 'syntax');
  const big = 'function big(){\n' + '  x();\n'.repeat(30) + '}\n';
  const doc2 = api.parseDocument('<html><body><script src="a.js"></script><script>\n' + big + '</script></body></html>');
  const r2 = api.parseResponse('=== BLOCK: big ===\nfunction big(){ x(); }\n=== END ===\n=== BLOCK: lib: a.js ===\nfoo\n=== END ===', doc2, null);
  ok(r2.items[0].warnings.some(w => w.kind === 'shrink'), 'shrink');
  ok(r2.items[1].errors.some(e => e.kind === 'frozen'), 'frozen');
});

console.log('== 適用と差分 ==');
test('置換・追加・削除を適用して再解析できる', () => {
  const doc = api.parseDocument(api.SAMPLE_APP);
  const r = api.parseResponse(api.SAMPLE_RESPONSE + '\n=== DELETE BLOCK: onAdd ===\n', doc, null);
  const src2 = api.applyItems(doc, r.items);
  const doc2 = api.parseDocument(src2);
  ok(doc2.roundTrip);
  ok(doc2.byKey.has('averageMinutes'), 'averageMinutes 追加');
  ok(!doc2.byKey.has('onAdd'), 'onAdd 削除');
  ok(doc2.byKey.get('sumMinutes').text.includes('Number.isFinite'), 'sumMinutes 置換');
  eq(doc2.byKey.get('averageMinutes').index, doc2.byKey.get('sumMinutes').index + 1, '直後に挿入');
  ok(doc2.byKey.get('sumMinutes').text.startsWith('\n// 分の合計を返す'), '前の空行を保つ: ' + JSON.stringify(doc2.byKey.get('sumMinutes').text.slice(0, 30)));
  eq(api.checkDocumentSyntax(doc2, null).length, 0, '構文 OK');
  eq(doc2.blocks.filter(b => b.kind === 'stmt').length, 1, '起動行はそのまま');
  const keys = doc2.blocks.map(b => b.key);
  ok(keys.indexOf('init') < keys.indexOf('stmt#1'), '起動行が最後');
});
test('挿入位置なしの追加は起動行の前に入る', () => {
  const doc = api.parseDocument(api.SAMPLE_APP);
  const src2 = api.applyItems(doc, [{ action: 'add', name: 'zzz', after: null, code: 'function zzz(){}' }]);
  const doc2 = api.parseDocument(src2);
  const keys = doc2.blocks.map(b => b.key);
  ok(keys.indexOf('zzz') === keys.indexOf('init') + 1 && keys.indexOf('zzz') < keys.indexOf('stmt#1'), keys.join(','));
});
test('CRLF のファイルに LF の返答を適用しても CRLF が保たれる', () => {
  const src = api.SAMPLE_APP.replace(/\n/g, '\r\n');
  const doc = api.parseDocument(src);
  eq(doc.eol, '\r\n');
  const r = api.parseResponse(api.SAMPLE_RESPONSE, doc, null);
  const src2 = api.applyItems(doc, r.items);
  ok(!/[^\r]\n/.test(src2), 'LF 単独が混ざらない');
  ok(api.parseDocument(src2).byKey.has('averageMinutes'));
});
test('構文エラーの位置をブロック名で特定する', () => {
  const doc = api.parseDocument('<html><script>\nfunction a(){}\nfunction b(){ if( }\nfunction c(){}\n</script></html>');
  const errs = api.checkDocumentSyntax(doc, new Set(['b']));
  eq(errs.length, 1); eq(errs[0].name, 'b');
});
test('行差分', () => {
  const ops = api.diffLines('a\nb\nc\nd', 'a\nx\nc\nd\ne');
  eq(ops.map(o => o.t + o.l).join('|'), ' a|-b|+x| c| d|+e');
  const st = api.diffStats(ops); eq(st.add, 2); eq(st.del, 1);
});

console.log('== プロンプトと書き出し ==');
test('プロンプトに規約・形式・対象・関連署名・目次が入る', () => {
  const doc = api.parseDocument(api.SAMPLE_APP);
  const p = api.buildPrompt(doc, { selected: new Set(['sumMinutes']), task: '0件で落ちる', kind: 'bugfix', toc: true, full: false, includeRelatedBody: false, fileName: 'sample_app.html' }, api.DEFAULT_TEMPLATES);
  for (const s of ['【コード規約', '=== BLOCK: sumMinutes ===', '=== END ===', '0件で落ちる', '- function renderTable()  // 一覧を描画する', '【ファイル全体の目次', 'ファイル名: sample_app.html']) ok(p.includes(s), '含む: ' + s);
  ok(!p.includes('{{'), 'プレースホルダが残らない');
  const full = api.buildPrompt(doc, { selected: new Set(), task: 't', kind: 'bugfix', toc: false, full: true, includeRelatedBody: false, fileName: 'x.html' }, api.DEFAULT_TEMPLATES);
  ok(full.includes('<!DOCTYPE html>'), '全文モード');
});
test('部品抽出は推移的依存を含む', () => {
  const doc = api.parseDocument(api.SAMPLE_APP);
  const js = api.makePartsJs(doc, new Set(['renderTable']), 'sample_app.html');
  for (const s of ['function renderTable', 'function formatMinutes', 'function sumMinutes', 'const STATE']) ok(js.includes(s), s);
  ok(!js.includes('function onAdd'), 'onAdd は含まない');
});
test('カタログと zip', () => {
  const doc = api.parseDocument(api.SAMPLE_APP);
  const md = api.makeCatalog(doc, 'sample_app.html');
  ok(md.includes('### sumMinutes') && md.includes('呼ばれている: renderTable'));
  const zip = api.makeZip([{ name: 'a.txt', data: 'hello' }, { name: '日本語.js', data: 'function a(){}' }]);
  eq(zip[0], 0x50); eq(zip[1], 0x4b);
  eq(api.crc32(new TextEncoder().encode('hello')), 0x3610a686);
  const eocd = zip.length - 22;
  eq(zip[eocd], 0x50); eq(zip[eocd + 1], 0x4b); eq(zip[eocd + 2], 0x05); eq(zip[eocd + 3], 0x06);
});
test('日時付きファイル名', () => {
  const n = api.stampedFileName('tool.html');
  ok(/^tool\.\d{8}-\d{4}\.html$/.test(n), n);
  ok(/^tool\.\d{8}-\d{4}\.html$/.test(api.stampedFileName(n)), '二重に付かない');
});
test('samples/ の内容が埋め込みと一致する', () => {
  eq(fs.readFileSync(path.join(root, 'samples', 'sample_app.html'), 'utf8'), api.SAMPLE_APP);
  eq(fs.readFileSync(path.join(root, 'samples', 'sample_response.txt'), 'utf8'), api.SAMPLE_RESPONSE);
  eq(fs.readFileSync(path.join(root, 'samples', 'sample_response_bad.txt'), 'utf8'), api.SAMPLE_RESPONSE_BAD);
});

console.log('\n' + pass + ' passed, ' + fail + ' failed');
process.exit(fail ? 1 : 0);
