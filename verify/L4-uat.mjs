/** 第 4 层验证：万能近似定理与多层迭代激活（速率版吸引子层，约半分钟）
 *   ① 一层吸引子 + 学出的读出能近似原始编码线性读不出的函数，误差随吸引子数下降；三因素规则接近最优读出；
 *   ② 分开的模块（只看 x / 只看 y）读不出交互；以下层吸引子激活强度为刺激的第二层长出合取吸引子后可以读出，前提是下层按强度分级；
 *   ③ 无监督地往上叠不增加信息（数据处理不等式）；逐层变窄时类别变少、细节丢失。
 * 完整版（更多吸引子数、四个目标函数、图）见 demo/uat.mjs。 */
import { AttractorLayer, fitReadout, trainReadout } from '../src/L4-network/attractor-layer.mjs';
import { makeRng } from '../src/core/rng.mjs';
import { checker, fmt } from '../src/core/check.mjs';

const NX = 15, SIG = 0.08;
const popCode = (x, y) => { const u = new Float64Array(2 * NX); for (let i = 0; i < NX; i++) { const c = i / (NX - 1); u[i] = Math.exp(-((x - c) ** 2) / (2 * SIG * SIG)); u[NX + i] = Math.exp(-((y - c) ** 2) / (2 * SIG * SIG)); } return u; };
const f_add = (x, y) => 0.5 * (x + y), f_sin = (x, y) => 0.5 + 0.5 * Math.sin(2 * Math.PI * x) * Math.cos(2 * Math.PI * y), f_x = (x) => x;

export function verifyL4Uat() {
  const c = checker('第 4 层 万能近似定理与多层迭代激活');
  const rng = makeRng(5);
  const sample = (n) => Array.from({ length: n }, () => [rng.uniform(), rng.uniform()]);
  const train = sample(1500), test = sample(600);
  const nrmse = (pred, f) => { const t = test.map(([x, y]) => f(x, y)), m = t.reduce((a, b) => a + b, 0) / t.length; const sd = Math.sqrt(t.reduce((a, v) => a + (v - m) ** 2, 0) / t.length); return Math.sqrt(test.reduce((a, [x, y], i) => a + (pred(x, y) - t[i]) ** 2, 0) / test.length) / sd; };
  const err = (rep, f, fit = fitReadout) => { const p = fit(train.map(([x, y]) => rep(x, y)), train.map(([x, y]) => f(x, y))); return nrmse((x, y) => p(rep(x, y)), f); };

  // ① 一层吸引子
  const raw = { add: err(popCode, f_add), sin: err(popCode, f_sin) };
  c.check('原始群体编码的线性读出只能表示可加的函数：f_add 几乎完美，f_sin（没有可加成分）读不出', raw.add < 0.05 && raw.sin > 0.95, `f_add ${fmt(raw.add, 3)}，f_sin ${fmt(raw.sin, 3)}`);
  const byN = [16, 64, 256].map((N) => { const L = new AttractorLayer(2 * NX, N, { seed: 11 }).train(train.map(([x, y]) => popCode(x, y)), 2); return { N, L, sin: err((x, y) => L.encode(popCode(x, y)), f_sin) }; });
  c.check('一层吸引子 + 读出能近似 f_sin，误差随吸引子数单调下降', byN[0].sin > byN[1].sin && byN[1].sin > byN[2].sin && byN[2].sin < 0.4, byN.map((r) => `${r.N} 个 ${fmt(r.sin, 3)}`).join('，'));
  const L64 = byN[1].L, lms = err((x, y) => L64.encode(popCode(x, y)), f_sin, (X, y) => trainReadout(X, y, { epochs: 100 }));
  c.check('三因素规则（在线、局部）学到的读出接近最优读出', lms < byN[1].sin + 0.15, `64 个吸引子：三因素 ${fmt(lms, 3)}，最优 ${fmt(byN[1].sin, 3)}`);

  // ② 分开的模块 → 第二层
  const exp2 = {};
  for (const kLow of [1, 3]) {
    const mx = new AttractorLayer(NX, 16, { seed: 21, k: kLow }).train(train.map(([x, y]) => popCode(x, y).slice(0, NX)), 2);
    const my = new AttractorLayer(NX, 16, { seed: 22, k: kLow }).train(train.map(([x, y]) => popCode(x, y).slice(NX)), 2);
    const low = (x, y) => { const u = popCode(x, y); return Float64Array.from([...mx.encode(u.slice(0, NX)), ...my.encode(u.slice(NX))]); };
    const top = new AttractorLayer(32, 64, { seed: 23, k: 4 }).train(train.map(([x, y]) => low(x, y)), 3);
    exp2[kLow] = { low: err(low, f_sin), top: err((x, y) => top.encode(low(x, y)), f_sin) };
  }
  c.check('分开的模块（只看 x、只看 y）读不出 x 与 y 的交互', exp2[1].low > 0.95 && exp2[3].low > 0.95, `硬性 ${fmt(exp2[1].low, 3)}，分级 ${fmt(exp2[3].low, 3)}`);
  c.check('以下层激活强度为刺激的第二层长出合取吸引子，交互变得可读；下层只传赢者时不行', exp2[3].top < 0.65 && exp2[1].top > 0.9, `下层分级 ${fmt(exp2[3].top, 3)}，下层硬性 ${fmt(exp2[1].top, 3)}`);

  // ③ 无监督地往上叠
  const study = (widths) => {
    const stack = [L64];
    for (let d = 1; d < widths.length; d++) { const prev = stack.slice(); stack.push(new AttractorLayer(widths[d - 1], widths[d], { seed: 30 + d }).train(train.map(([x, y]) => prev.reduce((v, L) => L.encode(v), popCode(x, y))), 2)); }
    return widths.map((w, d) => {
      const enc = (x, y) => stack.slice(0, d + 1).reduce((v, L) => L.encode(v), popCode(x, y));
      return { w, posX: err(enc, f_x), cells: new Set(test.map(([x, y]) => [...enc(x, y)].map((v) => (v > 0.5 ? 1 : 0)).join(''))).size };
    });
  };
  const eq = study([64, 64, 64]), nar = study([64, 16, 4]);
  c.check('等宽无监督叠层不增加信息：读出位置 x 的误差逐层只增不减', eq[1].posX >= eq[0].posX - 0.01 && eq[2].posX >= eq[1].posX - 0.01, eq.map((r, d) => `第 ${d + 1} 层 ${fmt(r.posX, 3)}`).join('，'));
  c.check('逐层变窄是粗粒化：激活模式变少、位置细节丢失', nar[2].cells < nar[1].cells && nar[1].cells < nar[0].cells && nar[2].posX > nar[0].posX + 0.3,
    nar.map((r) => `${r.w} 个吸引子：${r.cells} 种模式、x 误差 ${fmt(r.posX, 3)}`).join('；'));
  return { ok: c.done() };
}

if (import.meta.url === `file:///${process.argv[1].replace(/\\/g, '/')}`) process.exit(verifyL4Uat().ok ? 0 : 1);
