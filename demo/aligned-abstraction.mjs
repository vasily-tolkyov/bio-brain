/** 抽象的程度相同，丢掉的细节对不对？同样 4 个吸引子：无监督形成的划分 vs 用"后果预测误差"作第三因子塑造的划分。
 * 世界：(x, y) 均匀；"发生什么"由所在象限决定。
 * 误差引导：每个吸引子学一个后果预测 P_i（delta 规则，相当于读出）；原型更新 Δw_i = η·m·(输入 − w_i)，
 *   m = 赢者对实际后果的预测 − 机会水平（像多巴胺：预测得比机会好就靠近这个刺激，比机会差就远离）。
 * 用法：node demo/aligned-abstraction.mjs（几秒） */
import { AttractorLayer, fitReadout } from '../src/L4-network/attractor-layer.mjs';
import { makeRng } from '../src/core/rng.mjs';

const NX = 15, SIG = 0.08;
const popCode = (x, y) => { const u = new Float64Array(2 * NX); for (let i = 0; i < NX; i++) { const c = i / (NX - 1); u[i] = Math.exp(-((x - c) ** 2) / (2 * SIG * SIG)); u[NX + i] = Math.exp(-((y - c) ** 2) / (2 * SIG * SIG)); } return u; };
const norm = (v) => { let s = 0; for (const x of v) s += x * x; s = Math.sqrt(s) || 1; return v.map((x) => x / s); };
const quad = (x, y) => (x > 0.5 ? 1 : 0) + (y > 0.5 ? 2 : 0);
const f_xor = (x, y) => ((x > 0.5) !== (y > 0.5) ? 1 : 0);
const rng = makeRng(5);
const sample = (n) => Array.from({ length: n }, () => [rng.uniform(), rng.uniform()]);
const train = sample(3000), test = sample(1000);
const nrmse = (pred, f) => { const t = test.map(([x, y]) => f(x, y)), m = t.reduce((a, b) => a + b, 0) / t.length; const sd = Math.sqrt(t.reduce((a, v) => a + (v - m) ** 2, 0) / t.length); return Math.sqrt(test.reduce((a, [x, y], i) => a + (pred(x, y) - t[i]) ** 2, 0) / test.length) / sd; };
const centre = (w) => [w.slice(0, NX), w.slice(NX)].map((h) => h.reduce((a, v, i) => a + v * i, 0) / h.reduce((a, v) => a + v, 0) / (NX - 1));

function evaluate(L, name) {
  const enc = (x, y) => L.encode(popCode(x, y));
  const X = train.map(([x, y]) => enc(x, y));
  const qs = [0, 1, 2, 3].map((q) => fitReadout(X, train.map(([x, y]) => (quad(x, y) === q ? 1 : 0))));
  const acc = test.filter(([x, y]) => { const v = enc(x, y), s = qs.map((p) => p(v)); return s.indexOf(Math.max(...s)) === quad(x, y); }).length / test.length;
  const px = fitReadout(X, train.map(([x]) => x)), pxor = fitReadout(X, train.map(([x, y]) => f_xor(x, y)));
  console.log(`${name}：象限正确率 ${(acc * 100).toFixed(1)}%  f_xor 误差 ${nrmse((x, y) => pxor(enc(x, y)), f_xor).toFixed(3)}  读出 x 的误差 ${nrmse((x, y) => px(enc(x, y)), (x) => x).toFixed(3)}  原型中心 ${L.W.map((w) => `(${centre(w).map((c) => c.toFixed(2)).join(',')})`).join(' ')}`);
}

for (const seed of [11, 12, 13]) {
  evaluate(new AttractorLayer(2 * NX, 4, { seed, k: 1 }).train(train.map(([x, y]) => popCode(x, y)), 2), `[种子 ${seed}] 无监督`);
  const E = new AttractorLayer(2 * NX, 4, { seed, k: 1 });
  const P = Array.from({ length: 4 }, () => new Float64Array(4).fill(0.25));
  for (let ep = 0; ep < 2; ep++) for (const [x, y] of train) {
    const { r, un } = E.response(popCode(x, y));
    const i = r.indexOf(Math.max(...r)), q = quad(x, y), m = P[i][q] - 0.25;
    const w = E.W[i]; for (let j = 0; j < E.D; j++) w[j] += E.eta * 4 * m * (un[j] - w[j]); E.W[i] = norm(w);
    for (let c = 0; c < 4; c++) P[i][c] += 0.05 * ((c === q ? 1 : 0) - P[i][c]);
    for (let a = 0; a < 4; a++) { E.wins[a] = 0.99 * E.wins[a] + 0.01 * (a === i ? 1 : 0); E.theta[a] += E.etaTheta * (E.wins[a] - 0.25); }
  }
  evaluate(E, `[种子 ${seed}] 误差引导`);
}
