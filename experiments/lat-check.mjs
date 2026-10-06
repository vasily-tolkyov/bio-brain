// 侧向联系（吸引子之间的赫布增强）与训练时长对近似精度、激活模式数的影响
import { AttractorLayer, fitReadout } from '../src/L4-network/attractor-layer.mjs';
import { makeRng } from '../src/core/rng.mjs';

const NX = 15, SIG = 0.08;
const popCode = (x, y) => { const u = new Float64Array(2 * NX); for (let i = 0; i < NX; i++) { const c = i / (NX - 1); u[i] = Math.exp(-((x - c) ** 2) / (2 * SIG * SIG)); u[NX + i] = Math.exp(-((y - c) ** 2) / (2 * SIG * SIG)); } return u; };
const f_sin = (x, y) => 0.5 + 0.5 * Math.sin(2 * Math.PI * x) * Math.cos(2 * Math.PI * y);
const rng = makeRng(5);
const sample = (n) => Array.from({ length: n }, () => [rng.uniform(), rng.uniform()]);
const train = sample(3000), test = sample(1000);
const nrmse = (pred) => { const t = test.map(([x, y]) => f_sin(x, y)), m = t.reduce((a, b) => a + b, 0) / t.length; const sd = Math.sqrt(t.reduce((a, v) => a + (v - m) ** 2, 0) / t.length); return Math.sqrt(test.reduce((a, [x, y], i) => a + (pred(x, y) - t[i]) ** 2, 0) / test.length) / sd; };
for (const latGain of [0, 0.2]) for (const steps of [1500, 3000, 6000, 12000]) {
  const L = new AttractorLayer(2 * NX, 64, { seed: 11, latGain });
  for (let s = 0; s < steps; s++) { const [x, y] = train[s % train.length]; L.learn(popCode(x, y)); }
  const p = fitReadout(train.map(([x, y]) => L.encode(popCode(x, y))), train.map(([x, y]) => f_sin(x, y)));
  const cells = new Set(test.map(([x, y]) => [...L.encode(popCode(x, y))].map((v) => (v > 0.5 ? 1 : 0)).join(''))).size;
  let lat = 0; for (const row of L.L) for (const v of row) lat = Math.max(lat, v);
  console.log(`侧向增益 ${latGain}  学习 ${steps} 次：f_sin ${nrmse((x, y) => p(L.encode(popCode(x, y)))).toFixed(3)}  激活模式 ${cells} 种  最强侧向联系 ${lat.toFixed(2)}`);
}
