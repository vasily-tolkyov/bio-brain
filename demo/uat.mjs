/** 万能近似定理在吸引子网络上是否成立？多层迭代激活（后一层以前一层吸引子的激活强度为刺激）会带来什么？
 * 刺激：二维 (x, y) ∈ [0,1]²，用群体编码送进第一层（15 个神经元调谐到 x，15 个调谐到 y）。
 * 目标函数：f_add（可加）、f_mul（乘积）、f_sin（sin·cos，没有可加成分）、f_xor（异或棋盘，不连续）。
 * 实验一：一层吸引子 + 学出的读出，误差随吸引子数怎么变；对照：直接从原始编码读出（线性）。
 *   分两种：只有竞争；再加上吸引子之间学到的侧向联系（越学越"粘"，看精度怎么变）。
 * 实验二：第一层分成"只看 x"与"只看 y"两个模块，叠第二层后能否表示 x、y 的组合。
 * 实验三：在一层够用之后继续无监督地往上叠 5 层：信息、近似误差、抗噪、类别化怎么变。
 * 用法：node demo/uat.mjs → figures/fig6-uat.png、fig7-depth.png */
import fs from 'fs';
import { AttractorLayer, trainReadout as lmsReadout, fitReadout as trainReadout } from '../src/L4-network/attractor-layer.mjs';
import { makeRng } from '../src/core/rng.mjs';
import { Canvas, PALETTE, heat } from '../src/core/png.mjs';

const rng = makeRng(5);
const NX = 15, SIG = 0.08;
const popCode = (x, y) => { const u = new Float64Array(2 * NX); for (let i = 0; i < NX; i++) { const c = i / (NX - 1); u[i] = Math.exp(-((x - c) ** 2) / (2 * SIG * SIG)); u[NX + i] = Math.exp(-((y - c) ** 2) / (2 * SIG * SIG)); } return u; };
const F = {
  f_add: (x, y) => 0.5 * (x + y),
  f_mul: (x, y) => x * y,
  f_sin: (x, y) => 0.5 + 0.5 * Math.sin(2 * Math.PI * x) * Math.cos(2 * Math.PI * y),
  f_xor: (x, y) => ((x > 0.5) !== (y > 0.5) ? 1 : 0),
};
const sample = (n) => Array.from({ length: n }, () => [rng.uniform(), rng.uniform()]);
const trainPts = sample(3000), testPts = sample(1000);
const nrmse = (pred, pts, f) => { const t = pts.map(([x, y]) => f(x, y)), m = t.reduce((a, b) => a + b, 0) / t.length; const sd = Math.sqrt(t.reduce((a, v) => a + (v - m) ** 2, 0) / t.length); return Math.sqrt(pts.reduce((a, [x, y], i) => a + (pred(x, y) - t[i]) ** 2, 0) / pts.length) / sd; };
const log = console.log;
const out = { exp1: {}, exp2: {}, exp3: [] };

/** 给一个"表示"（(x, y) → 向量），对每个目标函数训练读出并给出测试误差（归一化 RMSE：0 = 完美，1 = 只会猜均值） */
function evalRep(rep, fs_ = Object.keys(F)) {
  const Xtr = trainPts.map(([x, y]) => rep(x, y)), Xte = new Map(testPts.map(([x, y], i) => [i, rep(x, y)]));
  const res = {};
  for (const name of fs_) {
    const pred = trainReadout(Xtr, trainPts.map(([x, y]) => F[name](x, y)));
    res[name] = nrmse((x, y) => pred(rep(x, y)), testPts, F[name]);
  }
  return res;
}
const fmtRes = (r) => Object.entries(r).map(([k, v]) => `${k} ${v.toFixed(3)}`).join('  ');

// ── 实验一：吸引子数 N 与近似误差 ──
const raw = evalRep(popCode);
log(`实验一  直接从原始群体编码线性读出：${fmtRes(raw)}`);
out.exp1.raw = raw;
out.exp1.byN = [];
// 两种吸引子层：只有竞争（衡量作为基函数的能力）；再加上吸引子之间学到的侧向联系参与沉降（默认）
const layers1 = {}, layers0 = {};
for (const N of [4, 16, 64, 256, 512]) {
  const t0 = Date.now();
  const L1 = new AttractorLayer(2 * NX, N, { seed: 11 }).train(trainPts.map(([x, y]) => popCode(x, y)), 2);
  const L0 = new AttractorLayer(2 * NX, N, { seed: 11, latGain: 0, settle: 0 }).train(trainPts.map(([x, y]) => popCode(x, y)), 2);
  layers1[N] = L1; layers0[N] = L0;
  const res = evalRep((x, y) => L1.encode(popCode(x, y))), res0 = evalRep((x, y) => L0.encode(popCode(x, y)));
  log(`实验一  一层 ${N} 个吸引子（同时活跃 ${L1.k} 个）：只有竞争 ${fmtRes(res0)}；加上侧向联系 ${fmtRes(res)}（${((Date.now() - t0) / 1000).toFixed(0)} s）`);
  out.exp1.byN.push({ N, res0, res });
}

// 侧向联系随学习变强时，经常一起激活的吸引子被粘成更大的复合吸引子：模式变少、近似变粗
out.exp1.sticky = [];
for (const steps of [1500, 3000, 6000, 12000]) {
  const L = new AttractorLayer(2 * NX, 64, { seed: 11 });
  for (let s = 0; s < steps; s++) { const [x, y] = trainPts[s % trainPts.length]; L.learn(popCode(x, y)); }
  const e = evalRep((x, y) => L.encode(popCode(x, y)), ['f_sin']).f_sin;
  const cells = new Set(testPts.map(([x, y]) => [...L.encode(popCode(x, y))].map((v) => (v > 0.5 ? 1 : 0)).join(''))).size;
  const lat = Math.max(...L.L.map((row) => Math.max(...row)));
  out.exp1.sticky.push({ steps, f_sin: e, cells, lat });
  log(`实验一  64 个吸引子、学习 ${steps} 次：f_sin ${e.toFixed(3)}  激活模式 ${cells} 种  最强侧向联系 ${lat.toFixed(2)}`);
}

// 三因素规则（在线、局部）能逼近最优读出到什么程度
{
  const L = layers0[256], X = trainPts.map(([x, y]) => L.encode(popCode(x, y)));
  const yS = trainPts.map(([x, y]) => F.f_sin(x, y));
  const lms = lmsReadout(X, yS, { epochs: 200 });
  const e = nrmse((x, y) => lms(L.encode(popCode(x, y))), testPts, F.f_sin);
  log(`实验一  256 个吸引子（只有竞争）、f_sin：三因素规则在线学 200 遍的误差 ${e.toFixed(3)}（最优读出 ${out.exp1.byN.find((r) => r.N === 256).res0.f_sin.toFixed(3)}）`);
  out.exp1.lms = e;
}

// ── 实验二：分开的 x、y 模块 → 叠第二层 ──
// 下层每个模块同时活跃 1 个（硬性赢者通吃）或 3 个（按相似程度分级），上层 = 以下层各吸引子激活强度为刺激的吸引子层
out.exp2 = {};
for (const kLow of [1, 3]) {
  const tag = kLow === 1 ? '硬性（每个模块只留 1 个赢者）' : '分级（每个模块 3 个吸引子按强度共同激活）';
  const modX = new AttractorLayer(NX, 16, { seed: 21, k: kLow }).train(trainPts.map(([x, y]) => popCode(x, y).slice(0, NX)), 2);
  const modY = new AttractorLayer(NX, 16, { seed: 22, k: kLow }).train(trainPts.map(([x, y]) => popCode(x, y).slice(NX)), 2);
  const lay1 = (x, y) => { const u = popCode(x, y); return Float64Array.from([...modX.encode(u.slice(0, NX)), ...modY.encode(u.slice(NX))]); };
  const row = { lower: evalRep(lay1), upper: {} };
  log(`实验二  第一层${tag}，只看 x 的 16 个 + 只看 y 的 16 个：${fmtRes(row.lower)}`);
  for (const [N2, k2] of [[64, 4], [256, 16]]) {
    const top = new AttractorLayer(32, N2, { seed: 23, k: k2 }).train(trainPts.map(([x, y]) => lay1(x, y)), 3);
    row.upper[N2] = evalRep((x, y) => top.encode(lay1(x, y)));
    log(`实验二    叠第二层 ${N2} 个吸引子（同时活跃 ${k2} 个）：${fmtRes(row.upper[N2])}`);
  }
  out.exp2[kLow === 1 ? 'hard' : 'graded'] = row;
}

// ── 实验三：一层够用之后继续无监督地往上叠；等宽（64×5）与逐层变窄（64→32→16→8→4） ──
const corr = (a, b) => { const n = a.length, ma = a.reduce((s, v) => s + v, 0) / n, mb = b.reduce((s, v) => s + v, 0) / n; let x = 0, y = 0, z = 0; for (let i = 0; i < n; i++) { x += (a[i] - ma) * (b[i] - mb); y += (a[i] - ma) ** 2; z += (b[i] - mb) ** 2; } return x / Math.sqrt(y * z || 1); };
const quad = (x, y) => (x > 0.5 ? 1 : 0) + (y > 0.5 ? 2 : 0);
function stackStudy(widths, seed0, name) {
  const stack = [layers1[64]];
  for (let d = 1; d < widths.length; d++) {
    const prev = stack.slice();
    const below = (x, y) => prev.reduce((v, L) => L.encode(v), popCode(x, y));
    stack.push(new AttractorLayer(widths[d - 1], widths[d], { seed: seed0 + d }).train(trainPts.map(([x, y]) => below(x, y)), 2));
  }
  const rows = [];
  for (let d = 1; d <= widths.length; d++) {
    const enc = (u) => stack.slice(0, d).reduce((v, L) => L.encode(v), u);
    const Xtr = trainPts.map(([x, y]) => enc(popCode(x, y)));
    const fit = (f) => { const p = trainReadout(Xtr, trainPts.map(([x, y]) => f(x, y))); return (x, y) => p(enc(popCode(x, y))); };
    const f_sin = nrmse(fit(F.f_sin), testPts, F.f_sin), posX = nrmse(fit((x) => x), testPts, (x) => x);
    // 粗类别（四个象限）还能不能读出来：四个"是不是第 q 象限"的读出取最大
    const qs = [0, 1, 2, 3].map((q) => fit((x, y) => (quad(x, y) === q ? 1 : 0)));
    const quadAcc = testPts.filter(([x, y]) => { const s = qs.map((p) => p(x, y)); return s.indexOf(Math.max(...s)) === quad(x, y); }).length / testPts.length;
    // 抗噪：同一刺激加噪声前后表示的相关；类别化：测试集上出现的不同激活模式（强度 > 0.5 的集合）个数
    const nr = makeRng(9);
    const robust = testPts.slice(0, 300).map(([x, y]) => { const u = popCode(x, y), un = u.map((v) => Math.max(0, v + 0.25 * nr.normal())); return corr(enc(u), enc(un)); });
    const cells = new Set(testPts.map(([x, y]) => [...enc(popCode(x, y))].map((v) => (v > 0.5 ? 1 : 0)).join(''))).size;
    const row = { depth: d, width: widths[d - 1], f_sin, posX, quadAcc, robust: robust.reduce((a, b) => a + b, 0) / robust.length, cells };
    rows.push(row);
    log(`实验三${name}  第 ${d} 层（${row.width} 个吸引子）：f_sin ${f_sin.toFixed(3)}  读出 x 的误差 ${posX.toFixed(3)}  象限类别正确率 ${(quadAcc * 100).toFixed(1)}%  加噪前后表示相关 ${row.robust.toFixed(3)}  不同的激活模式 ${cells} 种`);
  }
  return rows;
}
out.exp3 = { equal: stackStudy([64, 64, 64, 64, 64], 30, '（等宽）'), narrow: stackStudy([64, 32, 16, 8, 4], 40, '（变窄）') };

// ── 图 6：实验一的误差曲线 + f_sin 的目标与两种近似；图 7：实验三 ──
fs.mkdirSync('figures', { recursive: true });
const cv = new Canvas(1300, 560);
const Ns = out.exp1.byN.map((r) => r.N), names = Object.keys(F);
const px = (i) => 80 + i * 115, yOf = (v) => 440 - (Math.min(1.2, v) / 1.2) * 380;
cv.text(40, 14, 'EXP1: ERROR 0-1.2 VS ATTRACTORS 4 16 64 256 512  DASHED = RAW', [0, 0, 0], 2);
cv.text(40, 36, 'COLOR = COMPETITION ONLY   BLACK = SIN WITH LEARNED LATERAL LINKS', [0, 0, 0], 1);
cv.rect(80, 440, 480, 1, [150, 150, 150]); cv.rect(80, 60, 1, 380, [150, 150, 150]);
const curve = (get, col, w) => out.exp1.byN.forEach((r, i) => { cv.rect(px(i) - 4, yOf(get(r)) - 4, 8, 8, col); if (i) cv.line(px(i - 1), yOf(get(out.exp1.byN[i - 1])), px(i), yOf(get(r)), col, w); });
names.forEach((n, c) => {
  curve((r) => r.res0[n], PALETTE[c], 3);
  for (let x = 80; x < 560; x += 14) cv.rect(x, yOf(raw[n]), 7, 2, PALETTE[c]);
  cv.text(580, 80 + c * 30, n.slice(2).toUpperCase(), PALETTE[c], 2);
});
curve((r) => r.res.f_sin, [0, 0, 0], 2);
const grid = (f, x0, y0, title) => { cv.text(x0, y0 - 24, title, [0, 0, 0], 2); for (let i = 0; i < 60; i++) for (let j = 0; j < 60; j++) cv.rect(x0 + i * 3, y0 + (59 - j) * 3, 3, 3, heat(f(i / 59, j / 59))); };
const L256 = layers0[256];
const pS = trainReadout(trainPts.map(([x, y]) => L256.encode(popCode(x, y))), trainPts.map(([x, y]) => F.f_sin(x, y)));
const pR = trainReadout(trainPts.map(([x, y]) => popCode(x, y)), trainPts.map(([x, y]) => F.f_sin(x, y)));
grid(F.f_sin, 720, 80, 'TARGET F_SIN');
grid((x, y) => pS(L256.encode(popCode(x, y))), 920, 80, '256 ATTRACTORS');
grid((x, y) => pR(popCode(x, y)), 1120, 80, 'RAW LINEAR');
grid(F.f_xor, 720, 330, 'TARGET F_XOR');
const pX = trainReadout(trainPts.map(([x, y]) => L256.encode(popCode(x, y))), trainPts.map(([x, y]) => F.f_xor(x, y)));
grid((x, y) => pX(L256.encode(popCode(x, y))), 920, 330, '256 ATTRACTORS');
const pXr = trainReadout(trainPts.map(([x, y]) => popCode(x, y)), trainPts.map(([x, y]) => F.f_xor(x, y)));
grid((x, y) => pXr(popCode(x, y)), 1120, 330, 'RAW LINEAR');
cv.save('figures/fig6-uat.png');

// 图 7 左：实验二的 f_sin / f_xor 误差柱（R = 原始编码线性读出，H = 硬性下层，G = 分级下层，后面的数字 = 再叠一层的吸引子数）
const c7 = new Canvas(1460, 470);
c7.text(30, 15, 'EXP2: ERROR OF SIN (GREEN) XOR (ORANGE)', [0, 0, 0], 2);
const bars = [['R', raw], ['H', out.exp2.hard.lower], ['H64', out.exp2.hard.upper[64]], ['H256', out.exp2.hard.upper[256]], ['G', out.exp2.graded.lower], ['G64', out.exp2.graded.upper[64]], ['G256', out.exp2.graded.upper[256]]];
bars.forEach(([lab, r], b) => {
  const x = 40 + b * 62, base = 400, h = (v) => (Math.min(1.2, v) / 1.2) * 330;
  c7.rect(x, base - h(r.f_sin), 22, h(r.f_sin), PALETTE[2]); c7.rect(x + 24, base - h(r.f_xor), 22, h(r.f_xor), PALETTE[3]);
  c7.text(x, base + 12, lab, [0, 0, 0], 2);
});
c7.rect(40, 400, 440, 1, [150, 150, 150]);
// 图 7 中、右：实验三两种堆叠，各项指标随层数
const panel = (rows, x0, title) => {
  c7.text(x0, 15, title, [0, 0, 0], 2);
  const series = [['SIN ERR', (r) => r.f_sin / 1.2], ['X ERR', (r) => r.posX / 1.2], ['QUADRANT', (r) => r.quadAcc], ['NOISE ROBUST', (r) => r.robust], ['PATTERNS', (r) => r.cells / Math.max(...rows.map((q) => q.cells))]];
  series.forEach(([lab, f], s) => {
    rows.forEach((r, i) => { const X = x0 + 20 + i * 70, Y = 400 - f(r) * 330; c7.rect(X - 4, Y - 4, 8, 8, PALETTE[s]); if (i) c7.line(x0 + 20 + (i - 1) * 70, 400 - f(rows[i - 1]) * 330, X, Y, PALETTE[s], 2); });
    c7.text(x0 + 330, 60 + s * 26, lab, PALETTE[s], 2);
  });
  rows.forEach((r, i) => c7.text(x0 + 12 + i * 70, 412, String(r.width), [0, 0, 0], 2));
  c7.rect(x0 + 20, 400, 290, 1, [150, 150, 150]);
};
panel(out.exp3.equal, 520, 'EXP3 EQUAL WIDTH');
panel(out.exp3.narrow, 960, 'EXP3 NARROWING');
c7.save('figures/fig7-depth.png');
fs.writeFileSync('figures/uat-results.json', JSON.stringify(out, null, 1));
log('写入 figures/fig6-uat.png、figures/fig7-depth.png');
