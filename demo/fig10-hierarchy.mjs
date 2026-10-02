/** 多层脉冲网络：前馈抽象 ↑、层内有向通道 →、反馈 ↓ 接成一个回路；层数 1–4 的效果变化。
 * 每一层都是完整的脉冲网络（LIF 神经元、可塑突触、吸引子、有向通道），层间是真实突触（见 src/L4-network/layered-network.mjs）。
 * 世界与逐层训练见 demo/lib-hierarchy.mjs：物体 X = A + D、Y = C + B（第一层学到 A→D、C→B），X 之后是 Y。
 * 这个网络以群体爆发工作：一次刺激引起十几毫秒的集体爆发，之后沉默上百毫秒；学到的预测表现为同一次爆发里被预测的组提前被拉起。
 * 测试（全部冻结）：
 *   预测：只给一个与训练时一样短的 X——各层 Y 的组是否被拉起；第一层 Y 的组（从不被刺激）的膜电位是否被抬高（阈下预激活）；
 *   易化：短 X 之后紧接 Y vs 空白之后 Y——第一层对 Y 的早期响应；
 *   想象：第一层没有刺激，直接驱动第二层 X 的组——反馈能否在第一层重建 X、传下预测的 Y；
 *   对照：切断反馈；上层没经历物体序列（没有物体级通道）。两者的差别才是"预测"本身的作用；
 *   补全：只给 X 的一个部件的一半成员；不变性：同一物体两种互不重叠的刺激；抗噪：残缺 + 噪声；延迟与 ATP。
 * 用法：node demo/fig10-hierarchy.mjs（首次运行先逐层训练，之后读缓存）→ figures/fig10-hierarchy.png */
import fs from 'fs';
import { cfg, NE, proto, OBJ, mean, trainedStack, SEQ_ITEM_UPPER } from './lib-hierarchy.mjs';
import { inputRate, spikesIn } from '../src/L4-network/layered-network.mjs';
import { makeRng } from '../src/core/rng.mjs';
import { Canvas, PALETTE } from '../src/core/png.mjs';

let log = console.log;
const BL = ['A', 'B', 'C', 'D'], blocks = BL.map(proto), BIN = 5;
const XB = OBJ.X.map((p) => BL.indexOf(p)), YB = OBJ.Y.map((p) => BL.indexOf(p));
const t0 = Date.now(), el = () => `${((Date.now() - t0) / 1000).toFixed(0)} s`;
const corr = (a, b) => { const n = a.length, ma = mean(a), mb = mean(b); let x = 0, y = 0, z = 0; for (let i = 0; i < n; i++) { x += (a[i] - ma) * (b[i] - mb); y += (a[i] - ma) ** 2; z += (b[i] - mb) ** 2; } return x / Math.sqrt(y * z || 1); };

/** 一段刺激序列重复 reps 次：各层各组 5 ms 分箱的放电率、各层每个神经元的放电数、ATP、第一层 Y 的组的平均膜电位（1 ms） */
function trial(S, segments, { reps = 6, pre = 400 } = {}) {
  const total = segments.reduce((s, [ms]) => s + ms, 0), nb = Math.round(total / BIN);
  const course = S.layers.map(() => blocks.map(() => new Float64Array(nb)));
  const cells = S.layers.map(() => new Float64Array(NE)), v = new Float64Array(Math.round(total));
  const vIds = YB.flatMap((q) => blocks[q].map((a) => S.layers[0].E[a]));
  let atp = 0;
  for (let r = 0; r < reps; r++) {
    S.net.run(pre, { rate: inputRate(S), learn: false });
    const ts = S.net.t, a0 = S.net.ledger.total();
    let step = 0;
    const onStep = () => { if (step++ % 10) return; const ms = step / 10 | 0; if (ms < v.length) { let s = 0; for (const i of vIds) s += S.net.V[i]; v[ms] += s / vIds.length / reps; } };
    for (const [ms, drives, extra] of segments) S.net.run(ms, { rate: inputRate(S, drives, extra), learn: false, onStep });
    atp += (S.net.ledger.total() - a0) / reps;
    S.layers.forEach((L, k) => spikesIn(S.net, L.E, ts, ts + total).forEach((s, a) => { cells[k][a] += s.length / reps; const q = Math.floor(a / cfg.M); if (q < 4) for (const t of s) { const b = Math.floor((t - ts) / BIN); if (b < nb) course[k][q][b] += 1 / (cfg.M * BIN / 1000) / reps; } }));
  }
  return { course, cells, atp, v };
}
const avg = (arr, from, to) => mean(Array.from(arr).slice(Math.round(from / BIN), Math.round(to / BIN)));

function study(S, name, rng) {
  const at = (ids, hz = cfg.stim) => new Map(ids.map((i) => [i, hz]));
  const pick = (L, n) => rng.shuffle(proto(L).slice()).slice(0, n);
  const objStim = (o) => at(OBJ[o].flatMap((p) => pick(p, 12)));
  const K = S.layers.length, T = SEQ_ITEM_UPPER, [x0, x1] = OBJ.X, partner = BL.indexOf(x1);
  const Xs = at(OBJ.X.flatMap(proto), cfg.seqStim), Ys = at(OBJ.Y.flatMap(proto), cfg.seqStim);
  const rest = trial(S, [[200, new Map()]], { reps: 3 });
  const base = rest.course.map((L) => L.map((b) => avg(b, 0, 200))), vRest = mean(Array.from(rest.v));
  const P = trial(S, [[T, Xs], [200 - T, new Map()]], { reps: 16 });
  const F = trial(S, [[T, Xs], [40, Ys], [160 - T, new Map()]], { reps: 12 });
  const N = trial(S, [[T, new Map()], [40, Ys], [160 - T, new Map()]], { reps: 12 });
  const Lg = trial(S, [[150, objStim('X')], [100, new Map()]], { reps: 6 });
  const C = trial(S, [[150, at(pick(x0, 10))]], { reps: 6 });
  const v1 = trial(S, [[150, at([...proto(x0).slice(0, 10), ...proto(x1).slice(0, 10)])]], { reps: 4 });
  const v2 = trial(S, [[150, at([...proto(x0).slice(10), ...proto(x1).slice(10)])]], { reps: 4 });
  const xIds = OBJ.X.flatMap(proto), noise = rng.shuffle([...Array(NE).keys()].filter((i) => !xIds.includes(i))).slice(0, 8);
  const D = trial(S, [[150, at([...pick(x0, 6), ...pick(x1, 6), ...noise])]], { reps: 6 });
  // 想象：第一层没有任何刺激，直接驱动第二层 X 的组——反馈能否在第一层重建 X、传下预测的 Y
  const I = K > 1 ? trial(S, [[T, new Map(), new Map(XB.flatMap((q) => blocks[q].map((a) => [S.layers[1].E[a], cfg.seqStim])))], [120 - T, new Map()]], { reps: 16 }) : null;
  const yAct = (Tr, k, from, to) => mean(YB.map((q) => avg(Tr.course[k][q], from, to) - base[k][q]));
  const xIndex = (Tr, k) => { const a = mean(XB.map((q) => avg(Tr.course[k][q], 0, 150))), b = mean(YB.map((q) => avg(Tr.course[k][q], 0, 150))); return (a - b) / (a + b + 1e-9); };
  const latency = (k) => { const ab = Array.from(Lg.course[k][XB[0]]).map((x, b) => (x + Lg.course[k][XB[1]][b]) / 2), b0 = (base[k][XB[0]] + base[k][XB[1]]) / 2, peak = Math.max(...ab.slice(0, 30)); const b = ab.findIndex((x) => x >= b0 + 0.5 * (peak - b0)); return b < 0 ? null : (b + 0.5) * BIN; };
  const r = {
    name, K,
    pred: S.layers.map((_, k) => yAct(P, k, 0, T + 10)),
    vPeak: Math.max(...Array.from(P.v).slice(0, T + 20)) - vRest, vMean: mean(Array.from(P.v).slice(5, T + 10)) - vRest, vTrace: Array.from(P.v),
    facil: yAct(F, 0, T, T + 20), noCtx: yAct(N, 0, T, T + 20),
    complete: S.layers.map((_, k) => avg(C.course[k][partner], 0, 150) - base[k][partner]),
    invariance: S.layers.map((_, k) => corr(v1.cells[k], v2.cells[k])),
    clean: S.layers.map((_, k) => xIndex(Lg, k)), noisy: S.layers.map((_, k) => xIndex(D, k)),
    latency: S.layers.map((_, k) => latency(k)), atp: Lg.atp,
    imagery: I && { x1: mean(XB.map((q) => avg(I.course[0][q], 0, 60) - base[0][q])), y1: yAct(I, 0, 0, 60), y2: yAct(I, 1, 0, 60), vY: Math.max(...Array.from(I.v).slice(0, 60)) - mean(Array.from(I.v).slice(0, 3)) },
    course: P.course.map((L) => L.map((b) => Array.from(b))),
  };
  log(`\n【${name}】（${el()}）`);
  log(`  预测（只给 ${T} ms 的 X）：各层 Y 的组（${OBJ.Y.join('、')}）在 0–${T + 10} ms 比静息多 ${r.pred.map((x, k) => `第${k + 1}层 ${x.toFixed(1)}`).join('、')} Hz；第一层 Y 的组膜电位比静息高 ${r.vMean.toFixed(2)} mV（平均）、${r.vPeak.toFixed(2)} mV（峰值）`);
  log(`  易化：第一层 Y 的组在 Y 出现后头 20 ms 比静息多——短 X 之后 ${r.facil.toFixed(1)} Hz，空白之后 ${r.noCtx.toFixed(1)} Hz`);
  if (r.imagery) log(`  想象（第一层没有刺激，只驱动第二层 X 的组 ${T} ms）：第一层 X 的组 ${r.imagery.x1.toFixed(1)} Hz、Y 的组 ${r.imagery.y1.toFixed(1)} Hz（膜电位最多升高 ${r.imagery.vY.toFixed(2)} mV）；第二层 Y 的组 ${r.imagery.y2.toFixed(1)} Hz`);
  log(`  补全（只给 ${x0} 的 10 个）：${x1} 组比静息多 ${r.complete.map((x, k) => `第${k + 1}层 ${x.toFixed(1)}`).join('、')} Hz`);
  log(`  不变性（X 的两种互不重叠的刺激，各层响应相关）：${r.invariance.map((x, k) => `第${k + 1}层 ${x.toFixed(2)}`).join('、')}`);
  log(`  X 的辨识度（(X 的组 − Y 的组)/(两者之和)）：完整 ${r.clean.map((x) => x.toFixed(2)).join(' / ')}，残缺 + 噪声 ${r.noisy.map((x) => x.toFixed(2)).join(' / ')}`);
  log(`  各层对 X 的响应延迟 ${r.latency.map((x) => (x === null ? '—' : x.toFixed(0) + ' ms')).join(' / ')}；250 ms 内整个网络 ${(r.atp / 1e9).toFixed(0)}×10⁹ ATP`);
  return r;
}

/** 每层学到的结构：物体内绑定、物体间正向（X→Y）与反向（Y→X）、反馈指向（同一物体 vs 另一物体） */
function structure(S) {
  const blk = (a) => (a < 4 * cfg.M ? Math.floor(a / cfg.M) : -1);
  return S.layers.map((L) => {
    const m = (g, from, to, preL, postL) => { let s = 0, n = 0; for (let k = 0; k < g.pre.length; k++) { const a = preL.indexOf(g.pre[k]), b = postL.indexOf(g.post[k]); if (a < 0 || b < 0) continue; if (from.includes(blk(a)) && to.includes(blk(b))) { s += g.rho[k]; n++; } } return n ? s / n : 0; };
    const within = [...XB.map((a) => m(L.EE, [a], XB.filter((b) => b !== a), L.E, L.E)), ...YB.map((a) => m(L.EE, [a], YB.filter((b) => b !== a), L.E, L.E))].reduce((s, x) => s + x, 0) / 4;
    const out = { within, fwd: m(L.EE, XB, YB, L.E, L.E), bwd: m(L.EE, YB, XB, L.E, L.E),
      innerFwd: (m(L.EE, [XB[0]], [XB[1]], L.E, L.E) + m(L.EE, [YB[0]], [YB[1]], L.E, L.E)) / 2, innerBwd: (m(L.EE, [XB[1]], [XB[0]], L.E, L.E) + m(L.EE, [YB[1]], [YB[0]], L.E, L.E)) / 2 };
    if (L.FB) { const lo = S.layers[L.k - 2]; out.fbSame = (m(L.FB, YB, YB, L.E, lo.E) + m(L.FB, XB, XB, L.E, lo.E)) / 2; out.fbCross = (m(L.FB, YB, XB, L.E, lo.E) + m(L.FB, XB, YB, L.E, lo.E)) / 2; }
    return out;
  });
}

export function runHierarchy({ log: logTo = console.log, figures = true } = {}) {
log = logTo;
const rng = makeRng(77), results = [];
const stacks = [1, 2, 3, 4].map((k) => { const S = trainedStack(k, { log }); log(`${k} 层网络就绪（${el()}）`); return S; });
const st = structure(stacks[3]);
log(`\n学到的结构（4 层网络）：${st.map((s, k) => `第${k + 1}层 物体内 ${s.within.toFixed(2)}，X→Y ${s.fwd.toFixed(2)}，Y→X ${s.bwd.toFixed(2)}${s.fbSame !== undefined ? `，反馈 同物体 ${s.fbSame.toFixed(2)} / 跨物体 ${s.fbCross.toFixed(2)}` : ''}`).join('；')}`);
for (const S of stacks) results.push(study(S, `${S.layers.length} 层`, rng));
const cut = trainedStack(2); cut.layers[1].FB.scale.fill(0);
results.push(study(cut, '2 层，切断反馈', rng));
results.push(study(trainedStack(2, { variant: 'noseq', log }), '2 层，上层没经历物体序列', rng));
const [r1, r2, r3, r4, rCut, rNo] = results;
if (!figures) return { structure: st, results };
log(`\n预测本身的作用（2 层减去"上层没经历物体序列"）：第二层 Y 的组 +${(r2.pred[1] - rNo.pred[1]).toFixed(1)} Hz；第一层 Y 的组膜电位 +${(r2.vMean - rNo.vMean).toFixed(2)} mV（平均）；Y 到来时第一层早期响应 ${(r2.facil - rNo.facil).toFixed(1)} Hz；想象时第一层 Y 的组膜电位 +${(r2.imagery.vY - rNo.imagery.vY).toFixed(2)} mV`);

// ── 图 10 ──
fs.mkdirSync('figures', { recursive: true });
const cv = new Canvas(1500, 860);
const lineSeries = (arr, x0, y0, w, h, lo, hi, col, width = 2, dashed = false) => { for (let i = 1; i < arr.length; i++) { if (dashed && i % 4 < 2) continue; const y = (v) => y0 + h - ((Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo)) * h; cv.line(x0 + ((i - 1) / arr.length) * w, y(arr[i - 1]), x0 + (i / arr.length) * w, y(arr[i]), col, width); } };
cv.text(20, 12, 'SEE X: LAYER 1 Y GROUPS MEMBRANE POTENTIAL', [0, 0, 0], 2);
cv.text(20, 36, 'BLUE = 2 LAYERS  GRAY = NO X-THEN-Y  DASHED = FB CUT  (0-120 MS, X 0-40)', [0, 0, 0], 1);
const vs = [r2, rNo, rCut].map((r) => r.vTrace.slice(0, 120)), vLo = Math.min(...vs.flat()) - 0.2, vHi = Math.max(...vs.flat()) + 0.2;
lineSeries(vs[0], 40, 60, 640, 240, vLo, vHi, PALETTE[1], 3); lineSeries(vs[1], 40, 60, 640, 240, vLo, vHi, [150, 150, 150], 3); lineSeries(vs[2], 40, 60, 640, 240, vLo, vHi, [0, 0, 0], 1, true);
cv.rect(40, 300, 640, 1, [150, 150, 150]); cv.rect(40 + 640 * SEQ_ITEM_UPPER / 120, 60, 1, 240, [200, 200, 200]);
cv.text(760, 12, 'IMAGINE X (DRIVE LAYER 2 ONLY): LAYER 1 RATES', [0, 0, 0], 2);
cv.text(760, 36, 'PER NETWORK: X GROUPS (RED)  Y GROUPS (BLUE)  LAYER 2 Y (GRAY)', [0, 0, 0], 1);
const imgs = [[r2, '2 LAYERS'], [rNo, '2 NO SEQ'], [rCut, '2 FB CUT']], iHi = Math.max(...imgs.flatMap(([r]) => [r.imagery.x1, r.imagery.y1, r.imagery.y2]), 1);
imgs.forEach(([r, lb], i) => { [[r.imagery.x1, PALETTE[0]], [r.imagery.y1, PALETTE[1]], [r.imagery.y2, [150, 150, 150]]].forEach(([v, col], j) => { const h = (Math.max(0, v) / iHi) * 240; cv.rect(800 + i * 210 + j * 50, 300 - h, 40, h, col); }); cv.text(800 + i * 210, 312, lb, [0, 0, 0], 1); });
cv.rect(770, 300, 640, 1, [150, 150, 150]);
cv.text(20, 350, 'PREDICTED Y GROUPS ABOVE REST WHILE X SHOWN, BY LAYER (HZ)', [0, 0, 0], 2);
const lab = ['1 LAYER', '2 LAYERS', '3 LAYERS', '4 LAYERS', '2 FB CUT', '2 NO SEQ'];
const pHi = Math.max(...results.flatMap((r) => r.pred), 1);
results.forEach((r, i) => { r.pred.forEach((x, k) => { const h = (Math.max(0, x) / pHi) * 280; cv.rect(40 + i * 115 + k * 22, 700 - h, 18, h, PALETTE[k]); }); cv.text(30 + i * 115, 712, lab[i], [0, 0, 0], 1); });
cv.rect(40, 700, 690, 1, [150, 150, 150]);
cv.text(30, 740, 'BAR COLOR = LAYER 1 2 3 4', [0, 0, 0], 1);
cv.text(760, 350, 'BY DEPTH: TOP INVARIANCE  NOISY ID  ATP/4', [0, 0, 0], 2);
cv.text(760, 374, 'BLUE = INVARIANCE OF TOP LAYER   GREEN = NOISY X ID OF TOP LAYER   GRAY = ATP RELATIVE TO 1 LAYER / 4', [0, 0, 0], 1);
[r1, r2, r3, r4].forEach((r, i) => {
  const x = 780 + i * 160, vals = [[Math.max(0, r.invariance[r.K - 1]), PALETTE[1]], [Math.max(0, r.noisy[r.K - 1]), PALETTE[2]], [r.atp / r1.atp / 4, [150, 150, 150]]];
  vals.forEach(([v, col], j) => { const h = Math.min(1, v) * 280; cv.rect(x + j * 40, 700 - h, 32, h, col); });
  cv.text(x, 712, lab[i], [0, 0, 0], 1);
});
cv.rect(770, 700, 640, 1, [150, 150, 150]);
cv.save('figures/fig10-hierarchy.png');
fs.writeFileSync('figures/hierarchy-results.json', JSON.stringify({ structure: st, results: results.map(({ course, vTrace, ...r }) => r) }, null, 1));
log(`\n写入 figures/fig10-hierarchy.png（${el()}）`);
return { structure: st, results };
}

if (import.meta.url === `file:///${process.argv[1].replace(/\\/g, '/')}`) runHierarchy();
