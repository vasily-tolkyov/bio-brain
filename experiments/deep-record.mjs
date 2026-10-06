// 深层记录（路径积分）：上层加一群对浅层做慢积分的深层细胞，深层 → 下层中继按钙规则学（有向通道的落点）；可选 深层 → 本层浅层 的弱偏向
// 浅层保持默认（浅），负误差纠正打开。MODE=cal：标定（深层的时间进程）；默认：序列训练后测试。BIAS=1 开交接偏向；DEEP='{"sED":2}' 覆盖参数
import fs from 'fs';
import { PC2, OBJ, SEQ_ITEM, grp, sequenceCourse, timeCourse } from '../demo/predictive-hierarchy.mjs';
import { STIM, protoB } from '../demo/balanced-layer.mjs';
import { buildLayered, loadState, learnOnly, inputRate, spikesIn, saveState } from '../src/L4-network/layered-network.mjs';
import { mean } from '../demo/lib-hierarchy.mjs';
import { makeRng } from '../src/core/rng.mjs';

const BIAS = !!process.env.BIAS, DEEP = process.env.DEEP ?? '{}', log = console.log, t00 = Date.now();
const tag = `${BIAS ? 'bias' : 'nobias'}${DEEP === '{}' ? '' : '-' + DEEP.replace(/[^a-zA-Z0-9.]+/g, '')}`;
const S = buildLayered(2, { ...PC2, pcNeg: true, pcDeep: true, pcOver: { ...PC2.pcOver, deepBias: BIAS, ...JSON.parse(DEEP) } }), [L1, L2] = S.layers;
const f = (v) => v.toFixed(1).padStart(5), row = (a) => [...a].map(f).join('');
const rateOf = (ids, t0, t1) => mean(spikesIn(S.net, ids, t0, t1).map((s) => s.length)) / ((t1 - t0) / 1000);
const P = (tc, key, parts) => [...tc[`${key}${parts[0]}`]].map((v, i) => (v + tc[`${key}${parts[1]}`][i]) / 2);
const seg = (arr, a, b) => mean(arr.slice(a / 10, b / 10));
const at = (ids, hz) => new Map(ids.map((i) => [i, hz]));
const run = (ms, drives, learn) => S.net.run(ms, { rate: inputRate(S, drives), learn });
const X = ['A', 'D'], Y = ['C', 'B'];

loadState(S, JSON.parse(fs.readFileSync('cache/pc-2.json', 'utf8')).state);
loadState(S, JSON.parse(fs.readFileSync('cache/neg-G1.json', 'utf8'))); // 负误差的抑制（浅上层、AMPA + NMDA 回路，学过 60 个物体）
learnOnly(S, []);

/** 深层 → 下层中继 的平均效能：行 = 深层组（X 的 A、D / Y 的 C、B），列 = 中继组 */
function dsWeights() {
  const g = L2.DS, dsets = Object.fromEntries('ABCD'.split('').map((p) => [p, new Set(grp(S, 1, 'D', p))])), ssets = Object.fromEntries('ABCD'.split('').map((p) => [p, new Set(grp(S, 0, 'S', p))]));
  const between = (Ps, Qs) => { const v = []; for (let k = 0; k < g.pre.length; k++) if (Ps.some((p) => dsets[p].has(g.pre[k])) && Qs.some((q) => ssets[q].has(g.post[k]))) v.push(g.rho[k]); return mean(v); };
  return { xx: between(X, X), xy: between(X, Y), yx: between(Y, X), yy: between(Y, Y) };
}
function deWeights() {
  if (!L2.DE) return null;
  const g = L2.DE, dsets = Object.fromEntries('ABCD'.split('').map((p) => [p, new Set(grp(S, 1, 'D', p))])), esets = Object.fromEntries('ABCD'.split('').map((p) => [p, new Set(grp(S, 1, 'E', p))]));
  const between = (Ps, Qs) => { const v = []; for (let k = 0; k < g.pre.length; k++) if (Ps.some((p) => dsets[p].has(g.pre[k])) && Qs.some((q) => esets[q].has(g.post[k]))) v.push(g.rho[k]); return mean(v); };
  return { xx: between(X, X), xy: between(X, Y), yx: between(Y, X), yy: between(Y, Y) };
}

function tests(label) {
  run(500, new Map(), false);
  let t0 = S.net.t; run(2000, new Map(), false);
  const sp = { E2: rateOf(L2.E, t0, t0 + 2000), D2: rateOf(L2.D, t0, t0 + 2000), S1: rateOf(L1.S, t0, t0 + 2000), N1: rateOf(L1.N, t0, t0 + 2000), M1: rateOf(L1.M, t0, t0 + 2000) };
  const xy = sequenceCourse(S, [['AD', 150], ['CB', 150], ['', 200]], { reps: 12 }), yx = sequenceCourse(S, [['CB', 150], ['AD', 150], ['', 200]], { reps: 12 });
  const xo = sequenceCourse(S, [['AD', 150], ['', 350]], { reps: 12 }), fx = timeCourse(S, OBJ.X, { reps: 6 });
  const w = dsWeights(), we = deWeights();
  const r = {
    sp, w, we,
    dX: [seg(P(xo, 'D2', X), 100, 150), seg(P(xo, 'D2', X), 150, 200), seg(P(xo, 'D2', X), 200, 250), seg(P(xo, 'D2', X), 250, 300), seg(P(xo, 'D2', X), 300, 400)],
    eXoff: seg(P(xo, 'E2', X), 150, 200),
    dXinY: seg(P(xy, 'D2', X), 150, 200), dYinX: seg(P(xy, 'D2', Y), 100, 150), eYinX: seg(P(xy, 'E2', Y), 120, 150), mYinX: seg(P(xy, 'M1', Y), 50, 150),
    sYon: seg(P(xy, 'S1', Y), 140, 160), sYpre: seg(P(xy, 'S1', Y), 100, 150), eYon: seg(P(xy, 'E2', Y), 150, 170),
    fwd: seg(P(xy, 'N1', Y), 150, 200), fwdPeak: Math.max(...P(xy, 'N1', Y).slice(15, 25)), fwdAll: seg(P(xy, 'N1', Y), 150, 250),
    rev: seg(P(yx, 'N1', X), 150, 200), revPeak: Math.max(...P(yx, 'N1', X).slice(15, 25)), revAll: seg(P(yx, 'N1', X), 150, 250),
    omitE: seg(P(xo, 'E2', Y), 150, 250), omitM: seg(P(xo, 'M1', Y), 150, 250), omitN: seg(P(xo, 'N1', Y), 150, 200), omitS: seg(P(xo, 'S1', Y), 150, 250),
    famN: mean(P(fx, 'N1', X).slice(2)), famM: mean(P(fx, 'M1', X).slice(2)), famE: mean(P(fx, 'E2', X).slice(2)),
    tr: { D2X: P(xy, 'D2', X).slice(8, 27), D2Y: P(xy, 'D2', Y).slice(8, 27), E2X: P(xy, 'E2', X).slice(8, 27), E2Y: P(xy, 'E2', Y).slice(8, 27), S1Y: P(xy, 'S1', Y).slice(8, 27), N1Y: P(xy, 'N1', Y).slice(8, 27), M1Y: P(xy, 'M1', Y).slice(8, 27), N1Xrev: P(yx, 'N1', X).slice(8, 27) },
  };
  r.asym = r.rev - r.fwd;
  log(`  【${label}】深层 → 中继：X→X ${w.xx.toFixed(2)}，X→Y ${w.xy.toFixed(2)}，Y→X ${w.yx.toFixed(2)}，Y→Y ${w.yy.toFixed(2)}${we ? `；深层 → 浅层：X→Y ${we.xy.toFixed(2)}，Y→X ${we.yx.toFixed(2)}，同物体 ${((we.xx + we.yy) / 2).toFixed(2)}` : ''}`);
  log(`    自发：E2 ${f(sp.E2)} D2 ${f(sp.D2)} S1 ${f(sp.S1)} N1 ${f(sp.N1)} M1 ${f(sp.M1)} Hz`);
  log(`    X 150 ms 后撤掉：深层 X 在 X 的最后 50 ms / 撤掉后 0–50 / 50–100 / 100–150 / 150–250 ms：${row(r.dX)} Hz；浅层 X 撤掉后 0–50 ms ${f(r.eXoff)} Hz`);
  log(`    X → Y：Y 期间头 50 ms 深层 X ${f(r.dXinY)}；X 的后 50 ms 深层 Y ${f(r.dYinX)}、浅层 Y（最后 30 ms）${f(r.eYinX)}、Y 的负误差（50–150 ms）${f(r.mYinX)} Hz`);
  log(`    Y 的中继：X 的后 50 ms ${f(r.sYpre)}，Y 出现前后 ±10 ms ${f(r.sYon)} Hz；Y 的头 20 ms 浅层 Y ${f(r.eYon)} Hz`);
  log(`    后一项的正误差（头 50 ms / 峰值 / 头 100 ms）：X → Y ${f(r.fwd)}${f(r.fwdPeak)}${f(r.fwdAll)}，Y → X ${f(r.rev)}${f(r.revPeak)}${f(r.revAll)} Hz（反向 − 正向 ${r.asym.toFixed(1)}）`);
  log(`    X 之后 Y 不来（0–100 ms）：浅层 Y ${f(r.omitE)}，Y 的中继 ${f(r.omitS)}，Y 的负误差 ${f(r.omitM)}，Y 的正误差（头 50 ms）${f(r.omitN)} Hz`);
  log(`    熟悉的 X：正误差 ${f(r.famN)}、负误差 ${f(r.famM)}、浅层 X ${f(r.famE)} Hz`);
  log(`    Y 出现前后每 10 ms（-70 … +110）：深层 X ${row(r.tr.D2X)}`);
  log(`                                       深层 Y ${row(r.tr.D2Y)}`);
  log(`                                       浅层 X ${row(r.tr.E2X)}`);
  log(`                                       浅层 Y ${row(r.tr.E2Y)}`);
  log(`                                    Y 的中继 ${row(r.tr.S1Y)}`);
  log(`                                  Y 的正误差 ${row(r.tr.N1Y)}`);
  log(`                                  Y 的负误差 ${row(r.tr.M1Y)}`);
  log(`                          （反向）X 的正误差 ${row(r.tr.N1Xrev)}`);
  return r;
}

log(`深层记录：${BIAS ? '开' : '关'}交接偏向，参数 ${DEEP}`);
if (process.env.MODE === 'cal') { tests('标定（未学序列）'); process.exit(0); }
const before = tests('只学过物体');
const file = `cache/deep-${tag}.json`;
if (fs.existsSync(file)) loadState(S, JSON.parse(fs.readFileSync(file, 'utf8')));
else {
  const rng = makeRng(PC2.seed + 30), t0 = Date.now();
  const object = () => { const parts = OBJ[rng.pick(2) ? 'Y' : 'X']; run(200, at(parts.flatMap((p) => rng.shuffle(protoB(p).slice()).slice(0, 48)), STIM), true); run(300, new Map(), true); };
  learnOnly(S, [L2.DS, ...(L2.DE ? [L2.DE] : [])]);
  run(500, new Map(), false);
  for (let n = 0; n < 80; n++) {
    run(SEQ_ITEM, at(OBJ.X.flatMap(protoB), STIM), true);
    run(SEQ_ITEM, at(OBJ.Y.flatMap(protoB), STIM), true);
    run(300, new Map(), true);
    if (n % 2) object();
    if (n % 20 === 19) log(`  序列 ${n + 1}/80（${((Date.now() - t0) / 1000).toFixed(0)} s）`);
  }
  learnOnly(S, []);
  fs.writeFileSync(file, JSON.stringify(saveState(S)));
}
const after = tests('经历序列之后');
log(`  学到的期待（反向 − 正向，训练后 − 训练前）${(after.asym - before.asym).toFixed(1)} Hz；完成（${((Date.now() - t00) / 1000).toFixed(0)} s）`);
