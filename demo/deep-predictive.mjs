/** 三部分结构（浅层结构网络 / 深层预测网络 / 误差网络）的两层最小实验：只给上层加深层预测网络。
 * 深层 = "往前一步"的映射：活动表示"接下来"。学习全在输入侧（src/L4-network/layered-network.mjs 的 pcDeep）：
 *   老师：浅层 → 深层 同组固定（ED，Y 到来时直接驱动深层 Y）；
 *   线索：浅层 → 深层 只连别组、可塑带突触痕迹（CUE）——X 期间浅层 X 留痕，Y 到来时老师驱动深层 Y，
 *   痕迹 × 突触后 → 学成 浅层 X → 深层 Y；Y 之后是休息，反向没有配对，方向来自时序本身。
 * 输出侧不学：深层 → 下层中继 同组固定（DS，复用中继已学好的物体级抵消）；
 *   可选 深层 → 本层浅层 弱交接偏向（BIAS=1，DE；它的错误由负误差纠正）。
 * 从只学过物体的网络出发（cache/pc-2.json + cache/neg-G1.json 的负误差抑制），序列阶段只学 CUE（+DE）：
 *   浅层递归、中继 → 误差（PRED）、负误差抑制（NEGI）全部冻结，隔离深层的贡献。
 * 用法：node demo/deep-predictive.mjs（训练 + 全套检验；结果缓存 cache/deep2-<tag>.json）
 *       BIAS=1 … 开交接偏向；DEEP='{"cueTauN":40}' 覆盖深层参数；NSEQ=80 序列遍数
 *       MODE=cal   标定：各群自发、老师对深层的驱动与余留、DS 强度
 *       MODE=trace 逐遍记账：每遍 X→Y 后线索各方向的 Δρ（检验方向信号与配对窗口） */
import fs from 'fs';
import { PC2, OBJ, SEQ_ITEM, grp, sequenceCourse, timeCourse } from './predictive-hierarchy.mjs';
import { STIM, protoB } from './balanced-layer.mjs';
import { buildLayered, loadState, learnOnly, inputRate, spikesIn, saveState } from '../src/L4-network/layered-network.mjs';
import { mean } from './lib-hierarchy.mjs';
import { makeRng } from '../src/core/rng.mjs';

const BIAS = !!process.env.BIAS, DEEP = process.env.DEEP ?? '{}', log = console.log, t00 = Date.now();
const tag = `${BIAS ? 'bias' : 'nobias'}${DEEP === '{}' ? '' : '-' + DEEP.replace(/[^a-zA-Z0-9.]+/g, '')}`;
const S = buildLayered(2, { ...PC2, pcNeg: true, pcDeep: true, pcOver: { ...PC2.pcOver, deepBias: BIAS, ...JSON.parse(DEEP) } }), [L1, L2] = S.layers;
if (!L2.CUE) throw new Error('pcDeep 未生效：第二层没有线索连接 CUE');
const f = (v) => v.toFixed(1).padStart(5), row = (a) => [...a].map(f).join('');
const f3 = (v) => (v === null || Number.isNaN(v) ? '—' : v.toFixed(3));
const rateOf = (ids, t0, t1) => mean(spikesIn(S.net, ids, t0, t1).map((s) => s.length)) / ((t1 - t0) / 1000);
const P = (tc, key, parts) => [...tc[`${key}${parts[0]}`]].map((v, i) => (v + tc[`${key}${parts[1]}`][i]) / 2);
const seg = (arr, a, b) => mean(arr.slice(a / 10, b / 10));
const at = (ids, hz) => new Map(ids.map((i) => [i, hz]));
const run = (ms, drives, learn) => S.net.run(ms, { rate: inputRate(S, drives), learn });
const X = ['A', 'D'], Y = ['C', 'B'];

loadState(S, JSON.parse(fs.readFileSync('cache/pc-2.json', 'utf8')).state);
loadState(S, JSON.parse(fs.readFileSync('cache/neg-G1.json', 'utf8')));
learnOnly(S, []);

const esets = Object.fromEntries('ABCD'.split('').map((p) => [p, new Set(grp(S, 1, 'E', p))]));
const dsets = Object.fromEntries('ABCD'.split('').map((p) => [p, new Set(grp(S, 1, 'D', p))]));
const objOf = (sets) => (id) => (X.some((p) => sets[p].has(id)) ? 'X' : Y.some((p) => sets[p].has(id)) ? 'Y' : '-');
const eObj = objOf(esets), dObj = objOf(dsets);

/** 线索按方向分桶（只连别组，同组没有突触）：X→Y 要学的；Y→X 反向不该有的；X>X / Y>Y 同物体交叉（同时出现，双向难免）；other 含不成组。
 *  cueBuckets(null) 返回各桶的平均效能；cueBuckets(rho0, rho1) 返回各桶的平均 Δρ */
function cueBuckets(rhoA, rhoB = null) {
  const g = L2.CUE, acc = {};
  for (let k = 0; k < g.pre.length; k++) {
    const key = `${eObj(g.pre[k])}>${dObj(g.post[k])}`;
    (acc[key] ??= []).push(rhoB === null ? g.rho[k] : rhoB[k] - rhoA[k]);
  }
  const m = (key) => (acc[key]?.length ? mean(acc[key]) : null);
  const others = Object.entries(acc).filter(([k]) => k.includes('-')).flatMap(([, v]) => v);
  return { xy: m('X>Y'), yx: m('Y>X'), wx: m('X>X'), wy: m('Y>Y'), other: others.length ? mean(others) : null };
}

function deWeights() {
  if (!L2.DE) return null;
  const g = L2.DE;
  const between = (Ps, Qs) => { const v = []; for (let k = 0; k < g.pre.length; k++) if (Ps.some((p) => dsets[p].has(g.pre[k])) && Qs.some((q) => esets[q].has(g.post[k]))) v.push(g.rho[k]); return v.length ? mean(v) : null; };
  return { xy: between(X, Y), yx: between(Y, X), same: (between(X, X) + between(Y, Y)) / 2 };
}

function tests(label) {
  run(500, new Map(), false);
  let t0 = S.net.t; run(2000, new Map(), false);
  const sp = { E2: rateOf(L2.E, t0, t0 + 2000), D2: rateOf(L2.D, t0, t0 + 2000), S1: rateOf(L1.S, t0, t0 + 2000), N1: rateOf(L1.N, t0, t0 + 2000), M1: rateOf(L1.M, t0, t0 + 2000) };
  const xy = sequenceCourse(S, [['AD', 150], ['CB', 150], ['', 200]], { reps: 12 }), yx = sequenceCourse(S, [['CB', 150], ['AD', 150], ['', 200]], { reps: 12 });
  const xo = sequenceCourse(S, [['AD', 150], ['', 350]], { reps: 12 }), fx = timeCourse(S, OBJ.X, { reps: 6 });
  const w = cueBuckets(null), we = deWeights();
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
  log(`  【${label}】线索：X→Y ${f3(w.xy)}，Y→X ${f3(w.yx)}，同物体 X ${f3(w.wx)}、Y ${f3(w.wy)}，其他 ${f3(w.other)}${we ? `；深层 → 浅层：X→Y ${f3(we.xy)}，Y→X ${f3(we.yx)}，同物体 ${f3(we.same)}` : ''}`);
  log(`    自发：E2 ${f(sp.E2)} D2 ${f(sp.D2)} S1 ${f(sp.S1)} N1 ${f(sp.N1)} M1 ${f(sp.M1)} Hz`);
  log(`    X 150 ms 后撤掉：深层 X 在 X 的最后 50 ms / 撤掉后 0–50 / 50–100 / 100–150 / 150–250 ms：${row(r.dX)} Hz；浅层 X 撤掉后 0–50 ms ${f(r.eXoff)} Hz`);
  log(`    X → Y：X 的后 50 ms 深层 Y ${f(r.dYinX)}、浅层 Y（最后 30 ms）${f(r.eYinX)}、Y 的负误差（50–150 ms）${f(r.mYinX)} Hz；Y 期间头 50 ms 深层 X ${f(r.dXinY)}`);
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

log(`三部分结构最小实验：深层预测网络（线索只连别组），${BIAS ? '开' : '关'}交接偏向，参数 ${DEEP}`);

if (process.env.MODE === 'cal') {
  run(500, new Map(), false);
  let t0 = S.net.t; run(2000, new Map(), false);
  log(`  自发：E2 ${f(rateOf(L2.E, t0, t0 + 2000))} D2 ${f(rateOf(L2.D, t0, t0 + 2000))} S1 ${f(rateOf(L1.S, t0, t0 + 2000))} N1 ${f(rateOf(L1.N, t0, t0 + 2000))} M1 ${f(rateOf(L1.M, t0, t0 + 2000))} Hz`);
  const xo = sequenceCourse(S, [['AD', 150], ['', 350]], { reps: 8 });
  log('  呈现 X 150 ms 然后撤掉（10 ms 一箱，0 = X 开始）：');
  log(`    深层 X ${row(P(xo, 'D2', X).slice(0, 30))}`);
  log(`    浅层 X ${row(P(xo, 'E2', X).slice(0, 30))}`);
  log(`    中继 X ${row(P(xo, 'S1', X).slice(0, 30))}`);
  log(`    深层 Y ${row(P(xo, 'D2', Y).slice(0, 30))}`);
  log(`    中继 Y ${row(P(xo, 'S1', Y).slice(0, 30))}`);
  const dXids = X.flatMap((p) => grp(S, 1, 'D', p)), sXids = X.flatMap((p) => grp(S, 0, 'S', p));
  run(300, new Map(), false);
  t0 = S.net.t;
  S.net.run(300, { rate: inputRate(S, new Map(), new Map(dXids.map((i) => [i, 15]))), learn: false });
  log(`  DS 强度检验：深层 X 被直接驱动（输入 15 Hz）→ 深层 X 实放 ${f(rateOf(dXids, t0, t0 + 300))} Hz，中继 X ${f(rateOf(sXids, t0 + 100, t0 + 300))} Hz`);
  process.exit(0);
}

if (process.env.MODE === 'trace') {
  learnOnly(S, [L2.CUE]);
  const NP = Number(process.env.NPASS ?? 3);
  const dYids = Y.flatMap((p) => grp(S, 1, 'D', p)), dXids = X.flatMap((p) => grp(S, 1, 'D', p));
  const sYids = Y.flatMap((p) => grp(S, 0, 'S', p));
  for (let pass = 1; pass <= NP; pass++) {
    const rho0 = L2.CUE.rho.slice(), t0 = S.net.t;
    run(SEQ_ITEM, at(OBJ.X.flatMap(protoB), STIM), true);
    run(SEQ_ITEM, at(OBJ.Y.flatMap(protoB), STIM), true);
    run(300, new Map(), true);
    const d = cueBuckets(rho0, L2.CUE.rho);
    const e3 = (v) => (v === null || Number.isNaN(v) ? '—' : (v * 1e3).toFixed(1));
    log(`  第 ${pass} 遍 Δρ×10³：X→Y ${e3(d.xy)}，Y→X ${e3(d.yx)}，同物体 X ${e3(d.wx)}、Y ${e3(d.wy)}，其他 ${e3(d.other)}；`
      + `深层 Y 在 X 的后 50 ms ${f(rateOf(dYids, t0 + 100, t0 + 150))}，深层 X 在 Y 的头 50 ms ${f(rateOf(dXids, t0 + 150, t0 + 200))} / 后 50 ms ${f(rateOf(dXids, t0 + 250, t0 + 300))}，Y 的中继在 X 的后 50 ms ${f(rateOf(sYids, t0 + 100, t0 + 150))} Hz`);
  }
  process.exit(0);
}

const before = tests('只学过物体');
const file = `cache/deep2-${tag}.json`;
if (fs.existsSync(file)) loadState(S, JSON.parse(fs.readFileSync(file, 'utf8')));
else {
  const rng = makeRng(PC2.seed + 30), t0 = Date.now(), NSEQ = Number(process.env.NSEQ ?? 80);
  const object = () => { const parts = OBJ[rng.pick(2) ? 'Y' : 'X']; run(200, at(parts.flatMap((p) => rng.shuffle(protoB(p).slice()).slice(0, 48)), STIM), true); run(300, new Map(), true); };
  learnOnly(S, [L2.CUE, ...(L2.DE ? [L2.DE] : [])]);
  run(500, new Map(), false);
  for (let n = 0; n < NSEQ; n++) {
    run(SEQ_ITEM, at(OBJ.X.flatMap(protoB), STIM), true);
    run(SEQ_ITEM, at(OBJ.Y.flatMap(protoB), STIM), true);
    run(300, new Map(), true);
    if (n % 2) object();
    if (n % 20 === 19) log(`  序列 ${n + 1}/${NSEQ}（${((Date.now() - t0) / 1000).toFixed(0)} s）`);
  }
  learnOnly(S, []);
  fs.writeFileSync(file, JSON.stringify(saveState(S)));
}
const after = tests('经历序列之后');
log(`  学到的期待（反向 − 正向，训练后 − 训练前）${(after.asym - before.asym).toFixed(1)} Hz；完成（${((Date.now() - t00) / 1000).toFixed(0)} s）`);
