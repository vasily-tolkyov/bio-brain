// 学"X 之后是 Y"并在上层交接：上层加深（upperGain）+ 负误差纠正 + 上层递归的资格痕迹（upperTauN）
// 环境变量：GAIN（默认 1.4）、TAUN（上层配对窗口 ms，默认 200；0 = 不加）、NEGCUT=1（向上的纠正切断，训练与测试都切断）
import fs from 'fs';
import { PC2, OBJ, SEQ_ITEM, grp, sequenceCourse, timeCourse } from '../demo/predictive-hierarchy.mjs';
import { STIM, protoB } from '../demo/balanced-layer.mjs';
import { buildLayered, loadState, learnOnly, inputRate, spikesIn, saveState } from '../src/L4-network/layered-network.mjs';
import { mean } from '../demo/lib-hierarchy.mjs';
import { makeRng } from '../src/core/rng.mjs';

const GAIN = Number(process.env.GAIN ?? 1.4), TAUN = Number(process.env.TAUN ?? 200), NEGCUT = !!process.env.NEGCUT, ADAPT = Number(process.env.ADAPT ?? 0), log = console.log, t00 = Date.now();
const tag = `g${GAIN}-t${TAUN}${NEGCUT ? '-cut' : ''}${ADAPT ? `-b${ADAPT}` : ''}`;
const S = buildLayered(2, { ...PC2, pcNeg: true, upperGain: GAIN, upperTauN: TAUN || undefined, upperAdapt: ADAPT ? { b: ADAPT } : undefined }), [L1, L2] = S.layers;
const f = (v) => v.toFixed(1).padStart(5), row = (a) => [...a].map(f).join('');
const rateOf = (ids, t0, t1) => mean(spikesIn(S.net, ids, t0, t1).map((s) => s.length)) / ((t1 - t0) / 1000);
const P = (tc, key, parts) => [...tc[`${key}${parts[0]}`]].map((v, i) => (v + tc[`${key}${parts[1]}`][i]) / 2);
const seg = (arr, a, b) => mean(arr.slice(a / 10, b / 10));
const at = (ids, hz) => new Map(ids.map((i) => [i, hz]));
const run = (ms, drives, learn) => S.net.run(ms, { rate: inputRate(S, drives), learn });

loadState(S, JSON.parse(fs.readFileSync('cache/pc-2.json', 'utf8')).state);
// 负误差的抑制：与 demo/negative-error.mjs 同一份缓存（同样的加深、同样学 60 个物体）
const negFile = `cache/neg-G${GAIN}${ADAPT ? `-b${ADAPT}` : ''}.json`;
if (fs.existsSync(negFile)) loadState(S, JSON.parse(fs.readFileSync(negFile, 'utf8')));
else {
  const rng = makeRng(11);
  learnOnly(S, [L1.NEGI, L2.NEGI]); run(500, new Map(), false);
  for (let n = 0; n < 60; n++) { const parts = OBJ[rng.pick(2) ? 'Y' : 'X']; run(200, at(parts.flatMap((p) => rng.shuffle(protoB(p).slice()).slice(0, 48)), STIM), true); run(300, new Map(), true); }
  learnOnly(S, []); fs.writeFileSync(negFile, JSON.stringify(saveState(S)));
  log(`  负误差的抑制学完（${((Date.now() - t00) / 1000).toFixed(0)} s）`);
}
learnOnly(S, []);
if (NEGCUT) L2.JE.scale.fill(0);

function weights() {
  const g = L2.EE, sets = Object.fromEntries('ABCD'.split('').map((p) => [p, new Set(grp(S, 1, 'E', p))]));
  const between = (Ps, Qs) => { const v = []; for (let k = 0; k < g.pre.length; k++) if (Ps.some((p) => sets[p].has(g.pre[k])) && Qs.some((q) => sets[q].has(g.post[k]))) v.push(g.rho[k]); return mean(v); };
  return { within: (between(['A'], ['D']) + between(['D'], ['A']) + between(['C'], ['B']) + between(['B'], ['C'])) / 4, xy: between(['A', 'D'], ['C', 'B']), yx: between(['C', 'B'], ['A', 'D']) };
}

function tests(label) {
  run(500, new Map(), false);
  let t0 = S.net.t; run(2000, new Map(), false);
  const spE2 = rateOf(L2.E, t0, t0 + 2000), spN1 = rateOf(L1.N, t0, t0 + 2000), spM1 = rateOf(L1.M, t0, t0 + 2000);
  t0 = S.net.t; const ms = 3000; run(ms, new Map(), false);
  const groups = 'ABCD'.split('').map((p) => grp(S, 1, 'E', p));
  let hot = 0; for (let b = 0; b < ms / 20; b++) if (groups.some((ids) => rateOf(ids, t0 + b * 20, t0 + (b + 1) * 20) > 10)) hot++;
  const xy = sequenceCourse(S, [['AD', 150], ['CB', 150], ['', 200]], { reps: 12 }), yx = sequenceCourse(S, [['CB', 150], ['AD', 150], ['', 200]], { reps: 12 });
  const xo = sequenceCourse(S, [['AD', 150], ['', 350]], { reps: 12 }), fx = timeCourse(S, OBJ.X, { reps: 6 });
  const Y = ['C', 'B'], X = ['A', 'D'];
  const r = {
    w: weights(), spE2, spN1, spM1, hot: (100 * hot) / (ms / 20),
    preY: seg(P(xy, 'E2', Y), 120, 150), onY: seg(P(xy, 'E2', Y), 150, 170), relayY: seg(P(xy, 'S1', Y), 150, 160),
    fwd: seg(P(xy, 'N1', Y), 150, 200), fwdLate: seg(P(xy, 'N1', Y), 180, 230), fwdPeak: Math.max(...P(xy, 'N1', Y).slice(15, 25)),
    rev: seg(P(yx, 'N1', X), 150, 200), revPeak: Math.max(...P(yx, 'N1', X).slice(15, 25)),
    omitE: [seg(P(xo, 'E2', Y), 150, 200), seg(P(xo, 'E2', Y), 200, 300)], omitM: [seg(P(xo, 'M1', Y), 150, 200), seg(P(xo, 'M1', Y), 200, 300)], omitN: seg(P(xo, 'N1', Y), 150, 200),
    xOff: seg(P(xo, 'E2', X), 150, 200),
    famN: mean(P(fx, 'N1', X).slice(2)), famM: mean(P(fx, 'M1', X).slice(2)), famE: mean(P(fx, 'E2', X).slice(2)),
    tr: { E2Y: P(xy, 'E2', Y).slice(12, 27), E2X: P(xy, 'E2', X).slice(12, 27), S1Y: P(xy, 'S1', Y).slice(12, 27), N1Y: P(xy, 'N1', Y).slice(12, 27), N1Xrev: P(yx, 'N1', X).slice(12, 27) },
  };
  r.asym = r.rev - r.fwd;
  log(`  【${label}】上层 物体内 ${r.w.within.toFixed(2)}，X→Y ${r.w.xy.toFixed(2)}，Y→X ${r.w.yx.toFixed(2)}；自发 E2 ${f(spE2)} N1 ${f(spN1)} M1 ${f(spM1)} Hz，静息时自己亮起来 ${r.hot.toFixed(1)}%`);
  log(`    X 的最后 30 ms 上层 Y ${f(r.preY)}，Y 的头 20 ms 上层 Y ${f(r.onY)}，Y 的头 10 ms Y 的中继 ${f(r.relayY)} Hz`);
  log(`    后一项的正误差（头 50 ms / 30–80 ms / 峰值）：X → Y ${f(r.fwd)}${f(r.fwdLate)}${f(r.fwdPeak)}，Y → X ${f(r.rev)}    ${f(r.revPeak)} Hz（反向 − 正向 ${r.asym.toFixed(1)}）`);
  log(`    X 之后 Y 不来：上层 Y 0–50 / 50–150 ms ${f(r.omitE[0])}${f(r.omitE[1])}，Y 的负误差 ${f(r.omitM[0])}${f(r.omitM[1])}，Y 的正误差（头 50 ms）${f(r.omitN)} Hz；X 撤掉后 0–50 ms 上层 X ${f(r.xOff)} Hz`);
  log(`    熟悉的 X：正误差 ${f(r.famN)}、负误差 ${f(r.famM)}、上层 X ${f(r.famE)} Hz`);
  log(`    Y 出现前后每 10 ms（-30 … +110）：上层 Y ${row(r.tr.E2Y)}`);
  log(`                                       上层 X ${row(r.tr.E2X)}`);
  log(`                                    Y 的中继 ${row(r.tr.S1Y)}`);
  log(`                                    Y 的误差 ${row(r.tr.N1Y)}`);
  log(`                          （反向）X 的误差 ${row(r.tr.N1Xrev)}`);
  return r;
}

log(`上层加深 ${GAIN} 倍，上层配对窗口 ${TAUN || 40} ms，${NEGCUT ? '切断' : '保留'}负误差向上的纠正，上层适应 b ${ADAPT || 20} pA`);
const before = tests('只学过物体');
const file = `cache/ho-${tag}.json`;
if (fs.existsSync(file)) loadState(S, JSON.parse(fs.readFileSync(file, 'utf8')));
else {
  const rng = makeRng(PC2.seed + 30), t0 = Date.now();
  const object = () => { const parts = OBJ[rng.pick(2) ? 'Y' : 'X']; run(200, at(parts.flatMap((p) => rng.shuffle(protoB(p).slice()).slice(0, 48)), STIM), true); run(300, new Map(), true); };
  learnOnly(S, [L2.EE, L1.PRED, L1.NEGI]);
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
