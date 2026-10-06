/** 预测编码的两层：每层 = 表征群（平衡态，吸引子 + 有向通道）+ 误差群 + 预测中继群（见 src/L4-network/layered-network.mjs 的 pc）。
 * 第一层用训练好的平衡层（demo/balanced-layer.mjs，兴奋性连接冻结）；第二层学物体（X = A + D、Y = C + B），
 * 同时第一层的中继 → 误差抑制学习第二层的预测；再经历序列 X → Y，学"X 之后是 Y"。
 * 用法：CALIB=1 node demo/predictive-hierarchy.mjs（未训练的两层标定）；
 *       node demo/predictive-hierarchy.mjs（训练 + 全部检验 → figures/fig12-predictive.png；训练结果缓存到 cache/pc-2.json、pc-2-seq.json）；
 *       FIG=1 node demo/predictive-hierarchy.mjs（只按 figures/predictive-results.json 重画图）；
 *       PCO='{"tdKind":"exc","sTD":1,"predTauPre":0}' 之类覆盖预测编码的参数（这一组是加痕迹与慢中继之前的版本；缓存与图的文件名带上参数） */
import fs from 'fs';
import { BAL, STIM, protoB, trainBal } from './balanced-layer.mjs';
import { mean } from './lib-hierarchy.mjs';
import { buildLayered, learnOnly, inputRate, spikesIn, saveState, loadState, groupIds } from '../src/L4-network/layered-network.mjs';
import { makeRng } from '../src/core/rng.mjs';
import { Canvas, PALETTE } from '../src/core/png.mjs';

export const OBJ = { X: ['A', 'D'], Y: ['C', 'B'] };
// 标定（第一层训练好、第二层与预测未学）：第一层自发 E 3.0、误差 4.7、中继 8.3 Hz；呈现 X 时第一层 A 组的中继 27–33 Hz，
//   第二层 A、D 组 11–22 Hz、B 组 0–0.8 Hz。层间前馈与上层背景沿用平衡态两层的标定（上层自发 2.6 Hz，对第一层的输入分级响应）
export const PC2 = {
  ...BAL, pc: true, ffSame: 0.1, ffOther: 0.005, ffScale: 6, ffiP: 0.1, ffiScale: 1.5, bgUpper: Number(process.env.BG_UP ?? 18), upperBudget: 0.3,
  pcOver: JSON.parse(process.env.PCO ?? '{}'),
};
const TAG = process.env.PCO ? '-' + process.env.PCO.replace(/[^a-zA-Z0-9.]+/g, '') : '';
const log = console.log, G = 'ABCD', POPS = ['E', 'N', 'S', 'M', 'V', 'J', 'D'];
const at = (ids, hz) => new Map(ids.map((i) => [i, hz]));
const rate = (S, ids, t0, t1) => mean(spikesIn(S.net, ids, t0, t1).map((s) => s.length)) / ((t1 - t0) / 1000);
/** 第 k 层（从 0 起）某个群（E、N、S）里字母 p 的组 */
export const grp = (S, k, pop, p) => (pop === 'E' && k === 0 ? protoB(p).map((a) => S.layers[0].E[a]) : groupIds(S, S.layers[k][pop], G.indexOf(p)));
const run = (S, ms, drives, learn = false) => { const t0 = S.net.t; S.net.run(ms, { rate: inputRate(S, drives), learn }); return t0; };

/** 各层各群的自发频率 */
export function spontaneous(S, ms = 1500) {
  run(S, 500, new Map());
  const t0 = run(S, ms, new Map());
  return S.layers.map((L) => Object.fromEntries(['E', 'I', 'N', 'S'].map((p) => [p, rate(S, L[p], t0, t0 + ms)])));
}
/** 呈现若干部分（字母）ms 毫秒，按 bin 毫秒记录各层各群各组的频率；reps 次平均 */
export function timeCourse(S, parts, { ms = 200, bin = 25, reps = 6, after = 300, n = 48 } = {}) {
  const rng = makeRng(99), nb = ms / bin, out = {};
  for (let r = 0; r < reps; r++) {
    const drive = at(parts.flatMap((p) => rng.shuffle(protoB(p).slice()).slice(0, n)), STIM);
    const t0 = run(S, ms, drive);
    S.layers.forEach((L, k) => POPS.filter((pop) => L[pop]).forEach((pop) => G.split('').forEach((p) => {
      const key = `${pop}${k + 1}${p}`, ids = grp(S, k, pop, p);
      out[key] ??= new Float64Array(nb);
      for (let b = 0; b < nb; b++) out[key][b] += rate(S, ids, t0 + b * bin, t0 + (b + 1) * bin) / reps;
    })));
    run(S, after, new Map());
  }
  return out;
}
const row = (arr) => [...arr].map((v) => v.toFixed(1).padStart(5)).join('');
const errOf = (tc, parts, from = 2) => mean(parts.map((p) => mean([...tc[`N1${p}`]].slice(from))));

/** 训练第二层：物体 X、Y 随机交替（每个部分随机 48 个成员），第二层的递归与抑制、两层的预测抑制同时学；
 *  每 25 个物体不学习地测一次 X 引起的第一层误差（50–200 ms，A、D 组平均），得到学习曲线 */
export function trainPC({ seq = false, log: say = log } = {}) {
  const file = `cache/pc-2${seq ? '-seq' : ''}${TAG}.json`;
  if (fs.existsSync(file)) return JSON.parse(fs.readFileSync(file, 'utf8'));
  const S = buildLayered(2, PC2), [L1, L2] = S.layers, rng = makeRng(PC2.seed + (seq ? 30 : 20)), t0 = Date.now(), curve = [];
  const learners = [L2.EE, L2.IE, L1.PRED, L2.PRED];
  const object = () => { const parts = OBJ[rng.pick(2) ? 'Y' : 'X']; run(S, 200, at(parts.flatMap((p) => rng.shuffle(protoB(p).slice()).slice(0, 48)), STIM), true); run(S, 300, new Map(), true); };
  if (seq) {
    // 序列：在只学过物体的网络上继续，X 150 ms → Y 150 ms → 休息 300 ms，与物体交错（只学序列会把物体本身冲掉）
    const base = trainPC({ log: say });
    loadState(S, base.state); learnOnly(S, learners);
    run(S, 500, new Map());
    for (let n = 0; n < 80; n++) {
      run(S, SEQ_ITEM, at(OBJ.X.flatMap(protoB), STIM), true);
      run(S, SEQ_ITEM, at(OBJ.Y.flatMap(protoB), STIM), true);
      run(S, 300, new Map(), true);
      if (n % 2) object();
      if (n % 20 === 19) say(`  第 2 层：序列 ${n + 1}/80（${((Date.now() - t0) / 1000).toFixed(0)} s）`);
    }
    curve.push(...base.curve);
  } else {
    loadState(S, trainBal());
    const probe = () => { learnOnly(S, []); const tc = timeCourse(S, OBJ.X, { reps: 3 }); learnOnly(S, learners); return errOf(tc, OBJ.X); };
    curve.push(probe());
    run(S, 500, new Map());
    for (let n = 0; n < 100; n++) {
      object();
      if (n % 25 === 24) { curve.push(probe()); say(`  第 2 层：物体 ${n + 1}/100（${((Date.now() - t0) / 1000).toFixed(0)} s），X 的误差 ${curve.at(-1).toFixed(1)} Hz`); }
    }
  }
  const out = { state: saveState(S), curve };
  fs.mkdirSync('cache', { recursive: true });
  fs.writeFileSync(file, JSON.stringify(out));
  return out;
}
export const SEQ_ITEM = 150;
export function pcStack({ trained = true, seq = false } = {}) {
  const S = buildLayered(2, PC2);
  loadState(S, trained ? trainPC({ seq }).state : trainBal());
  learnOnly(S, []);
  return S;
}
/** 依次呈现若干段（每段：部分字母、毫秒；空字母 = 空白），按 bin 记录第 k 层某个群各组的频率，reps 次平均 */
export function sequenceCourse(S, segments, { bin = 10, reps = 8, after = 400 } = {}) {
  const ms = segments.reduce((s, [, d]) => s + d, 0), nb = ms / bin, out = {};
  for (let r = 0; r < reps; r++) {
    const t0 = S.net.t;
    for (const [parts, d] of segments) run(S, d, at([...parts].flatMap(protoB), STIM));
    S.layers.forEach((L, k) => POPS.filter((pop) => L[pop]).forEach((pop) => G.split('').forEach((p) => {
      const key = `${pop}${k + 1}${p}`, ids = grp(S, k, pop, p);
      out[key] ??= new Float64Array(nb);
      for (let b = 0; b < nb; b++) out[key][b] += rate(S, ids, t0 + b * bin, t0 + (b + 1) * bin) / reps;
    })));
    run(S, after, new Map());
  }
  return out;
}
const avg = (tc, keys, b0, b1) => mean(keys.map((key) => mean([...tc[key]].slice(b0, b1))));
/** 暂时把某个连接组的强度乘以 f（切断 = 0），跑完 fn 再恢复 */
function scaled(group, f, fn) { const keep = group.scale.slice(); group.scale.forEach((v, k) => (group.scale[k] = v * f)); try { return fn(); } finally { group.scale.set(keep); } }

const pair = (tc, p, q) => [...tc[p]].map((v, b) => (v + tc[q][b]) / 2);

/** 物体层面的检验（只学过物体的网络）：解释掉证据、两种切断、缺一半、没见过的组合、误差的开销、误差上行的作用 */
export function objectTests(S, say = log) {
  const [L1, L2] = S.layers, sp = spontaneous(S);
  sp.forEach((s, k) => say(`  第${k + 1}层自发：E ${s.E.toFixed(2)}  I ${s.I.toFixed(2)}  N ${s.N.toFixed(2)}  S ${s.S.toFixed(2)} Hz`));
  const show = (title, tc) => { say(`  ${title}（每 25 ms）：`); for (const key of ['N1A', 'N1D', 'N1B', 'N1C', 'S1A', 'S1B', 'E2A', 'E2D', 'E2B']) say(`    ${key} ${row(tc[key])}`); };
  const famX = timeCourse(S, OBJ.X); show('熟悉的物体 X', famX);
  const cutP = scaled(L1.PRED, 0, () => timeCourse(S, OBJ.X)); show('切断预测抑制（中继 → 误差）', cutP);
  const cutT = scaled(L2.TD, 0, () => timeCourse(S, OBJ.X)); show('切断预测下行（上层 → 中继）', cutT);
  const onlyA = timeCourse(S, ['A']); show('只有 A（D 缺失）', onlyA);
  const novel = timeCourse(S, ['A', 'C']); show('没见过的组合 A + C', novel);
  const sum = (tc) => G.split('').reduce((s, p) => s + mean([...tc[`N1${p}`]]), 0);
  const err = {
    base: sp[0].N, onset: mean(pair(famX, 'N1A', 'N1D').slice(0, 1)), fam: errOf(famX, OBJ.X), cutP: errOf(cutP, OBJ.X), cutT: errOf(cutT, OBJ.X),
    onlyA: errOf(onlyA, ['A']), missD: errOf(onlyA, ['D']), novelAC: errOf(novel, ['A', 'C']), novelBD: errOf(novel, ['B', 'D']),
  };
  const total = { fam: sum(famX), cutP: sum(cutP), onlyA: sum(onlyA), novel: sum(novel) };
  say(`  第一层误差（50–200 ms，自发 ${err.base.toFixed(1)} Hz）：熟悉 X 的 A、D 头 25 ms ${err.onset.toFixed(1)}、之后 ${err.fam.toFixed(1)}；切断预测 ${err.cutP.toFixed(1)}；切断下行 ${err.cutT.toFixed(1)}`);
  say(`    只有 A：A ${err.onlyA.toFixed(1)}、缺失的 D ${err.missD.toFixed(1)}；A + C：A、C ${err.novelAC.toFixed(1)}，B、D ${err.novelBD.toFixed(1)}`);
  say(`  第一层误差群四组合计（0–200 ms 平均，Hz）：熟悉 X ${total.fam.toFixed(1)}，切断预测 ${total.cutP.toFixed(1)}，只有 A ${total.onlyA.toFixed(1)}，A + C ${total.novel.toFixed(1)}`);
  // 误差上行的作用：输入从 X 直接换成 Y，换后上层 Y 的组多强、下层 Y 的误差多快消退；切断误差上行作对照
  const sw = (tc) => { const X = pair(tc, 'E2A', 'E2D'), Y = pair(tc, 'E2C', 'E2B'); return { x: mean(X.slice(25, 40)), y: mean(Y.slice(25, 40)), n: avg(tc, ['N1C', 'N1B'], 20, 25) }; };
  const swOn = sw(sequenceCourse(S, [['AD', 200], ['CB', 200]])), swOff = scaled(L2.UP, 0, () => sw(sequenceCourse(S, [['AD', 200], ['CB', 200]])));
  say(`  X 直接换成 Y：换后 50–200 ms 上层 Y / X 的组 ${swOn.y.toFixed(1)} / ${swOn.x.toFixed(1)} Hz（切断误差上行 ${swOff.y.toFixed(1)} / ${swOff.x.toFixed(1)}）；换后头 50 ms 第一层 Y 的误差 ${swOn.n.toFixed(1)} Hz（切断 ${swOff.n.toFixed(1)}）`);
  const keep = (tc) => Object.fromEntries(['N1A', 'N1D', 'N1B', 'N1C', 'E2A', 'E2D', 'E2B', 'E2C'].map((key) => [key, [...tc[key]]]));
  return { sp, err, total, sw: { on: swOn, off: swOff }, tc: { famX: keep(famX), cutP: keep(cutP), cutT: keep(cutT), onlyA: keep(onlyA), novel: keep(novel) } };
}

/** 第一层预测抑制里，从 from 各部分的中继组到 to 各部分的误差组的突触编号 */
const predPairs = (S, from, to) => {
  const g = S.layers[0].PRED, a = new Set(from.flatMap((p) => grp(S, 0, 'S', p))), b = new Set(to.flatMap((p) => grp(S, 0, 'N', p))), ks = [];
  for (let k = 0; k < g.pre.length; k++) if (a.has(g.pre[k]) && b.has(g.post[k])) ks.push(k);
  return ks;
};

/** 序列期待。和实验里的做法一样，比较学过的先后（X → Y）与没学过的反向（Y → X）：后一项出现后头 50 ms 它的误差。
 *  两个物体本身并不完全对称，所以期待 = 经历序列的网络里（反向 − 正向）减去只学物体的网络里的同一个差（见 runPredictive）。
 *  另把「X 的中继 → Y 的误差细胞」这组突触置零看它们的贡献——这组突触也承担 X 的中继对 Y 的误差细胞的基线抑制，
 *  只学物体的网络里置零同样有效果，所以也要与只学物体的网络相减。另记：X 之后 Y 没来时 Y 的误差，X 末尾上层 Y 的组 */
export function sequenceTests(S, name, say = log) {
  const [L1] = S.layers, base = spontaneous(S)[0].N, g = L1.PRED, kXY = predPairs(S, OBJ.X, OBJ.Y);
  const cut = (fn) => { const keep = g.scale.slice(); for (const k of kXY) g.scale[k] = 0; try { return fn(); } finally { g.scale.set(keep); } };
  const XY = [['AD', 150], ['CB', 150], ['', 200]], YX = [['CB', 150], ['AD', 150], ['', 200]], XO = [['AD', 150], ['', 350]];
  const xy = sequenceCourse(S, XY, { reps: 16 }), yx = sequenceCourse(S, YX, { reps: 16 }), xyCut = cut(() => sequenceCourse(S, XY, { reps: 16 }));
  const xo = sequenceCourse(S, XO), by = sequenceCourse(S, [['', 150], ['CB', 150], ['', 200]]);
  const Yn = ['N1C', 'N1B'], Xn = ['N1A', 'N1D'], Ye = ['E2C', 'E2B'], peak = (tc, p, q) => Math.max(...pair(tc, p, q).slice(15, 25));
  const r = {
    base, w: mean(kXY.map((k) => g.scale[k])), wYX: mean(predPairs(S, OBJ.Y, OBJ.X).map((k) => g.scale[k])), wXX: mean(predPairs(S, OBJ.X, OBJ.X).map((k) => g.scale[k])),
    fwd: avg(xy, Yn, 15, 20), rev: avg(yx, Xn, 15, 20), fwdPeak: peak(xy, 'N1C', 'N1B'), revPeak: peak(yx, 'N1A', 'N1D'),
    fwdCut: avg(xyCut, Yn, 15, 20), afterBlank: avg(by, Yn, 15, 20), omitN: avg(xo, Yn, 15, 20), preY: avg(xy, Ye, 10, 15),
    traceX: pair(xy, 'N1C', 'N1B').slice(12, 27), traceRev: pair(yx, 'N1A', 'N1D').slice(12, 27), traceCut: pair(xyCut, 'N1C', 'N1B').slice(12, 27),
  };
  r.asym = r.rev - r.fwd; r.cutGain = r.fwdCut - r.fwd;
  say(`  【${name}】第一层误差自发 ${base.toFixed(1)} Hz；预测抑制：X 的中继 → Y 的误差 ${r.w.toFixed(2)}、Y 的中继 → X 的误差 ${r.wYX.toFixed(2)}、X 的中继 → X 的误差 ${r.wXX.toFixed(2)}`);
  say(`    后一项出现后头 50 ms 它的误差：X → Y 的 Y ${r.fwd.toFixed(1)} Hz，Y → X 的 X ${r.rev.toFixed(1)} Hz（反向 − 正向 ${r.asym.toFixed(1)}）；10 ms 峰值 ${r.fwdPeak.toFixed(1)} / ${r.revPeak.toFixed(1)} Hz`);
  say(`    置零「X 的中继 → Y 的误差」后 X → Y 的 Y ${r.fwdCut.toFixed(1)} Hz（升高 ${r.cutGain.toFixed(1)}）；空白之后的 Y ${r.afterBlank.toFixed(1)} Hz；X 之后 Y 没来 ${r.omitN.toFixed(1)} Hz；X 的最后 50 ms 上层 Y 的组 ${r.preY.toFixed(1)} Hz`);
  say(`    后一项出现前后每 10 ms（-30 … +110 ms）它的误差：X → Y ${row(r.traceX)}`);
  say(`                                                Y → X ${row(r.traceRev)}`);
  say(`                                            X → Y 置零 ${row(r.traceCut)}`);
  return r;
}

/** 全部检验：只学物体的网络做物体层面的检验；两个网络都做序列检验。结果写到 figures/predictive-results.json，并画图 12 */
export function runPredictive({ log: say = log, figure = true } = {}) {
  const { curve } = trainPC({ log: say }), S = pcStack(), Sq = pcStack({ seq: true });
  say(`  学习曲线（X 引起的第一层误差，50–200 ms，每 25 个物体）：${curve.map((v) => v.toFixed(1)).join(' → ')} Hz`);
  const obj = objectTests(S, say);
  const seq = { obj: sequenceTests(S, '只学物体', say), seq: sequenceTests(Sq, '经历序列', say) };
  seq.expect = seq.seq.asym - seq.obj.asym; seq.expectCut = seq.seq.cutGain - seq.obj.cutGain;
  say(`  学到的期待（经历序列 − 只学物体）：反向 − 正向 ${seq.expect.toFixed(1)} Hz；置零「X 的中继 → Y 的误差」的升高 ${seq.expectCut.toFixed(1)} Hz`);
  const res = { cfg: S.cfg.pcP, curve, obj, seq };
  fs.mkdirSync('figures', { recursive: true });
  fs.writeFileSync(`figures/predictive-results${TAG}.json`, JSON.stringify(res));
  if (figure) drawPredictive(res, `figures/fig12-predictive${TAG}.png`);
  return res;
}

/** 图 12：第一层误差的时间进程、学习曲线、各种情形下的误差、Y 出现前后的误差 */
function drawPredictive(res, file) {
  const cv = new Canvas(1500, 860), dark = [60, 60, 60], gray = [160, 160, 160], blue = PALETTE[1], red = PALETTE[0];
  const dash = (xa, ya, xb, yb, col) => { const n = Math.max(1, Math.round(Math.hypot(xb - xa, yb - ya) / 6)); for (let k = 0; k < n; k += 2) cv.line(xa + ((xb - xa) * k) / n, ya + ((yb - ya) * k) / n, xa + ((xb - xa) * (k + 1)) / n, ya + ((yb - ya) * (k + 1)) / n, col, 3); };
  const plot = (x0, y0, w, h, lo, hi, traces, base) => {
    const y = (v) => y0 + h - ((Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo)) * h;
    cv.rect(x0, y0 + h, w, 1, gray); cv.rect(x0, y0, 1, h, gray); cv.text(x0 + 6, y0, `${hi} HZ`, gray, 1);
    if (base !== undefined) for (let x = 0; x < w; x += 8) cv.rect(x0 + x, y(base), 4, 1, red);
    for (const [arr, col, dashed] of traces) for (let i = 1; i < arr.length; i++) {
      const xa = x0 + ((i - 1) / (arr.length - 1)) * w, xb = x0 + (i / (arr.length - 1)) * w;
      if (dashed) dash(xa, y(arr[i - 1]), xb, y(arr[i]), col); else cv.line(xa, y(arr[i - 1]), xb, y(arr[i]), col, 3);
    }
  };
  const o = res.obj, tc = o.tc, ad = (t) => t.N1A.map((v, b) => (v + t.N1D[b]) / 2);
  cv.text(20, 12, 'L1 ERROR CELLS A D WHILE X IS SHOWN (0-200 MS)', [0, 0, 0], 2);
  plot(40, 50, 640, 260, 0, 50, [[ad(tc.cutP), dark, true], [ad(tc.cutT), gray], [ad(tc.famX), blue]], o.err.base);
  cv.text(40, 322, 'BLUE = TRAINED  GRAY = TOP-DOWN CUT  DASHED = PREDICTION CUT  RED = BASELINE  0-50 HZ', [0, 0, 0], 1);
  cv.text(760, 12, 'LEARNING: L1 ERROR TO X (EVERY 25 OBJECTS)', [0, 0, 0], 2);
  plot(780, 50, 640, 260, 0, 20, [[res.curve, blue]], o.err.base);
  cv.text(780, 322, `${res.curve.map((v) => v.toFixed(1)).join(' > ')} HZ`, [0, 0, 0], 1);
  cv.text(20, 372, 'L1 ERROR 50-200 MS (HZ)', [0, 0, 0], 2);
  const bars = [['X', o.err.fam, blue], ['ONLY A: A', o.err.onlyA, blue], ['ONLY A: D', o.err.missD, red], ['A+C: A C', o.err.novelAC, blue], ['A+C: B D', o.err.novelBD, red], ['X CUT', o.err.cutP, dark]];
  bars.forEach(([lab, v, col], k) => { const h = (Math.min(40, v) / 40) * 300, x = 40 + k * 108; cv.rect(x, 720 - h, 80, h, col); cv.text(x, 730, lab, [0, 0, 0], 1); cv.text(x, 700 - h, v.toFixed(1), [0, 0, 0], 1); });
  for (let x = 0; x < 650; x += 8) cv.rect(40 + x, 720 - (o.err.base / 40) * 300, 4, 1, red);
  cv.rect(40, 720, 650, 1, gray);
  cv.text(760, 372, 'L1 ERROR FOR THE SECOND ITEM: -30 TO +110 MS', [0, 0, 0], 2);
  const q = res.seq.seq, ob = res.seq.obj;
  plot(780, 410, 640, 300, 0, 50, [[ob.traceX, gray, true], [ob.traceRev, red, true], [q.traceRev, red], [q.traceX, blue]], q.base);
  cv.text(780, 722, 'SEQUENCE TRAINED: BLUE = Y AFTER X (LEARNED ORDER)  RED = X AFTER Y (NEVER SEEN)', [0, 0, 0], 1);
  cv.text(780, 736, `DASHED = SAME IN THE OBJECTS-ONLY NETWORK   LEARNED EXPECTATION ${res.seq.expect.toFixed(1)} HZ`, [0, 0, 0], 1);
  cv.save(file);
}

function calibrate() {
  const S = buildLayered(2, PC2); loadState(S, trainBal()); learnOnly(S, []);
  const sp = spontaneous(S);
  sp.forEach((s, k) => log(`  第${k + 1}层自发：E ${s.E.toFixed(2)}  I ${s.I.toFixed(2)}  N ${s.N.toFixed(2)}  S ${s.S.toFixed(2)} Hz`));
  const tc = timeCourse(S, OBJ.X);
  log('  呈现 X（A + D），每 25 ms 的频率：');
  for (const key of ['E1A', 'E1B', 'N1A', 'N1B', 'N1D', 'S1A', 'S1B', 'E2A', 'E2B', 'E2D', 'N2A', 'N2B']) log(`    ${key} ${row(tc[key])}`);
  const sq = sequenceCourse(S, [['AD', 150], ['CB', 150], ['', 200]], { reps: 4 });
  log('  X 150 ms → Y 150 ms，Y 出现前后每 10 ms（-30 … +110 ms）：');
  for (const key of ['E2A', 'E2C', 'S1A', 'S1C', 'N1C']) log(`    ${key} ${row(sq[key].slice(12, 27))}`);
}

if (process.argv[1]?.endsWith('predictive-hierarchy.mjs')) {
  if (process.env.CALIB) calibrate();
  else if (process.env.FIG) drawPredictive(JSON.parse(fs.readFileSync(`figures/predictive-results${TAG}.json`, 'utf8')), `figures/fig12-predictive${TAG}.png`);
  else { runPredictive(); log(`\n写入 figures/fig12-predictive${TAG}.png`); }
}
