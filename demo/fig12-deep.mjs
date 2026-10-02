/** 深层结构：每层分浅层（带顶树突、吸引子竞争、往上送证据）与深层（学有向通道、发出预测），
 * 预测从上一层深层落到下一层浅层的顶树突与下一层深层（见 src/L4-network/layered-network.mjs 的 deep）。
 * 世界与训练同 demo/lib-hierarchy.mjs（X = A + D、Y = C + B，X 之后是 Y）；序列训练时只有深层学习——第二层若出现 Y 的预测，只可能来自深层。
 * 测试：
 *   预测的源头：只给 40 ms 的 X，第二层深层 Y 的组；传给下一层深层：第一层深层 Y 的组；传给下一层浅层：第一层浅层 Y 的组的顶树突；
 *   易化：Y 紧跟 X vs 空白之后 Y，第一层浅层 Y 的组的首个脉冲延迟与早期放电；
 *   第一层自己的深层：只给 A 的 10 个成员，第一层深层 D 的组、浅层 D 的组的顶树突；
 *   对照：第二层没经历物体序列；切断深层发出的层间反馈。
 * 用法：node demo/fig12-deep.mjs（首次运行先训练，约 8 分钟）→ figures/fig12-deep.png */
import fs from 'fs';
import { cfg, proto, OBJ, mean, trainedStack } from './lib-hierarchy.mjs';
import { inputRate, spikesIn } from '../src/L4-network/layered-network.mjs';
import { makeRng } from '../src/core/rng.mjs';
import { Canvas, PALETTE } from '../src/core/png.mjs';

const log = console.log, BIN = 5, BL = ['A', 'B', 'C', 'D'];
const XB = OBJ.X.map((p) => BL.indexOf(p)), YB = OBJ.Y.map((p) => BL.indexOf(p));
const at = (ids, hz) => new Map(ids.map((i) => [i, hz]));
const Xs = at(OBJ.X.flatMap(proto), cfg.seqStim), Ys = at(OBJ.Y.flatMap(proto), cfg.seqStim);

/** segments 重复 reps 次：各层浅层 / 深层各组 5 ms 分箱的放电率，第一层浅层某些组的顶树突与胞体平均电位（1 ms），某窗口内的首个脉冲延迟 */
function trial(S, segments, { reps = 12, vGroups = YB, win = null } = {}) {
  const total = segments.reduce((s, [ms]) => s + ms, 0), nb = total / BIN;
  const course = S.layers.map(() => ({ E: BL.map(() => new Float64Array(nb)), D: BL.map(() => new Float64Array(nb)) }));
  const vd = new Float64Array(total), vs = new Float64Array(total), vIds = vGroups.flatMap((q) => proto(BL[q]).map((a) => S.layers[0].E[a]));
  const lats = []; let fired = 0;
  for (let r = 0; r < reps; r++) {
    S.net.run(400, { rate: inputRate(S), learn: false });
    const ts = S.net.t; let step = 0;
    const onStep = () => { if (step++ % 10) return; const ms = step / 10 | 0; if (ms < total) { vd[ms] += mean(vIds.map((i) => S.net.Vd[i])) / reps; vs[ms] += mean(vIds.map((i) => S.net.V[i])) / reps; } };
    for (const [ms, d] of segments) S.net.run(ms, { rate: inputRate(S, d), learn: false, onStep });
    S.layers.forEach((L, k) => { for (const pop of ['E', 'D']) BL.forEach((_, q) => spikesIn(S.net, proto(BL[q]).map((a) => L[pop][a]), ts, ts + total).forEach((s) => s.forEach((t) => (course[k][pop][q][Math.floor((t - ts) / BIN)] += 1 / (20 * BIN / 1000) / reps)))); });
    if (win) for (const s of spikesIn(S.net, vIds, ts + win[0], ts + win[1])) if (s.length) { fired++; lats.push(s[0] - ts - win[0]); }
  }
  return { course, vd: Array.from(vd), vs: Array.from(vs), latency: lats.length ? mean(lats) : null, firing: fired / (reps * vIds.length) };
}
const avg = (arr, a, b) => mean(Array.from(arr).slice(a / BIN, b / BIN));
const grp = (c, qs, a, b) => mean(qs.map((q) => avg(c[q], a, b)));

function study(S, name) {
  const T = cfg.seqItemMs, K = S.layers.length;
  const rest = trial(S, [[200, new Map()]], { reps: 4 });
  const vd0 = mean(rest.vd), vs0 = mean(rest.vs);
  const P = trial(S, [[T, Xs], [200 - T, new Map()]], { reps: 16 });
  const F = trial(S, [[T, Xs], [40, Ys], [160 - T, new Map()]], { win: [T, T + 60] });
  const N = trial(S, [[T, new Map()], [40, Ys], [160 - T, new Map()]], { win: [T, T + 60] });
  const rng = makeRng(9), A10 = at(rng.shuffle(proto(OBJ.X[0]).slice()).slice(0, 10), cfg.stim);
  const W = trial(S, [[150, A10], [50, new Map()]], { reps: 8, vGroups: [BL.indexOf(OBJ.X[1])] });
  const wRest = trial(S, [[100, new Map()]], { reps: 3, vGroups: [BL.indexOf(OBJ.X[1])] });
  const r = {
    name, K,
    deepY: S.layers.map((_, k) => grp(P.course[k].D, YB, 0, 120)), deepX: S.layers.map((_, k) => grp(P.course[k].D, XB, 0, 120)),
    surfY: S.layers.map((_, k) => grp(P.course[k].E, YB, 0, 120)),
    vdY: mean(P.vd.slice(10, 120)) - vd0, vdYat40: mean(P.vd.slice(30, 50)) - vd0, vsY: mean(P.vs.slice(10, 120)) - vs0,
    latExp: F.latency, latNo: N.latency, fireExp: F.firing, fireNo: N.firing,
    earlyExp: grp(F.course[0].E, YB, T, T + 20), earlyNo: grp(N.course[0].E, YB, T, T + 20),
    within: { deep: grp(W.course[0].D, [BL.indexOf(OBJ.X[1])], 0, 150), surf: grp(W.course[0].E, [BL.indexOf(OBJ.X[1])], 0, 150), vd: mean(W.vd.slice(10, 150)) - mean(wRest.vd) },
    vdTrace: P.vd.slice(0, 160).map((v) => +(v - vd0).toFixed(3)), course: P.course,
  };
  log(`\n【${name}】`);
  log(`  只给 40 ms 的 X（0–120 ms）：深层 Y 的组 ${r.deepY.map((v, k) => `第${k + 1}层 ${v.toFixed(1)}`).join('、')} Hz；深层 X 的组 ${r.deepX.map((v, k) => `第${k + 1}层 ${v.toFixed(1)}`).join('、')} Hz；浅层 Y 的组 ${r.surfY.map((v, k) => `第${k + 1}层 ${v.toFixed(1)}`).join('、')} Hz`);
  log(`  第一层浅层 Y 的组：顶树突比静息高 ${r.vdY.toFixed(2)} mV（10–120 ms 平均），30–50 ms ${r.vdYat40.toFixed(2)} mV；胞体 ${r.vsY.toFixed(2)} mV`);
  log(`  Y 紧跟 X：首个脉冲延迟 ${r.latExp?.toFixed(1)} ms、头 20 ms ${r.earlyExp.toFixed(1)} Hz、放电神经元 ${(r.fireExp * 100).toFixed(0)}%；空白之后 Y：${r.latNo?.toFixed(1)} ms、${r.earlyNo.toFixed(1)} Hz、${(r.fireNo * 100).toFixed(0)}%`);
  log(`  第一层自己的深层（只给 ${OBJ.X[0]} 的 10 个）：深层 ${OBJ.X[1]} 组 ${r.within.deep.toFixed(1)} Hz，浅层 ${OBJ.X[1]} 组 ${r.within.surf.toFixed(1)} Hz，浅层 ${OBJ.X[1]} 组顶树突比静息高 ${r.within.vd.toFixed(2)} mV`);
  return r;
}

export function runDeep({ arch = 'deep' } = {}) {
  const nets = [['1 层', trainedStack(1, { arch })], ['2 层', trainedStack(2, { arch })], ['2 层，第二层没经历物体序列', trainedStack(2, { arch, variant: 'noseq' })]];
  const cut = trainedStack(2, { arch }); cut.layers[1].FBA.scale.fill(0); cut.layers[1].FBD.scale.fill(0);
  nets.push(['2 层，切断深层发出的反馈', cut]);
  const res = nets.map(([n, S]) => study(S, n));
  const [r1, r2, rNo, rCut] = res;
  log(`\n预测本身的作用（2 层减去没经历序列）：第二层深层 Y 的组 +${(r2.deepY[1] - rNo.deepY[1]).toFixed(1)} Hz，第一层深层 Y 的组 +${(r2.deepY[0] - rNo.deepY[0]).toFixed(1)} Hz，第一层浅层 Y 的组顶树突 +${(r2.vdY - rNo.vdY).toFixed(2)} mV（30–50 ms +${(r2.vdYat40 - rNo.vdYat40).toFixed(2)} mV），Y 紧跟 X 时首个脉冲早 ${(rNo.latExp - r2.latExp).toFixed(1)} ms`);

  // 图 12：左上 = 2 层网络只给 X 时各群的时间进程；右上 = 第一层浅层 Y 的组顶树突；下 = 预测沿通路传递的强度与易化
  fs.mkdirSync('figures', { recursive: true });
  const cv = new Canvas(1500, 800);
  const plotLine = (arr, x0, y0, w, h, lo, hi, col, width = 3, dashed = false) => { for (let i = 1; i < arr.length; i++) { if (dashed && i % 4 < 2) continue; const y = (v) => y0 + h - ((Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo)) * h; cv.line(x0 + ((i - 1) / arr.length) * w, y(arr[i - 1]), x0 + (i / arr.length) * w, y(arr[i]), col, width); } };
  cv.text(20, 12, 'X FOR 40 MS (2 LAYERS): WHO FIRES WHEN', [0, 0, 0], 2);
  const lines = [['L1 SURF X', (c) => c[0].E, XB, PALETTE[0]], ['L2 DEEP X', (c) => c[1].D, XB, PALETTE[3]], ['L2 DEEP Y', (c) => c[1].D, YB, PALETTE[1]], ['L1 DEEP Y', (c) => c[0].D, YB, PALETTE[2]]];
  const series = lines.map(([, get, qs]) => { const g = get(r2.course); return Array.from(g[qs[0]], (v, b) => (v + g[qs[1]][b]) / 2).slice(0, 32); });
  const hi = Math.max(...series.flat(), 1);
  series.forEach((s, j) => { plotLine(s, 40, 50, 640, 250, 0, hi, lines[j][3]); cv.text(700, 60 + j * 26, lines[j][0], lines[j][3], 2); });
  cv.rect(40, 300, 640, 1, [150, 150, 150]); cv.rect(40 + 640 * 40 / 160, 50, 1, 250, [200, 200, 200]);
  cv.text(40, 310, '0-160 MS   X 0-40 MS', [0, 0, 0], 1);
  cv.text(860, 12, 'L1 SURFACE Y GROUPS: APICAL DENDRITE', [0, 0, 0], 2);
  const vds = [r2, rNo, rCut].map((r) => r.vdTrace.slice(0, 120)), lo = Math.min(...vds.flat()) - 0.3, vhi = Math.max(...vds.flat()) + 0.3;
  plotLine(vds[0], 880, 50, 580, 250, lo, vhi, PALETTE[1]); plotLine(vds[1], 880, 50, 580, 250, lo, vhi, [150, 150, 150]); plotLine(vds[2], 880, 50, 580, 250, lo, vhi, [0, 0, 0], 1, true);
  cv.rect(880, 300, 580, 1, [150, 150, 150]); cv.rect(880 + 580 * 40 / 120, 50, 1, 250, [200, 200, 200]);
  cv.text(880, 310, `${lo.toFixed(1)} TO ${vhi.toFixed(1)} MV ABOVE REST  BLUE 2 LAYERS  GRAY NO X-THEN-Y  DASHED FB CUT`, [0, 0, 0], 1);
  cv.text(20, 360, 'PREDICTED Y DOWN THE PATH (HZ / MV)  BLUE 2 LAYERS  GRAY NO SEQ  DARK FB CUT', [0, 0, 0], 2);
  const bars = [['L2 DEEP Y HZ', (r) => r.deepY[1] ?? 0, 20], ['L1 Y APICAL MV (GIVEN X)', (r) => r.vdY, 10], ['L1 PARTNER APICAL MV (GIVEN A)', (r) => r.within.vd, 10], ['Y AFTER X: FIRST SPIKE MS', (r) => r.latExp ?? 0, 40]];
  bars.forEach(([lab, get, mx], b) => {
    const x0 = 40 + b * 360;
    cv.text(x0, 400, lab, [0, 0, 0], 1);
    [r2, rNo, rCut].forEach((r, j) => { const h = (Math.min(mx, Math.max(0, get(r))) / mx) * 300; cv.rect(x0 + j * 90, 740 - h, 70, h, [PALETTE[1], [150, 150, 150], [60, 60, 60]][j]); });
    cv.rect(x0, 740, 300, 1, [150, 150, 150]);
  });
  const sfx = arch === 'deep' ? '' : `-${arch}`;
  cv.save(`figures/fig12-deep${sfx}.png`);
  fs.writeFileSync(`figures/deep-results${sfx}.json`, JSON.stringify(res.map(({ course, ...r }) => r)));
  log(`\n写入 figures/fig12-deep${sfx}.png`);
  return res;
}

if (import.meta.url === `file:///${process.argv[1].replace(/\\/g, '/')}`) runDeep({ arch: process.argv[2] ?? 'deep' });
