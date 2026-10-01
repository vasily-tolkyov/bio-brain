/** 吸引子与有向通道建好之后，给网络一个新的刺激，能否从它的激活里读出"哪些吸引子被激活、各自多强"？
 * 1. 训练：与 demo/fig2-network.mjs 相同的两个阶段（变体 150 次 + 序列 A→B→C→D 40 遍），结果缓存到 cache/
 * 2. 找吸引子：不用原型标签，只看网络自己学到的突触（对称化的效能矩阵），用标签传播划分社团，再与真实原型核对
 * 3. 读出：把响应投影到每个吸引子上——激活强度 = 成员平均放电；吸引子自身的贡献 = 没被直接刺激的成员被拉起多少；
 *    同样的刺激也施加在未训练的网络上作对照
 * 用法：node demo/fig5-decode.mjs → figures/fig5-decode.png */
import fs from 'fs';
import { buildNetwork } from './fig2-network.mjs';
import { cfg, NE, mean, proto, groupOf, trainedNetwork, discoverAttractors, labelAttractors } from './lib-trained.mjs';
import { makeRng } from '../src/core/rng.mjs';
import { Canvas, PALETTE } from '../src/core/png.mjs';

/** 施加一个刺激（每个被刺激神经元的输入速率可不同），返回 200 ms 内各神经元放电数与各吸引子 10 ms 分箱的放电率 */
function apply(net, drives, attractors, { ms = 200, reps = 3 } = {}) {
  const counts = new Array(NE).fill(0), course = attractors.map(() => new Array(ms / 10).fill(0));
  for (let r = 0; r < reps; r++) {
    net.run(400, { rate: (i) => (i < NE ? cfg.bg : 0), learn: false });
    const t0 = net.t;
    net.run(ms, { rate: (i) => (i < NE ? drives.get(i) ?? cfg.bg : 0), learn: false });
    net.counts(t0, net.t, [...Array(NE).keys()]).forEach((c, i) => (counts[i] += c / reps));
    attractors.forEach((g, a) => { for (let b = 0; b < ms / 10; b++) course[a][b] += mean(net.counts(t0 + b * 10, t0 + (b + 1) * 10, g)) / 0.01 / reps; });
  }
  return { rate: counts.map((c) => c / (ms / 1000)), course };
}

/** 投影到每个吸引子：成员平均放电（Hz）、激活度（0 = 静息，1 = 整个吸引子被完全驱动时）、被刺激覆盖的比例、
 * 没被刺激的成员被拉起的平均放电（吸引子自身的贡献） */
function decode(resp, drives, attractors, cal) {
  return attractors.map((g, a) => {
    const stim = g.filter((i) => drives.has(i)), free = g.filter((i) => !drives.has(i));
    const rate = mean(g.map((i) => resp.rate[i]));
    return { rate, level: (rate - cal[a].rest) / (cal[a].full - cal[a].rest), coverage: stim.length / g.length, recruited: free.length ? mean(free.map((i) => resp.rate[i])) : null };
  });
}
/** 标定：无刺激时与整个吸引子被完全驱动时，各吸引子成员的平均放电 */
function calibrate(net, attractors) {
  return attractors.map((g) => ({ rest: mean(apply(net, new Map(), [g]).rate.filter((_, i) => g.includes(i))), full: mean(apply(net, new Map(g.map((i) => [i, cfg.stim])), [g]).rate.filter((_, i) => g.includes(i))) }));
}
/** 各吸引子成员第一次放电的平均潜伏期（ms，从刺激开始算） */
function latency(net, drives, attractors, reps = 6) {
  const lat = attractors.map(() => []);
  for (let r = 0; r < reps; r++) {
    net.run(400, { rate: (i) => (i < NE ? cfg.bg : 0), learn: false });
    const t0 = net.t;
    net.run(200, { rate: (i) => (i < NE ? drives.get(i) ?? cfg.bg : 0), learn: false });
    attractors.forEach((g, a) => { for (const i of g) { if (drives.has(i)) continue; const s = net.spikes[i].find((x) => x >= t0 && x < t0 + 200); if (s !== undefined) lat[a].push(s - t0); } });
  }
  return lat.map((l) => (l.length >= 3 ? mean(l) : null));
}

const log = console.log;
const trained = trainedNetwork(log);
const naive = buildNetwork(cfg);
const attractors = discoverAttractors(trained.EE);
const purity = labelAttractors(attractors);
log(`只看学到的突触找到 ${attractors.length} 个吸引子：${purity.map((p) => `${p.label}（${p.size} 个神经元，纯度 ${(p.purity * 100).toFixed(0)}%，覆盖原型 ${(p.recall * 100).toFixed(0)}%）`).join('，')}`);
const names = purity.map((p) => p.label);

const rng = makeRng(77);
const pick = (arr, n) => rng.shuffle(arr.slice()).slice(0, n);
const others = [...Array(NE).keys()].filter((i) => groupOf(i) < 0);
const P = (L) => proto['ABCD'.indexOf(L)]; // 刺激按实验者定义的原型构造，与网络自己找到的吸引子无关
const set = (ids, r) => new Map(ids.map((i) => [i, r]));
const merge = (...maps) => new Map(maps.flatMap((m) => [...m]));
const stimuli = [
  ['A 的残缺版 + 噪声（A 的 10 个 + 10 个随机）', merge(set(pick(P('A'), 10), cfg.stim), set(pick([...Array(NE).keys()].filter((i) => groupOf(i) !== 0), 10), cfg.stim))],
  ['A 与 C 混合，强度 1:0（各取 10 个成员）', set(pick(P('A'), 10), cfg.stim)],
  ['A 与 C 混合，强度 3:1', merge(set(pick(P('A'), 10), cfg.stim), set(pick(P('C'), 10), cfg.stim / 3))],
  ['A 与 C 混合，强度 1:1', merge(set(pick(P('A'), 10), cfg.stim), set(pick(P('C'), 10), cfg.stim))],
  ['A 与 C 混合，强度 1:3', merge(set(pick(P('A'), 10), cfg.stim / 3), set(pick(P('C'), 10), cfg.stim))],
  ['A 与 C 混合，强度 0:1', set(pick(P('C'), 10), cfg.stim)],
  ['全新：只落在原型之外的 20 个神经元', set(others.slice(0, 20), cfg.stim)],
  ['全新：每个原型各 3 个 + 8 个无关', merge(...'ABCD'.split('').map((L) => set(pick(P(L), 3), cfg.stim)), set(pick(others, 8), cfg.stim))],
];

const calT = calibrate(trained.net, attractors), calN = calibrate(naive.net, attractors);
log(`标定（训练后）：${calT.map((c, a) => `${names[a]} 静息 ${c.rest.toFixed(1)} / 完全驱动 ${c.full.toFixed(1)} Hz`).join('，')}`);
const rows = [];
for (const [name, drives] of stimuli) {
  const rt = decode(apply(trained.net, drives, attractors), drives, attractors, calT);
  const rn = decode(apply(naive.net, drives, attractors), drives, attractors, calN);
  rows.push({ name, rt, rn });
  const f = (d) => d.map((x, a) => `${names[a]} ${x.level.toFixed(2)}${x.recruited === null ? '' : `（未刺激成员 ${x.recruited.toFixed(1)} Hz）`}`).join('  ');
  log(`\n【${name}】刺激覆盖 ${rt.map((x, a) => `${names[a]} ${(x.coverage * 100).toFixed(0)}%`).join(' ')}\n  训练后激活度：${f(rt)}\n  未训练激活度：${f(rn)}`);
}

// 有向通道：只刺激 A 的一半成员时，各吸引子未被刺激成员的首次放电潜伏期与平均放电
const seqDr = set(pick(P('A'), 10), cfg.stim);
const course = apply(trained.net, seqDr, attractors, { ms: 200, reps: 6 }).course;
const lat = latency(trained.net, seqDr, attractors);
log(`\n只刺激 A 的一半成员：各吸引子平均放电 ${course.map((c, a) => `${names[a]} ${mean(c).toFixed(1)} Hz`).join('，')}；未被刺激成员的首次放电潜伏期 ${lat.map((v, a) => `${names[a]} ${v === null ? '—' : v.toFixed(0) + ' ms'}`).join('，')}`);

// 图 5：每个刺激一组柱（训练后实心、未训练空心），以及只刺激 A 时的时间进程
fs.mkdirSync('figures', { recursive: true });
const cv = new Canvas(1300, 760);
const maxR = Math.max(...rows.flatMap((r) => [...r.rt, ...r.rn].map((x) => x.level)), 0.5);
rows.forEach((r, s) => {
  const x0 = 40 + s * 155, y0 = 60, h = 300;
  cv.text(x0, 20, `S${s + 1}`, [0, 0, 0], 2);
  cv.rect(x0, y0 + h, 140, 1, [150, 150, 150]);
  r.rt.forEach((d, a) => {
    const bh = (Math.max(0, d.level) / maxR) * h, bn = (Math.max(0, r.rn[a].level) / maxR) * h, x = x0 + a * 34;
    cv.rect(x, y0 + h - bh, 15, bh, PALETTE[a]);
    cv.rect(x + 16, y0 + h - bn, 12, 1, PALETTE[a]); cv.rect(x + 16, y0 + h - bn, 1, bn, PALETTE[a]); cv.rect(x + 27, y0 + h - bn, 1, bn, PALETTE[a]);
    cv.text(x + 2, y0 + h + 8, names[a], PALETTE[a], 2);
  });
});
cv.text(40, 400, 'DRIVE HALF OF A: RATE OF EACH ATTRACTOR OVER 200 MS', [0, 0, 0], 2);
const cmax = Math.max(...course.flat(), 1);
course.forEach((c, a) => { for (let b = 1; b < c.length; b++) cv.line(60 + (b - 1) * 60, 720 - (c[b - 1] / cmax) * 280, 60 + b * 60, 720 - (c[b] / cmax) * 280, PALETTE[a], 3); });
cv.rect(60, 720, 1140, 1, [150, 150, 150]);
cv.save('figures/fig5-decode.png');
log('\n写入 figures/fig5-decode.png');
