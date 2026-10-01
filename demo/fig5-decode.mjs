/** 吸引子与有向通道建好之后，给网络一个新的刺激，能否从它的激活里读出"哪些吸引子被激活、各自多强"？
 * 1. 训练：与 demo/fig2-network.mjs 相同的两个阶段（变体 150 次 + 序列 A→B→C→D 40 遍），结果缓存到 cache/
 * 2. 找吸引子：不用原型标签，只看网络自己学到的突触（对称化的效能矩阵），用标签传播划分社团，再与真实原型核对
 * 3. 读出：把响应投影到每个吸引子上——激活强度 = 成员平均放电；吸引子自身的贡献 = 没被直接刺激的成员被拉起多少；
 *    同样的刺激也施加在未训练的网络上作对照
 * 用法：node demo/fig5-decode.mjs → figures/fig5-decode.png */
import fs from 'fs';
import { buildNetwork, CFG } from './fig2-network.mjs';
import { makeRng } from '../src/core/rng.mjs';
import { Canvas, PALETTE } from '../src/core/png.mjs';

const cfg = CFG, { NE, K, M } = cfg;
const mean = (x) => x.reduce((s, v) => s + v, 0) / (x.length || 1);
const proto = Array.from({ length: K }, (_, k) => Array.from({ length: M }, (_, i) => k * M + i));
const groupOf = (i) => (i < K * M ? Math.floor(i / M) : -1);

function train(log) {
  const built = buildNetwork(cfg), { net, EE } = built;
  const ie = net.groups.find((g) => g.istdp);
  const file = `cache/trained-seed${cfg.seed}.json`;
  if (fs.existsSync(file)) {
    const s = JSON.parse(fs.readFileSync(file, 'utf8'));
    EE.rho.set(s.rho); for (let k = 0; k < EE.rho.length; k++) EE.scale[k] = 0.1 + (cfg.sMax - 0.1) * EE.rho[k];
    ie.scale.set(s.ie);
    log(`读入已训练的突触状态 ${file}`);
    return built;
  }
  const rng = makeRng(cfg.seed + 2);
  const variant = (k) => new Set([...rng.shuffle(proto[k].slice()).slice(0, 16), ...rng.shuffle([...Array(NE).keys()].filter((i) => groupOf(i) !== k)).slice(0, 4)]);
  const drive = (set, r) => (i) => (i < NE ? (set.has(i) ? r : cfg.bg) : 0);
  const t0 = Date.now();
  net.run(500, { rate: drive(new Set(), 0), learn: false });
  for (let n = 0; n < cfg.nLearn; n++) { net.run(200, { rate: drive(variant(rng.pick(K)), cfg.stim), learn: true }); net.run(300, { rate: drive(new Set(), 0), learn: true }); if (n % 50 === 49) log(`  训练：第一阶段 ${n + 1}/${cfg.nLearn}，${((Date.now() - t0) / 1000).toFixed(0)} s`); }
  for (let n = 0; n < cfg.nSeq; n++) { for (let k = 0; k < K; k++) net.run(cfg.seqItemMs, { rate: drive(new Set(proto[k]), cfg.seqStim), learn: true }); net.run(cfg.seqRestMs, { rate: drive(new Set(), 0), learn: true }); }
  log(`  训练完成，${((Date.now() - t0) / 1000).toFixed(0)} s`);
  fs.mkdirSync('cache', { recursive: true });
  fs.writeFileSync(file, JSON.stringify({ rho: [...EE.rho], ie: [...ie.scale] }));
  return built;
}

/** 只看学到的突触找吸引子：井由对称部分决定、有向通道是反对称部分，所以只用"双向都强"的连接——
 * 两个方向都有突触时取 min(ρ_ij, ρ_ji)（单向强的通道 A→B 因此不会把两口井并成一口），只有一个方向时取它的一半；
 * 强连接（> θ）上做标签传播，成员 ≥ 5 的社团算一个吸引子 */
function discoverAttractors(EE, theta = 0.25) {
  const R = Array.from({ length: NE }, () => new Float64Array(NE).fill(NaN));
  for (let k = 0; k < EE.pre.length; k++) { const a = EE.pre[k], b = EE.post[k]; if (a < NE && b < NE) R[b][a] = EE.rho[k]; }
  const S = Array.from({ length: NE }, () => new Float64Array(NE));
  for (let i = 0; i < NE; i++) for (let j = 0; j < NE; j++) {
    if (i === j) continue;
    const ab = R[i][j], ba = R[j][i];
    S[i][j] = Number.isNaN(ab) && Number.isNaN(ba) ? 0 : Number.isNaN(ab) ? ba / 2 : Number.isNaN(ba) ? ab / 2 : Math.min(ab, ba);
  }
  const label = [...Array(NE).keys()];
  for (let it = 0; it < 30; it++) {
    let changed = false;
    for (let i = 0; i < NE; i++) {
      const score = new Map();
      for (let j = 0; j < NE; j++) if (j !== i && S[i][j] > theta) score.set(label[j], (score.get(label[j]) ?? 0) + S[i][j]);
      if (!score.size) continue;
      const best = [...score].sort((x, y) => y[1] - x[1] || x[0] - y[0])[0][0];
      if (best !== label[i]) { label[i] = best; changed = true; }
    }
    if (!changed) break;
  }
  const groups = new Map();
  label.forEach((l, i) => { if (!groups.has(l)) groups.set(l, []); groups.get(l).push(i); });
  let comms = [...groups.values()].filter((g) => g.length >= 5);
  const meanS = (a, b) => { let s = 0, n = 0; for (const i of a) for (const j of b) if (i !== j) { s += S[i][j]; n++; } return n ? s / n : 0; };
  // 合并：两个社团之间的双向连接与各自内部一样强（≥ 内部较弱者的一半）就是同一口井被标签传播切开了
  for (let merged = true; merged;) {
    merged = false;
    for (let a = 0; a < comms.length && !merged; a++) for (let b = a + 1; b < comms.length && !merged; b++) {
      if (meanS(comms[a], comms[b]) >= 0.5 * Math.min(meanS(comms[a], comms[a]), meanS(comms[b], comms[b]))) { comms[a] = [...comms[a], ...comms[b]]; comms.splice(b, 1); merged = true; }
    }
  }
  // 剔除挂靠成员：与本社团其余成员的平均双向连接不到社团中位数的一半
  comms = comms.map((g) => { const m = g.map((i) => meanS([i], g)); const med = [...m].sort((x, y) => x - y)[m.length >> 1]; return g.filter((_, k) => m[k] >= 0.5 * med); });
  return comms.filter((g) => g.length >= 5).sort((a, b) => a[0] - b[0]);
}

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
const trained = train(log);
const naive = buildNetwork(cfg);
const attractors = discoverAttractors(trained.EE);
const purity = attractors.map((g) => { const c = [0, 0, 0, 0, 0]; for (const i of g) c[groupOf(i) + 1]++; const best = Math.max(...c.slice(1)); return { size: g.length, label: 'ABCD'[c.slice(1).indexOf(best)], purity: best / g.length, recall: best / M }; });
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
