/** 演示共用：训练好的脉冲网络（两个阶段，结果缓存到 cache/）、只看学到的突触找吸引子、施加刺激、读出激活度 */
import fs from 'fs';
import { buildNetwork, CFG } from './fig2-network.mjs';
import { makeRng } from '../src/core/rng.mjs';

export const cfg = CFG;
export const { NE, K, M } = CFG;
export const mean = (x) => x.reduce((s, v) => s + v, 0) / (x.length || 1);
export const proto = Array.from({ length: K }, (_, k) => Array.from({ length: M }, (_, i) => k * M + i));
export const groupOf = (i) => (i < K * M ? Math.floor(i / M) : -1);

export function trainedNetwork(log = () => {}, seed = CFG.seed) {
  const c = { ...CFG, seed };
  const built = buildNetwork(c), { net, EE } = built;
  const ie = net.groups.find((g) => g.istdp);
  const file = `cache/trained-seed${seed}.json`;
  if (fs.existsSync(file)) {
    const s = JSON.parse(fs.readFileSync(file, 'utf8'));
    EE.rho.set(s.rho); for (let k = 0; k < EE.rho.length; k++) EE.scale[k] = 0.1 + (c.sMax - 0.1) * EE.rho[k];
    ie.scale.set(s.ie);
    log(`读入已训练的突触状态 ${file}`);
    return built;
  }
  const rng = makeRng(seed + 2);
  const variant = (k) => new Set([...rng.shuffle(proto[k].slice()).slice(0, 16), ...rng.shuffle([...Array(NE).keys()].filter((i) => groupOf(i) !== k)).slice(0, 4)]);
  const drive = (set, r) => (i) => (i < NE ? (set.has(i) ? r : c.bg) : 0);
  const t0 = Date.now();
  net.run(500, { rate: drive(new Set(), 0), learn: false });
  for (let n = 0; n < c.nLearn; n++) { net.run(200, { rate: drive(variant(rng.pick(K)), c.stim), learn: true }); net.run(300, { rate: drive(new Set(), 0), learn: true }); if (n % 50 === 49) log(`  训练：第一阶段 ${n + 1}/${c.nLearn}，${((Date.now() - t0) / 1000).toFixed(0)} s`); }
  for (let n = 0; n < c.nSeq; n++) { for (let k = 0; k < K; k++) net.run(c.seqItemMs, { rate: drive(new Set(proto[k]), c.seqStim), learn: true }); net.run(c.seqRestMs, { rate: drive(new Set(), 0), learn: true }); }
  log(`  训练完成，${((Date.now() - t0) / 1000).toFixed(0)} s`);
  fs.mkdirSync('cache', { recursive: true });
  fs.writeFileSync(file, JSON.stringify({ rho: [...EE.rho], ie: [...ie.scale] }));
  return built;
}

/** 只看学到的突触找吸引子：井由对称部分决定，所以只用双向都强的连接（两个方向都有时取较小者，单向时取一半）；
 * 标签传播 → 合并被切开的同一口井 → 剔除挂靠成员 */
export function discoverAttractors(EE, theta = 0.25) {
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
  for (let merged = true; merged;) {
    merged = false;
    for (let a = 0; a < comms.length && !merged; a++) for (let b = a + 1; b < comms.length && !merged; b++) {
      if (meanS(comms[a], comms[b]) >= 0.5 * Math.min(meanS(comms[a], comms[a]), meanS(comms[b], comms[b]))) { comms[a] = [...comms[a], ...comms[b]]; comms.splice(b, 1); merged = true; }
    }
  }
  comms = comms.map((g) => { const m = g.map((i) => meanS([i], g)); const med = [...m].sort((x, y) => x - y)[m.length >> 1]; return g.filter((_, k) => m[k] >= 0.5 * med); });
  return comms.filter((g) => g.length >= 5).sort((a, b) => a[0] - b[0]);
}

export function labelAttractors(attractors) {
  return attractors.map((g) => { const c = [0, 0, 0, 0, 0]; for (const i of g) c[groupOf(i) + 1]++; const best = Math.max(...c.slice(1)); return { size: g.length, label: 'ABCD'[c.slice(1).indexOf(best)], purity: best / g.length, recall: best / M }; });
}

/** 施加刺激（drives：神经元 → 每条输入的速率），返回 ms 内各神经元的平均放电率 */
export function applyStim(net, drives, { ms = 200, reps = 3, nE = NE, bg = CFG.bg } = {}) {
  const counts = new Array(nE).fill(0);
  for (let r = 0; r < reps; r++) {
    net.run(400, { rate: (i) => (i < nE ? bg : 0), learn: false });
    const t0 = net.t;
    net.run(ms, { rate: (i) => (i < nE ? drives.get(i) ?? bg : 0), learn: false });
    net.counts(t0, net.t, [...Array(nE).keys()]).forEach((c, i) => (counts[i] += c / reps));
  }
  return counts.map((c) => c / (ms / 1000));
}
