/** 有向通道演化链 v2。GB_RATE、刺激期兴奋学习、组内加密与学习态增益。
 * MODE=attractors/learn-gain/sequence/calib/test/controls/fig/all；BG/BG_I 仅用于计划允许的背景标定。
 * 学习态：兴奋学习按窗口冻结，Vogels 全程学习；演化态：全部冻结、绝对增益 G、适应 B。
 * 缓存保存连接效能及明确的配置指纹；结果与验收使用计划指定的时间分箱。
 */
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { SpikingNetwork } from '../src/L4-network/spiking-network.mjs';
import { GB_NMDA, GB_RATE } from '../src/L3-plasticity/calcium-rule.mjs';
import { mammalianCosts } from '../src/L1-neuron/hh.mjs';
import { makeRng } from '../src/core/rng.mjs';
import { Canvas, heat, PALETTE } from '../src/core/png.mjs';

export const ROOT = fileURLToPath(new URL('../', import.meta.url));
export const LABELS = 'ABCDEFGH';
export const EDGES = [[0, 1], [1, 2], [2, 3], [2, 6], [3, 4], [4, 5], [6, 7]];
export const CONFIG = Object.freeze({
  version: 2, rule: 'GB_RATE', initialGain: 1, seed: 1, NE: 720, NI: 180, K: 8, M: 80, dt: 0.1,
  pIn: 0.25, pOut: 0.083, contacts: 10, sMin: 0.1, sMax: 9, rho0: 0.1,
  accel: 5, budget: 0.2, bg: 30, bgI: 12, stim: 52, seqStim: 58,
  b: 20, tauW: 300, pEI: 0.2, sEI: 1.5, pIE: 0.4, sIE: 4,
  pII: 0.4, sII: 4, eta: 0.02, target: 3, ieMax: 40,
});
const atPath = (...parts) => path.join(ROOT, ...parts);
const jsonRead = file => JSON.parse(fs.readFileSync(file, 'utf8'));
const jsonWrite = (file, value) => {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(`${file}.tmp`, `${JSON.stringify(value, null, 2)}\n`);
  fs.renameSync(`${file}.tmp`, file);
};
export const mean = xs => xs.reduce((s, x) => s + x, 0) / (xs.length || 1);
const sd = xs => { const m = mean(xs); return Math.sqrt(mean(xs.map(x => (x - m) ** 2))); };
const median = xs => { const a = xs.slice().sort((x, y) => x - y); return a.length ? (a[Math.floor((a.length - 1) / 2)] + a[Math.floor(a.length / 2)]) / 2 : 0; };
const hash = value => crypto.createHash('sha256').update(JSON.stringify(value)).digest('hex');
const weightHash = state => hash({ configHash: state.configHash, rho: state.rho, inhibition: state.inhibition });
export const cacheFile = (name, cfg = CONFIG) => atPath('cache', `chain-v2-${name}${cfg.rule === 'GB_RATE' ? '' : '-gbnmda'}.json`);
export const RULES = { GB_RATE, GB_NMDA, GB_NMDA_ETA0: { ...GB_NMDA, eta: 0 } };
const groupOf = (i, cfg) => i < cfg.K * cfg.M ? Math.floor(i / cfg.M) : -1;
const prototype = (g, cfg) => Array.from({ length: cfg.M }, (_, k) => g * cfg.M + k);
let costs;

// 与 core/makeRng 的 uniform/pick/shuffle 逐项相同，额外公开状态用于精确续算。
export function resumableRng(initial = 1) {
  let s = initial >>> 0;
  const uniform = () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return { uniform, pick: n => Math.floor(uniform() * n),
    shuffle: a => { for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(uniform() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; },
    state: () => s };
}

export function buildChain(cfg = CONFIG, state = null, { G = state?.learnGain ?? cfg.initialGain ?? 1, b = 20, noiseSeed = 1001 } = {}) {
  costs ??= mammalianCosts();
  const plasticity = RULES[cfg.rule];
  if (!plasticity) throw new Error(`未知规则 ${cfg.rule}`);
  const net = new SpikingNetwork({ costs, seed: cfg.seed, dt: cfg.dt, plasticity,
    pops: [{ name: 'E', n: cfg.NE, p: { b, tauW: cfg.tauW } }, { name: 'I', n: cfg.NI, p: { C: 100, tRef: 1 } }] });
  const pairs = [];
  for (let a = 0; a < cfg.NE; a++) for (let z = 0; z < cfg.NE; z++) {
    if (a === z) continue;
    const ga = groupOf(a, cfg), gz = groupOf(z, cfg);
    if (net.rng.uniform() < (ga >= 0 && ga === gz ? cfg.pIn : cfg.pOut)) pairs.push([a, z]);
  }
  const EE = net.connect({ pairs, contacts: cfg.contacts, pRelease: 0.3, tauRec: 100,
    plastic: { sMin: cfg.sMin, sMax: cfg.sMax, rho0: cfg.rho0, accel: cfg.accel, budget: cfg.budget } });
  net.connect({ from: 'E', to: 'I', p: cfg.pEI, contacts: 10, pRelease: 0.3, tauRec: 100, scale: cfg.sEI });
  const IE = net.connect({ from: 'I', to: 'E', p: cfg.pIE, contacts: 10, pRelease: 0.5, tauRec: 300, kind: 'inhFast', scale: cfg.sIE,
    istdp: { eta: cfg.eta, target: cfg.target, wMax: cfg.ieMax } });
  net.connect({ from: 'I', to: 'I', p: cfg.pII, contacts: 10, pRelease: 0.5, tauRec: 300, kind: 'inhFast', scale: cfg.sII });
  if (state) {
    if (state.configHash !== hash(cfg) || state.rho.length !== EE.rho.length || state.inhibition.length !== IE.scale.length) throw new Error('缓存与网络配置不匹配');
    EE.rho.set(state.rho); IE.scale.set(state.inhibition);
  }
  net.rng = resumableRng(noiseSeed);
  const S = { net, EE, IE, cfg, G, b, E: net.idx('E'), I: net.idx('I') };
  setGain(S, G);
  return S;
}

export function setGain(S, G) {
  if (!(G > 0) || !Number.isFinite(G)) throw new Error('增益必须为有限正数');
  S.EE.plastic.sMin = S.cfg.sMin * G; S.EE.plastic.sMax = S.cfg.sMax * G;
  for (let k = 0; k < S.EE.rho.length; k++) S.EE.scale[k] = S.EE.plastic.sMin + (S.EE.plastic.sMax - S.EE.plastic.sMin) * S.EE.rho[k];
  S.G = G;
}

const snapshot = S => ({ config: S.cfg, configHash: hash(S.cfg), learnGain: S.G, rho: Array.from(S.EE.rho), inhibition: Array.from(S.IE.scale) });
export function captureRuntime(S) {
  const net = S.net, vectors = {};
  for (const name of ['V', 'w', 'ref', 'Rin', 'x', 'y', 'y2', 'nmda', 'xs', 'ys']) if (net[name]) vectors[name] = Array.from(net[name]);
  return { t: net.t, vectors, ra: net.ra.map(a => Array.from(a)), rb: net.rb.map(a => Array.from(a)),
    ea: net.ea.map(a => Array.from(a)), eb: net.eb.map(a => Array.from(a)),
    groups: net.groups.map(g => ({ R: Array.from(g.R), z: g.z ? Array.from(g.z) : null })),
    pending: net.pending.slice(), pendHead: net.pendHead, rng: net.rng.state(), ledger: Array.from(net.ledger.atp) };
}
export function restoreRuntime(S, runtime) {
  const net = S.net; net.t = runtime.t;
  for (const [name, values] of Object.entries(runtime.vectors)) {
    net[name] ??= new Float64Array(net.N);
    net[name].set(name === 'ref' ? values.map(x => x === null ? -Infinity : x) : values);
  }
  for (const name of ['ra', 'rb', 'ea', 'eb']) runtime[name].forEach((a, k) => net[name][k].set(a));
  runtime.groups.forEach((g, k) => { net.groups[k].R.set(g.R); if (g.z) net.groups[k].z.set(g.z); });
  net.pending = runtime.pending.slice(); net.pendHead = runtime.pendHead;
  net.rng = resumableRng(runtime.rng); net.ledger.atp = new Map(runtime.ledger);
}
const rate = (S, ids = [], hz = S.cfg.stim) => {
  const rates = new Float64Array(S.net.N);
  rates.fill(S.cfg.bg, 0, S.cfg.NE); rates.fill(S.cfg.bgI, S.cfg.NE);
  for (const i of ids) rates[i] = hz;
  return i => rates[i];
};
export function run(S, ms, { ids = [], hz = S.cfg.stim, learn = false, lowNoise = false, excLearn = ids.length > 0 } = {}) {
  S.EE.frozen = !learn || !excLearn;
  if (S.EE.frozen) S.EE.z.fill(0);
  S.net.run(ms, { rate: rate(S, ids, hz), learn, ...(lowNoise ? { nIn: 600, inContacts: 1 } : {}) });
}
const clearSpikes = S => { for (const spikes of S.net.spikes) spikes.length = 0; };
export function groupBins(S, start, ms, binMs = 10) {
  const bins = Array.from({ length: Math.round(ms / binMs) }, () => Array(S.cfg.K).fill(0));
  for (let i = 0; i < S.cfg.K * S.cfg.M; i++) for (const t of S.net.spikes[i]) {
    const bin = Math.floor((t - start + 1e-8) / binMs);
    if (t >= start - 1e-8 && t < start + ms - 1e-8 && bin >= 0 && bin < bins.length) bins[bin][groupOf(i, S.cfg)]++;
  }
  const factor = 1000 / (S.cfg.M * binMs);
  return bins.map(row => row.map(n => n * factor));
}
const frequency = (S, ids, start, ms) => mean(ids.map(i => S.net.spikes[i].filter(t => t >= start && t < start + ms).length)) * 1000 / ms;

export function restMetrics(S, ms = 3000, threshold = 10, lowNoise = false) {
  clearSpikes(S); const start = S.net.t; run(S, ms, { lowNoise });
  const bin5 = Array(Math.round(ms / 5)).fill(0);
  for (const i of S.E) for (const t of S.net.spikes[i]) {
    const k = Math.floor((t - start + 1e-8) / 5);
    if (t >= start - 1e-8 && t < start + ms - 1e-8 && k >= 0 && k < bin5.length) bin5[k]++;
  }
  const bins10 = groupBins(S, start, ms), bins20 = groupBins(S, start, ms, 20);
  return { rE: frequency(S, S.E, start, ms), rI: frequency(S, S.I, start, ms),
    fano: mean(bin5) ? sd(bin5) ** 2 / mean(bin5) : 0,
    ignitionRate: mean(bins20.map(row => Number(row.some(r => r > threshold)))),
    bins10, threshold, ms, atp: S.net.ledger.total() };
}

export function weightMatrix(S) {
  const sum = Array.from({ length: S.cfg.K }, () => Array(S.cfg.K).fill(0));
  const count = sum.map(row => row.slice());
  for (let k = 0; k < S.EE.rho.length; k++) {
    const a = groupOf(S.EE.pre[k], S.cfg), z = groupOf(S.EE.post[k], S.cfg);
    if (a >= 0 && z >= 0) { sum[a][z] += S.EE.rho[k]; count[a][z]++; }
  }
  const matrix = sum.map((row, a) => row.map((x, z) => count[a][z] ? x / count[a][z] : 0));
  let offSum = 0, offCount = 0;
  for (let a = 0; a < S.cfg.K; a++) for (let z = 0; z < S.cfg.K; z++) if (a !== z) { offSum += sum[a][z]; offCount += count[a][z]; }
  const off = offSum / offCount, ratios = matrix.map((row, a) => row[a] / off);
  return { matrix, counts: count, off, ratios, meanRatio: mean(matrix.map((row, a) => row[a])) / off };
}

export function completionMetrics(state, { G = state.learnGain ?? 1 } = {}) {
  const cfg = state.config, S = buildChain(cfg, state, { G, noiseSeed: 2001 }), rng = makeRng(2002), rows = [];
  run(S, 500);
  for (let g = 0; g < cfg.K; g++) {
    const samples = [];
    for (let rep = 0; rep < 6; rep++) {
      run(S, 300); clearSpikes(S);
      const proto = prototype(g, cfg), cue = rng.shuffle(proto.slice()).slice(0, 32), unseen = proto.filter(i => !cue.includes(i));
      const outside = S.E.filter(i => groupOf(i, cfg) !== g), start = S.net.t;
      run(S, 150, { ids: cue });
      samples.push({ uncued: frequency(S, unseen, start, 150), outside: frequency(S, outside, start, 150) });
    }
    rows.push({ group: LABELS[g], uncued: mean(samples.map(x => x.uncued)), outside: mean(samples.map(x => x.outside)), delta: mean(samples.map(x => x.uncued - x.outside)), samples });
  }
  return { groups: rows, meanDelta: mean(rows.map(x => x.delta)), positiveGroups: rows.filter(x => x.delta > 0).length };
}

export function attractorChecks(r) {
  return [
    { id: '1.1', name: '训练前频率与异步性', ok: r.baseline.rE >= 2 && r.baseline.rE <= 5 && r.baseline.rI >= 5 && r.baseline.rI <= 20 && r.baseline.fano < 2,
      actual: { rE: r.baseline.rE, rI: r.baseline.rI, fano: r.baseline.fano } },
    { id: '1.2', name: '全部 8 组的组内/组间效能比至少 2', ok: r.weights.ratios.every(x => x >= 2), actual: r.weights.ratios },
    { id: '1.3', name: '补全平均至少 +0.5 Hz 且至少 7 组为正', ok: r.completion.meanDelta >= 0.5 && r.completion.positiveGroups >= 7, actual: { meanDelta: r.completion.meanDelta, positiveGroups: r.completion.positiveGroups } },
  ];
}

/** 10 ms 解码；只合并至多 30 ms 的无状态空档，短于 20 ms 的停留丢弃。 */
export function decodeRates(bins, threshold, binMs = 10) {
  const states = bins.map(row => {
    const ranked = row.map((rate, group) => ({ rate, group })).sort((a, b) => b.rate - a.rate);
    return ranked[0].rate >= threshold && ranked[0].rate - ranked[1].rate >= 5 ? ranked[0].group : -1;
  });
  const dwells = [];
  let current = null;
  const finish = () => {
    if (!current) return;
    const start = current.first * binMs, end = (current.last + 1) * binMs;
    if (end - start >= 20) dwells.push({ group: current.group, label: LABELS[current.group], start, end, duration: end - start,
      activeMs: current.rates.length * binMs, rate: mean(current.rates) });
    current = null;
  };
  for (let k = 0; k < states.length; k++) {
    const group = states[k];
    if (group < 0) continue;
    if (current && (current.group !== group || (k - current.last - 1) * binMs > 30)) finish();
    if (!current) current = { group, first: k, last: k, rates: [] };
    current.last = k; current.rates.push(bins[k][group]);
  }
  finish();
  const transitions = [];
  for (let k = 1; k < dwells.length; k++) {
    const previous = dwells[k - 1], next = dwells[k], gap = next.start - previous.end;
    if (previous.group !== next.group && gap <= 100) transitions.push({ from: previous.group, to: next.group, gap, at: next.start,
      forward: EDGES.some(([a, z]) => a === previous.group && z === next.group),
      reverse: EDGES.some(([a, z]) => z === previous.group && a === next.group) });
  }
  let chainLength = 0, completed = false;
  if (dwells[0]?.group === 0) {
    for (let k = 1; k < dwells.length; k++) {
      const previous = dwells[k - 1], next = dwells[k];
      if (next.start - previous.end > 100) break;
      if (previous.group === next.group) continue;
      if (!EDGES.some(([a, z]) => a === previous.group && z === next.group)) break;
      chainLength++;
      if (next.group === 5 || next.group === 7) { completed = true; break; }
    }
  }
  return { states, dwells, transitions, chainLength, completed,
    activeRates: states.flatMap((g, k) => g >= 0 ? [bins[k][g]] : []) };
}

export function summarizeTrials(trials) {
  const transitions = trials.flatMap(t => t.decoded.transitions), dwells = trials.flatMap(t => t.decoded.dwells);
  const durations = dwells.map(x => x.duration), matrix = Array.from({ length: 8 }, () => Array(8).fill(0));
  for (const t of transitions) matrix[t.from][t.to]++;
  const toD = matrix[2][3], toG = matrix[2][6];
  return { trials: trials.length, completionRate: mean(trials.map(t => Number(t.decoded.completed))),
    chainMedian: median(trials.map(t => t.decoded.chainLength)), chainLengths: trials.map(t => t.decoded.chainLength),
    transitionCount: transitions.length, forwardProportion: transitions.length ? mean(transitions.map(t => Number(t.forward))) : 0,
    reverseProportion: transitions.length ? mean(transitions.map(t => Number(t.reverse))) : 0,
    branchD: toD, branchG: toG, branchRatio: toG ? toD / toG : null,
    dwellMean: mean(durations), dwellCV: mean(durations) ? sd(durations) / mean(durations) : 0,
    dwellDurations: durations, firstDwellMedian: median(trials.map(t => t.decoded.dwells[0]?.duration ?? 0)),
    noTransitionIn1s: mean(trials.map(t => Number(!t.decoded.transitions.some(x => x.at < 1000)))),
    activeRate: mean(trials.flatMap(t => t.decoded.activeRates)), transitionMatrix: matrix };
}

export function chainChecks(r) {
  return [
    { id: '4.1', name: '沿正向连接到 F/H 的完成率至少 50%', ok: r.completionRate >= 0.5, actual: r.completionRate },
    { id: '4.2', name: '链长中位数至少 4', ok: r.chainMedian >= 4, actual: r.chainMedian },
    { id: '4.3', name: '正向至少 70%，反向不超过 10%', ok: r.forwardProportion >= 0.7 && r.reverseProportion <= 0.1,
      actual: { forward: r.forwardProportion, reverse: r.reverseProportion } },
    { id: '4.4', name: 'C 的两条分叉至少各 3 次，分叉比 1.5–9', ok: r.branchD >= 3 && r.branchG >= 3 && r.branchRatio >= 1.5 && r.branchRatio <= 9,
      actual: { D: r.branchD, G: r.branchG, ratio: r.branchRatio } },
    { id: '4.5', name: '停留均值 40–400 ms，CV 至少 0.3', ok: r.dwellMean >= 40 && r.dwellMean <= 400 && r.dwellCV >= 0.3,
      actual: { mean: r.dwellMean, CV: r.dwellCV } },
  ];
}

export function thresholdFor(S, lowNoise = false) {
  run(S, 500, { lowNoise });
  const rest = restMetrics(S, 2000, 10, lowNoise), values = rest.bins10.flat();
  const mu = mean(values), sigma = sd(values), threshold = Math.max(12, mu + 4 * sigma);
  delete rest.bins10;
  return { threshold, mu, sigma, rest, lowNoise };
}

const averageRanks = xs => {
  const sorted = xs.map((value, index) => ({ value, index })).sort((a, b) => a.value - b.value), ranks = Array(xs.length);
  for (let i = 0; i < sorted.length;) {
    let end = i + 1; while (end < sorted.length && sorted[end].value === sorted[i].value) end++;
    for (let j = i; j < end; j++) ranks[sorted[j].index] = (i + end - 1) / 2 + 1;
    i = end;
  }
  return ranks;
};
export function spearman(xs, ys) {
  const x = averageRanks(xs), y = averageRanks(ys), mx = mean(x), my = mean(y);
  const vx = mean(x.map(z => (z - mx) ** 2)), vy = mean(y.map(z => (z - my) ** 2));
  return vx && vy ? mean(x.map((z, i) => (z - mx) * (y[i] - my))) / Math.sqrt(vx * vy) : 0;
}

function directionMetrics(S, experienced) {
  const weights = weightMatrix(S), pairs = [];
  for (let a = 0; a < 8; a++) for (let z = 0; z < 8; z++) if (a !== z) pairs.push({ from: a, to: z, rho: weights.matrix[a][z], count: experienced[a][z] });
  const edges = EDGES.map(([a, z]) => ({ from: LABELS[a], to: LABELS[z], forward: weights.matrix[a][z], reverse: weights.matrix[z][a],
    difference: weights.matrix[a][z] - weights.matrix[z][a], count: experienced[a][z] }));
  const top7 = pairs.slice().sort((a, b) => b.rho - a.rho).slice(0, 7);
  return { ...weights, edges, meanDifference: mean(edges.map(x => x.difference)), meanAbsoluteDifference: mean(edges.map(x => Math.abs(x.difference))),
    spearman: spearman(pairs.map(x => x.rho), pairs.map(x => x.count)),
    top7, top7Experienced: top7.filter(p => EDGES.some(([a, z]) => a === p.from && z === p.to)).length,
    branchD: weights.matrix[2][3], branchG: weights.matrix[2][6], experienced, allPairs: pairs };
}

export function sequenceChecks(r) {
  return [
    { id: '2.1', name: '每条正向差至少 0.02，均差至少 0.05', ok: r.direction.edges.every(x => x.difference >= 0.02) && r.direction.meanDifference >= 0.05,
      actual: { edges: r.direction.edges, mean: r.direction.meanDifference } },
    { id: '2.2', name: '分叉效能 C→D 高于 C→G', ok: r.direction.branchD > r.direction.branchG, actual: { D: r.direction.branchD, G: r.direction.branchG } },
    { id: '2.3', name: '56 对 Spearman 至少 0.5，前 7 大至少 5 对有经历', ok: r.direction.spearman >= 0.5 && r.direction.top7Experienced >= 5,
      actual: { spearman: r.direction.spearman, top7Experienced: r.direction.top7Experienced } },
    { id: '2.4', name: '平均吸引子比至少 2，每组至少 1.8', ok: r.direction.meanRatio >= 2 && r.direction.ratios.every(x => x >= 1.8),
      actual: { mean: r.direction.meanRatio, groups: r.direction.ratios } },
    { id: '2.5', name: '学习态静息点燃率仍不超过 5%', ok: r.rest.ignitionRate <= 0.05, actual: r.rest.ignitionRate },
  ];
}
const reportChecks = checks => { for (const c of checks) console.log(`${c.ok ? '✓' : '✗'} ${c.id} ${c.name}: ${JSON.stringify(c.actual)}`); };
export const readResults = () => {
  const result = fs.existsSync(atPath('figures', 'chain-results.json')) ? jsonRead(atPath('figures', 'chain-results.json')) : null;
  return result?.schema === 2 ? result : { schema: 2, design: { labels: LABELS, edges: EDGES, config: CONFIG }, phases: {}, grid: [], controls: {}, checks: [] };
};
const updateResults = patch => { const result = { ...readResults(), ...patch }; jsonWrite(atPath('figures', 'chain-results.json'), result); return result; };

function variant(S, rng) {
  const g = rng.pick(S.cfg.K), proto = prototype(g, S.cfg), outside = S.E.filter(i => groupOf(i, S.cfg) !== g);
  const ids = [...rng.shuffle(proto.slice()).slice(0, 64), ...rng.shuffle(outside).slice(0, 16)];
  run(S, 200, { ids, learn: true }); run(S, 300, { learn: true }); clearSpikes(S);
  return g;
}

export function baseline(cfg) {
  const S = buildChain(cfg); run(S, 500);
  const r = restMetrics(S, 2000);
  delete r.bins10;
  console.log(`训练前 BG=${cfg.bg}, BG_I=${cfg.bgI}: E=${r.rE.toFixed(3)} Hz, I=${r.rI.toFixed(3)} Hz, Fano=${r.fano.toFixed(3)}`);
  return r;
}

export function trainAttractors(cfg = CONFIG, { baselineOnly = false, storeResults = true, allowFailedBaseline = false } = {}) {
  const file = cacheFile('attractors', cfg);
  if (fs.existsSync(file) && !baselineOnly) {
    const state = jsonRead(file);
    if (state.configHash !== hash(cfg)) throw new Error('已有吸引子缓存属于其他配置；请明确选择其配置，不能覆盖');
    if (storeResults) updateResults({ attractors: state.measurements, config: cfg,
      phases: { ...readResults().phases, attractors: state.measurements.checks.every(x => x.ok) ? 'passed' : 'failed' } });
    reportChecks(state.measurements.checks); return state;
  }
  const progressFile = cacheFile('attractor-progress', cfg);
  const progress = !baselineOnly && fs.existsSync(progressFile) ? jsonRead(progressFile) : null;
  if (progress && progress.state.configHash !== hash(cfg)) throw new Error('已有训练检查点配置不同，不能覆盖');
  const before = progress?.baseline ?? baseline(cfg);
  if (storeResults) updateResults({ config: cfg, baseline: before, baselineCandidates: [...(readResults().baselineCandidates ?? []), { cfg, ...before }] });
  const baselineOK = before.rE >= 2 && before.rE <= 5 && before.rI >= 5 && before.rI <= 20 && before.fano < 2;
  if (baselineOnly) return { baseline: before, ok: baselineOK };
  if (!baselineOK && !allowFailedBaseline) throw new Error('训练前标定未达标；仅允许在 BG=20–32、BG_I=8–16 内调整背景，尚未开始训练');
  if (!baselineOK && allowFailedBaseline) console.log('5e 原规则对照的基线不作门槛，记录后继续');
  const S = buildChain(cfg, progress?.state), rng = resumableRng(progress?.variantRng ?? 3);
  const counts = progress?.counts ?? Array(cfg.K).fill(0), t0 = Date.now(), elapsed = progress?.seconds ?? 0;
  if (progress) { restoreRuntime(S, progress.runtime); console.log(`精确恢复吸引子训练 ${progress.completed}/300`); }
  else run(S, 500);
  for (let n = progress?.completed ?? 0; n < 300; n++) {
    counts[variant(S, rng)]++;
    if (n % 5 === 4) {
      const seconds = elapsed + (Date.now() - t0) / 1000;
      jsonWrite(progressFile, { state: snapshot(S), runtime: captureRuntime(S), baseline: before, variantRng: rng.state(), counts, completed: n + 1, seconds });
      console.log(`吸引子变体 ${n + 1}/300，${seconds.toFixed(0)} s（检查点已保存）`);
    }
  }
  const state = snapshot(S);
  state.training = { variants: 300, groupCounts: counts, seconds: elapsed + (Date.now() - t0) / 1000 };
  const weights = weightMatrix(S), completion = completionMetrics(state);
  const restS = buildChain(cfg, state, { noiseSeed: 3001 }); run(restS, 500);
  const rest = restMetrics(restS); delete rest.bins10;
  const measurements = { baseline: before, weights, completion, rest, training: state.training };
  measurements.checks = attractorChecks(measurements);
  state.measurements = measurements; jsonWrite(file, state);
  if (storeResults) updateResults({ config: cfg, attractors: measurements, phases: { ...readResults().phases, attractors: measurements.checks.every(x => x.ok) ? 'passed' : 'failed' } });
  reportChecks(measurements.checks);
  return state;
}

export const LEARN_GAIN_GRID = [1, 0.9, 0.8, 0.7, 0.6, 0.5];
export function learningGainChecks(r) {
  return [
    { id: '1b.0', name: '完整六点学习态增益表', ok: r?.grid?.length === 6, actual: r?.grid?.length ?? 0 },
    { id: '1b.1', name: '最大可用学习增益：静息点燃 ≤ 5%，补全 ≥ +0.5 Hz 且 ≥ 7/8 为正',
      ok: !!r?.selected && r.selected.ok, actual: r?.selected ? { G: r.selected.G, ignition: r.selected.rest.ignitionRate, completion: r.selected.completion.meanDelta, positive: r.selected.completion.positiveGroups } : '未找到或未运行' },
  ];
}
export function learningGainCondition(attractors, G) {
  if (!LEARN_GAIN_GRID.includes(G)) throw new Error('学习增益不在计划网格中');
  if (!attractors.measurements.checks.every(c => c.ok)) throw new Error('S1 未通过，禁止标定学习增益');
  const file = cacheFile(`learn-g${G}`, attractors.config), stateHash = weightHash(attractors);
  if (fs.existsSync(file)) { const saved = jsonRead(file); if (saved.stateHash !== stateHash) throw new Error('学习增益缓存来源不符'); return saved; }
  const S = buildChain(attractors.config, attractors, { G, noiseSeed: 3001 }), t0 = Date.now();
  run(S, 500); const rest = restMetrics(S); delete rest.bins10;
  const completion = completionMetrics(attractors, { G });
  const r = { G, stateHash, rest, completion, seconds: (Date.now() - t0) / 1000 };
  r.ok = rest.ignitionRate <= 0.05 && completion.meanDelta >= 0.5 && completion.positiveGroups >= 7;
  jsonWrite(file, r); console.log(`学习增益 G=${G}: ${JSON.stringify({ ignition: rest.ignitionRate, completion: completion.meanDelta, positive: completion.positiveGroups, ok: r.ok })}`);
  return r;
}
export function collectLearningGain(attractors) {
  const grid = [];
  for (const G of LEARN_GAIN_GRID) {
    const file = cacheFile(`learn-g${G}`, attractors.config);
    if (fs.existsSync(file)) { const r = jsonRead(file); if (r.stateHash !== weightHash(attractors)) throw new Error('学习增益表来源不符'); grid.push(r); }
  }
  const selected = grid.length === 6 ? grid.find(r => r.ok) ?? null : null;
  const r = { grid, selected }; r.checks = learningGainChecks(r);
  updateResults({ learningGain: r, phases: { ...readResults().phases, learningGain: grid.length < 6 ? 'incomplete' : selected ? 'passed' : 'failed' } });
  return r;
}
export function ensureLearningGain(attractors) {
  for (const G of LEARN_GAIN_GRID) learningGainCondition(attractors, G);
  return collectLearningGain(attractors);
}

function trainSequence(attractors, { episodes = 160, itemMs = 40, shuffled = false, matchedPaths = null, learnGain = 1, allowFailedAttractors = false } = {}) {
  if (!allowFailedAttractors && !attractors.measurements.checks.every(x => x.ok)) throw new Error('步骤 1 未通过，禁止继续序列训练');
  const cfg = attractors.config, name = `${shuffled ? 'shuffled' : 'sequence'}-${episodes}-${itemMs}`;
  const file = cacheFile(name, cfg), sourceHash = hash({ weights: weightHash(attractors), learnGain });
  if (fs.existsSync(file)) {
    const cached = jsonRead(file);
    if (cached.sourceHash !== sourceHash || cached.configHash !== hash(cfg)) throw new Error('序列缓存起点不匹配');
    return cached;
  }
  const progressFile = cacheFile(`${name}-progress`, cfg);
  let progress = fs.existsSync(progressFile) ? jsonRead(progressFile) : null;
  // 320 回合补救从首轮 160 的完整动力学继续，等价于从头运行同一前缀。
  if (!progress && !shuffled && episodes === 320 && itemMs === 40) {
    const firstProgress = cacheFile('sequence-160-40-progress', cfg);
    if (fs.existsSync(firstProgress)) progress = jsonRead(firstProgress);
  }
  if (progress && (progress.sourceHash !== sourceHash || progress.state.configHash !== hash(cfg))) throw new Error('序列检查点起点不匹配');
  const S = buildChain(cfg, progress?.state ?? attractors, { G: learnGain, noiseSeed: 4001 });
  const worldRng = resumableRng(progress?.worldRng ?? 4002), variantRng = resumableRng(progress?.variantRng ?? 4003), reverseRng = resumableRng(progress?.reverseRng ?? 4004);
  const experienced = progress?.experienced ?? Array.from({ length: 8 }, () => Array(8).fill(0));
  const paths = progress?.paths ?? [], variantCounts = progress?.variantCounts ?? Array(8).fill(0);
  const t0 = Date.now(), elapsed = progress?.seconds ?? 0;
  if (progress) { restoreRuntime(S, progress.runtime); console.log(`恢复序列训练 ${progress.completed}/${episodes}，每项 ${itemMs} ms`); }
  else run(S, 500);
  for (let n = progress?.completed ?? 0; n < episodes; n++) {
    const original = matchedPaths ? matchedPaths[n].original.slice() : worldRng.uniform() < 0.75 ? [0, 1, 2, 3, 4, 5] : [0, 1, 2, 6, 7];
    const reversed = shuffled && reverseRng.uniform() < 0.5;
    const sequence = reversed ? original.slice().reverse() : original.slice();
    paths.push({ original, actual: sequence, reversed });
    for (let k = 1; k < sequence.length; k++) experienced[sequence[k - 1]][sequence[k]]++;
    for (const g of sequence) run(S, itemMs, { ids: prototype(g, cfg), hz: cfg.seqStim, learn: true });
    run(S, 400, { learn: true }); clearSpikes(S);
    if (n % 2 === 1) variantCounts[variant(S, variantRng)]++;
    if (n % 5 === 4 || n === episodes - 1) {
      const seconds = elapsed + (Date.now() - t0) / 1000;
      jsonWrite(progressFile, { state: snapshot(S), runtime: captureRuntime(S), sourceHash, experienced, paths, variantCounts,
        worldRng: worldRng.state(), variantRng: variantRng.state(), reverseRng: reverseRng.state(), completed: n + 1, seconds });
      console.log(`${shuffled ? '打乱' : '主'}序列 ${n + 1}/${episodes}，${itemMs} ms/项，${seconds.toFixed(0)} s`);
    }
  }
  const state = snapshot(S);
  state.sourceHash = sourceHash;
  state.training = { episodes, itemMs, shuffled, learnGain, paths, experienced, variantCounts, reversedEpisodes: paths.filter(p => p.reversed).length,
    seconds: elapsed + (Date.now() - t0) / 1000 };
  const restS = buildChain(cfg, state, { noiseSeed: 5001 }); run(restS, 500);
  const rest = restMetrics(restS); delete rest.bins10;
  const measurements = { direction: directionMetrics(S, experienced), rest, training: state.training };
  measurements.checks = sequenceChecks(measurements); state.measurements = measurements;
  jsonWrite(file, state); reportChecks(measurements.checks);
  return state;
}

export function ensureSequence(cfg = environmentConfig()) {
  const attractors = trainAttractors(cfg), mainFile = cacheFile('main', cfg);
  const gain = ensureLearningGain(attractors);
  if (!gain.selected) throw new Error('S1b 无可用学习增益，禁止序列学习');
  const learnGain = gain.selected.G;
  if (fs.existsSync(mainFile)) {
    const main = jsonRead(mainFile);
    if (main.configHash !== hash(cfg) || main.sourceHash !== hash({ weights: weightHash(attractors), learnGain })) throw new Error('主缓存与配置、吸引子起点或学习增益不符');
    const attempts = [[160, 40], [320, 40], [320, 30]].slice(0, main.remedies + 1).map(([episodes, itemMs]) => {
      const state = jsonRead(cacheFile(`sequence-${episodes}-${itemMs}`, cfg));
      return { episodes, itemMs, measurements: state.measurements };
    });
    updateResults({ sequence: main.measurements, sequenceAttempts: attempts,
      phases: { ...readResults().phases, sequence: main.measurements.checks.every(x => x.ok) ? 'passed' : 'failed' } });
    return main;
  }
  const attempts = [];
  for (const [episodes, itemMs] of [[160, 40], [320, 40], [320, 30]]) {
    const state = trainSequence(attractors, { episodes, itemMs, learnGain });
    attempts.push({ episodes, itemMs, measurements: state.measurements });
    const ok = state.measurements.checks.every(x => x.ok);
    updateResults({ sequenceAttempts: attempts, sequence: state.measurements,
      phases: { ...readResults().phases, sequence: ok ? 'passed' : attempts.length === 3 ? 'failed' : 'remedy' } });
    if (ok || attempts.length === 3) {
      state.remedies = attempts.length - 1; jsonWrite(mainFile, state);
      return state;
    }
    console.log(`序列验收未全部通过，启用计划允许的第 ${attempts.length} 次补救`);
  }
}

export function calibrationChecks(r) {
  return [
    { id: '3.1', name: '至少 80% 的终点 F 保持至少 80 ms', ok: r.holdRate >= 0.8, actual: r.holdRate },
    { id: '3.2', name: '至少 80% 的终点 F 在 1 s 内收尾', ok: r.endRate >= 0.8, actual: r.endRate },
    { id: '3.3', name: '激活群平均频率 15–80 Hz', ok: r.activeRate >= 15 && r.activeRate <= 80, actual: r.activeRate },
    { id: '3.4', name: '演化态静息点燃率不超过 10%', ok: r.rest.ignitionRate <= 0.1, actual: r.rest.ignitionRate },
  ];
}
export function calibrateCondition(main, G, b) {
  if (!main.measurements.checks.every(x => x.ok)) throw new Error('步骤 2 未通过，禁止标定');
  const cfg = main.config, file = cacheFile(`calib-g${G}-b${b}`, cfg), stateHash = weightHash(main);
  if (fs.existsSync(file)) {
    const r = jsonRead(file); if (r.stateHash !== stateHash) throw new Error('标定缓存与序列权重不匹配'); return r;
  }
  const S = buildChain(cfg, main, { G, b, noiseSeed: 6001 }), threshold = thresholdFor(S), trials = [], t0 = Date.now();
  for (let k = 0; k < 10; k++) {
    clearSpikes(S); const start = S.net.t;
    run(S, 40, { ids: prototype(5, cfg), hz: cfg.seqStim }); run(S, 3000);
    const bins = groupBins(S, start, 3040), decoded = decodeRates(bins, threshold.threshold);
    const endpoint = decoded.dwells.find(d => d.group === 5 && d.start <= 40);
    trials.push({ bins, decoded, holdMs: endpoint ? Math.max(0, endpoint.end - 40) : 0,
      endedIn1s: !!endpoint && endpoint.end - 40 <= 1000 });
    run(S, 2000); clearSpikes(S);
    console.log(`标定 G=${G}, b=${b}: ${k + 1}/10`);
  }
  const rest = restMetrics(S, 3000, threshold.threshold); delete rest.bins10;
  const r = { G, b, stateHash, threshold, holdRate: mean(trials.map(t => Number(t.holdMs >= 80))),
    endRate: mean(trials.map(t => Number(t.endedIn1s))), activeRate: mean(trials.flatMap(t => t.decoded.activeRates)),
    rest, trials, seconds: (Date.now() - t0) / 1000 };
  r.checks = calibrationChecks(r); r.ok = r.checks.every(x => x.ok); jsonWrite(file, r); reportChecks(r.checks);
  return r;
}
export const G_GRID = [0.6, 0.7, 0.8, 0.9, 1, 1.2];
export const B_GRID = [20, 40, 60, 80];
export function collectGrid(main) {
  const grid = [];
  for (const G of G_GRID) for (const b of B_GRID) {
    const file = cacheFile(`calib-g${G}-b${b}`, main.config);
    if (fs.existsSync(file)) { const r = jsonRead(file); if (r.stateHash !== weightHash(main)) throw new Error('网格来源不匹配'); grid.push(r); }
  }
  let workingPoint = null;
  if (grid.length === 24) {
    const ok = grid.filter(r => r.ok), minG = ok.length ? Math.min(...ok.map(r => r.G)) : null;
    const candidates = ok.filter(r => r.G === minG).sort((a, b) => a.b - b.b);
    if (candidates.length) workingPoint = candidates[Math.floor(candidates.length / 2)];
  }
  updateResults({ grid, workingPoint, phases: { ...readResults().phases,
    calibration: grid.length < 24 ? 'incomplete' : workingPoint ? 'passed' : 'failed' } });
  return { grid, workingPoint };
}

export function runTrials(state, { G, b, count = 40, lowNoise = false, enforceRest = false, name = 'main' }) {
  const cfg = state.config, file = cacheFile(`trials-${name}-g${G}-b${b}`, cfg), stateHash = weightHash(state);
  if (fs.existsSync(file)) {
    const cached = jsonRead(file);
    if (cached.stateHash !== stateHash || cached.lowNoise !== lowNoise || cached.enforceRest !== enforceRest) throw new Error('试次缓存条件不符');
    return cached;
  }
  const progressFile = cacheFile(`trials-${name}-g${G}-b${b}-progress`, cfg);
  const progress = fs.existsSync(progressFile) ? jsonRead(progressFile) : null;
  if (progress && progress.stateHash !== stateHash) throw new Error('试次检查点来源不符');
  const S = buildChain(cfg, state, { G, b, noiseSeed: 7001 });
  const threshold = progress?.threshold ?? thresholdFor(S, lowNoise);
  if (progress) restoreRuntime(S, progress.runtime);
  const trials = progress?.trials ?? [], rejected = progress?.rejected ?? [];
  let attempts = progress?.attempts ?? 0;
  const t0 = Date.now(), elapsed = progress?.seconds ?? 0;
  while (trials.length < count) {
    attempts++; clearSpikes(S); const start = S.net.t;
    run(S, 40, { ids: prototype(0, cfg), hz: cfg.seqStim, lowNoise }); run(S, 3000, { lowNoise });
    const bins = groupBins(S, start, 3040), decoded = decodeRates(bins, threshold.threshold);
    clearSpikes(S); const restStart = S.net.t; run(S, 2000, { lowNoise });
    const tail = groupBins(S, restStart + 1900, 100, 20);
    const returnedToRest = tail.every(row => row.every(r => r <= threshold.threshold));
    const trial = { attempt: attempts, bins, decoded, returnedToRest, restTail: tail };
    if (enforceRest && !returnedToRest) rejected.push(trial); else trials.push(trial);
    jsonWrite(progressFile, { stateHash, threshold, trials, rejected, attempts, runtime: captureRuntime(S), seconds: elapsed + (Date.now() - t0) / 1000 });
    console.log(`${name}: ${trials.length}/${count} 有效试次，作废 ${rejected.length}`);
  }
  const r = { stateHash, G, b, lowNoise, enforceRest, threshold, trials, rejected, attempts, finished: trials.length === count,
    summary: summarizeTrials(trials), seconds: elapsed + (Date.now() - t0) / 1000 };
  jsonWrite(file, r); return r;
}

export function mainTest(main) {
  const { workingPoint } = collectGrid(main);
  if (!workingPoint) throw new Error('没有通过完整网格的工作点，不能做主测试');
  const result = runTrials(main, { G: workingPoint.G, b: workingPoint.b, enforceRest: true });
  result.checks = [{ id: '4.0', name: '取得 40 个有效试次', ok: result.finished, actual: { valid: result.trials.length, rejected: result.rejected.length } }, ...chainChecks(result.summary)];
  updateResults({ mainTest: result, phases: { ...readResults().phases, chain: result.checks.every(x => x.ok) ? 'passed' : 'failed' } });
  reportChecks(result.checks); return result;
}

export function controlChecks(controls, mainSummary) {
  const noseq = controls.noSequence?.summary, shuffled = controls.shuffled?.summary, noadapt = controls.noAdaptation?.summary, low = controls.lowNoise?.summary;
  return [
    { id: '5.1', name: '没学序列：正向不超过 30%，链长中位数不超过 1', ok: !!noseq && noseq.forwardProportion <= 0.3 && noseq.chainMedian <= 1,
      actual: noseq ? { forward: noseq.forwardProportion, median: noseq.chainMedian } : '未运行' },
    { id: '5.2', name: '打乱顺序：方向绝对差不超过 0.02，正反向比例差不超过 15 个百分点',
      ok: !!shuffled && controls.shuffled.direction.meanAbsoluteDifference <= 0.02 && Math.abs(shuffled.forwardProportion - shuffled.reverseProportion) <= 0.15,
      actual: shuffled ? { meanAbsoluteDifference: controls.shuffled.direction.meanAbsoluteDifference, proportionDifference: Math.abs(shuffled.forwardProportion - shuffled.reverseProportion) } : '未运行' },
    { id: '5.3', name: '无适应：首状态停留中位数至少 1 s 或至少 50% 在 1 s 内无转移', ok: !!noadapt && (noadapt.firstDwellMedian >= 1000 || noadapt.noTransitionIn1s >= 0.5),
      actual: noadapt ? { firstDwellMedian: noadapt.firstDwellMedian, noTransitionIn1s: noadapt.noTransitionIn1s } : '未运行' },
    { id: '5.4', name: '低噪声：分叉比至少翻倍或转移总数减半',
      ok: !!low && ((Number.isFinite(mainSummary?.branchRatio) && ((Number.isFinite(low.branchRatio) && low.branchRatio >= 2 * mainSummary.branchRatio)
        || (low.branchG === 0 && low.branchD > 0)))
        || (mainSummary?.transitionCount > 0 && low.transitionCount <= 0.5 * mainSummary.transitionCount)),
      actual: low ? { lowBranchRatio: low.branchRatio, mainBranchRatio: mainSummary?.branchRatio, lowTransitions: low.transitionCount, mainTransitions: mainSummary?.transitionCount } : '未运行' },
    { id: '5.5', name: '原规则对照 5e 的两阶段与比较指标已记录（数值不作门槛）',
      ok: !!controls.originalRule?.attractors && !!controls.originalRule?.sequence && !!controls.originalRule?.comparison,
      actual: controls.originalRule?.comparison ?? '未运行' },
  ];
}

export function runControl(main, workingPoint, kind) {
  let result;
  const { G, b } = workingPoint;
  if (kind === 'noSequence') result = runTrials(jsonRead(cacheFile('attractors', main.config)), { G, b, name: 'no-sequence' });
  else if (kind === 'noAdaptation') result = runTrials(main, { G, b: 0, name: 'no-adaptation' });
  else if (kind === 'lowNoise') result = runTrials(main, { G, b, lowNoise: true, name: 'low-noise' });
  else if (kind === 'shuffled') {
    const attractors = jsonRead(cacheFile('attractors', main.config));
    const shuffled = trainSequence(attractors, { episodes: main.training.episodes, itemMs: main.training.itemMs, shuffled: true, matchedPaths: main.training.paths, learnGain: main.learnGain });
    jsonWrite(cacheFile('shuffled', main.config), shuffled);
    result = runTrials(shuffled, { G, b, name: 'shuffled' }); result.direction = shuffled.measurements.direction;
  } else if (kind === 'originalRule') {
    // 按 5e 的文字，两阶段训练都用主实验 S1b 选出的增益，不为此对照重新选参数。
    const cfg = { ...main.config, rule: 'GB_NMDA', initialGain: main.learnGain };
    const attractors = trainAttractors(cfg, { storeResults: false, allowFailedBaseline: true });
    const sequence = trainSequence(attractors, { episodes: main.training.episodes, itemMs: main.training.itemMs,
      matchedPaths: main.training.paths, learnGain: main.learnGain, allowFailedAttractors: true });
    jsonWrite(cacheFile('main', cfg), sequence);
    const rateAttractors = jsonRead(cacheFile('attractors', main.config));
    const compare = (state, base) => {
      const pairs = state.measurements.direction.allPairs.filter(p => p.count === 0);
      return { attractorRatio: base.measurements.weights.meanRatio, attractorGroupRatios: base.measurements.weights.ratios,
        sequenceRatio: state.measurements.direction.meanRatio, spearman: state.measurements.direction.spearman,
        unexperiencedPairs: pairs.length,
        unexperiencedDeltaFromAttractors: mean(pairs.map(p => p.rho - base.measurements.weights.matrix[p.from][p.to])),
        unexperiencedDeltaFromInitial: mean(pairs.map(p => p.rho - state.config.rho0)) };
    };
    result = { gain: main.learnGain, initialGain: cfg.initialGain, rule: 'GB_NMDA', gate: false, attractors: attractors.measurements,
      sequence: sequence.measurements, comparison: { GB_RATE: compare(main, rateAttractors), GB_NMDA: compare(sequence, attractors) } };
  } else throw new Error(`未知对照 ${kind}`);
  const file = cacheFile(`control-${kind}`, main.config);
  jsonWrite(file, { mainHash: weightHash(main), result });
  return result;
}

export function collectControls(main) {
  const controls = {};
  for (const kind of ['noSequence', 'shuffled', 'noAdaptation', 'lowNoise', 'originalRule']) {
    const file = cacheFile(`control-${kind}`, main.config);
    if (fs.existsSync(file)) { const saved = jsonRead(file); if (saved.mainHash !== weightHash(main)) throw new Error('对照来源不符'); controls[kind] = saved.result; }
  }
  const checks = controlChecks(controls, readResults().mainTest?.summary);
  updateResults({ controls, controlChecks: checks, phases: { ...readResults().phases,
    controls: Object.keys(controls).length < 5 ? 'incomplete' : checks.every(x => x.ok) ? 'passed' : 'failed' } });
  return { controls, checks };
}

export function acceptanceChecks(result = readResults()) {
  const missing = (ids, phase) => ids.map(id => ({ id, name: `${phase}未执行`, ok: false, actual: '未运行' }));
  return [
    ...(result.rulePrechecks?.R1 ? [{ id: 'R1', name: 'L3 配对与原规则验证', ok: result.rulePrechecks.R1.ok, actual: result.rulePrechecks.R1 }] : missing(['R1'], 'L3 规则检查')),
    ...(result.rulePrechecks?.R2 ? [{ id: 'R2', name: 'L4 逐拍对拍与 FAST 回归', ok: result.rulePrechecks.R2.ok && result.rulePrechecks.fast?.ok, actual: result.rulePrechecks.R2 }] : missing(['R2'], 'L4 规则检查')),
    ...(result.ruleAcceptance?.checks ?? missing(['R3.0', 'R3.1', 'R3.2', 'R3.3'], '网络规则验收')),
    ...(result.attractors ? attractorChecks(result.attractors) : missing(['1.1', '1.2', '1.3'], '吸引子阶段')),
    ...learningGainChecks(result.learningGain),
    ...(result.sequence ? sequenceChecks(result.sequence) : missing(['2.1', '2.2', '2.3', '2.4', '2.5'], '序列阶段')),
    { id: '2.6', name: '序列补救不超过两次并通过', ok: !!result.sequence && result.sequenceAttempts?.length <= 3 && sequenceChecks(result.sequence).every(x => x.ok), actual: result.sequenceAttempts?.length ?? '未运行' },
    ...(result.workingPoint ? calibrationChecks(result.workingPoint) : missing(['3.1', '3.2', '3.3', '3.4'], '演化标定')),
    { id: '3.0', name: '完整的 24 点 G×b 网格', ok: result.grid?.length === 24, actual: result.grid?.length ?? 0 },
    ...(result.mainTest ? [{ id: '4.0', name: '40 个有效线索试次', ok: result.mainTest.finished, actual: result.mainTest.trials.length }, ...chainChecks(result.mainTest.summary)]
      : missing(['4.0', '4.1', '4.2', '4.3', '4.4', '4.5'], '链条测试')),
    ...controlChecks(result.controls ?? {}, result.mainTest?.summary),
  ];
}

export function environmentConfig() {
  const saved = readResults().config ?? CONFIG;
  const cfg = { ...saved, bg: Number(process.env.BG ?? saved.bg), bgI: Number(process.env.BG_I ?? saved.bgI) };
  if (!Number.isFinite(cfg.bg) || !Number.isFinite(cfg.bgI) || cfg.bg < 20 || cfg.bg > 32 || cfg.bgI < 8 || cfg.bgI > 16) throw new Error('背景超出计划允许范围');
  return cfg;
}

export function drawFigure(result = readResults()) {
  const c = new Canvas(1440, 1060, [248, 250, 252]), ink = [25, 40, 55], muted = [90, 104, 118];
  const colors = [...PALETTE, [23, 170, 165]], actual = result.sequence?.direction ?? result.attractors?.weights;
  c.text(30, 22, 'DIRECTED ATTRACTOR CHAIN V2 - GB_RATE', ink, 3);
  c.text(30, 58, `P-IN ${result.config?.pIn ?? 0.1}  BG ${result.config?.bg ?? 28}  ${result.stopReason ? 'STOPPED' : 'MEASURED'}`, muted, 2);
  const panel = (x, y, title) => { c.rect(x, y, 675, 445, [255, 255, 255]); c.text(x + 18, y + 16, title, ink, 2); };
  panel(25, 100, 'A. LEARNED MEAN RHO'); panel(740, 100, 'B. DECODED STATES - 20 TRIALS');
  panel(25, 575, 'C. OBSERVED TRANSITIONS'); panel(740, 575, 'D. BRANCH COUNTS AND DWELL TIMES');
  const matrix = (values, x, y, markers = false, integer = false) => {
    const cell = 42, max = Math.max(1e-9, ...values.flat());
    c.text(x - 36, y - 25, 'FROM / TO', muted, 1);
    for (let a = 0; a < 8; a++) {
      c.text(x + a * cell + 14, y - 20, LABELS[a], ink, 2); c.text(x - 23, y + a * cell + 14, LABELS[a], ink, 2);
      for (let z = 0; z < 8; z++) {
        const value = values[a][z], px = x + z * cell, py = y + a * cell;
        c.rect(px, py, cell - 1, cell - 1, heat(value / max));
        c.text(px + 3, py + 16, integer ? String(value) : value.toFixed(2), value / max > 0.55 ? [255, 255, 255] : ink, 1);
        if (markers && EDGES.some(([i, j]) => a === i && z === j)) {
          c.line(px + 2, py + 2, px + cell - 4, py + 2, [255, 90, 0], 3);
          c.line(px + 2, py + 2, px + 2, py + cell - 4, [255, 90, 0], 3);
        }
      }
    }
    c.text(x + 365, y + 10, 'SCALE', muted, 2);
    for (let k = 0; k < 100; k++) c.rect(x + 366, y + 40 + k * 2, 22, 2, heat(1 - k / 99));
    c.text(x + 398, y + 40, integer ? max : max.toFixed(3), ink, 2); c.text(x + 398, y + 225, '0', ink, 2);
  };
  if (actual) {
    matrix(actual.matrix, 78, 166, true);
    c.text(78, 515, result.sequence ? 'AFTER SEQUENCE - ORANGE: EXPERIENCED EDGES' : 'AFTER VARIANTS - ORANGE: PLANNED EDGES - SEQUENCE NOT RUN', muted, 1);
  } else c.text(78, 270, 'NOT RUN', muted, 3);
  const trials = result.mainTest?.trials?.slice(0, 20), summary = result.mainTest?.summary;
  if (trials?.length) {
    const x = 790, y = 170, width = 584, step = 14;
    for (let k = 0; k < trials.length; k++) {
      c.text(x - 31, y + k * step, k + 1, muted, 1);
      const states = trials[k].decoded.states;
      states.forEach((g, t) => { if (g >= 0) c.rect(x + t / states.length * width, y + k * step, Math.ceil(width / states.length), 11, colors[g]); });
    }
    for (let s = 0; s <= 3; s++) c.text(x + s / 3.04 * width, y + 20 * step + 10, `${s}S`, ink, 1);
    for (let g = 0; g < 8; g++) { c.rect(x + g * 70, 495, 16, 14, colors[g]); c.text(x + g * 70 + 22, 495, LABELS[g], ink, 2); }
    matrix(summary.transitionMatrix, 78, 640, false, true);
    const dx = 800, base = 756, peak = Math.max(1, summary.branchD, summary.branchG);
    for (const [k, count] of [summary.branchD, summary.branchG].entries()) {
      const h = count / peak * 92; c.rect(dx + k * 215, base - h, 100, h, colors[k ? 6 : 3]);
      c.text(dx + k * 215, base + 10, k ? 'C > G' : 'C > D', ink, 2); c.text(dx + k * 215 + 114, base - 24, count, ink, 2);
    }
    const durations = summary.dwellDurations, counts = Array(10).fill(0);
    for (const ms of durations) counts[Math.min(9, Math.floor(ms / 100))]++;
    const hy = 944, maxCount = Math.max(1, ...counts);
    c.text(790, 807, 'DWELL HISTOGRAM - MS', ink, 2);
    counts.forEach((n, k) => { const h = n / maxCount * 90; c.rect(795 + k * 57, hy - h, 45, h, [31, 119, 180]); c.text(795 + k * 57, hy + 10, k === 9 ? '900+' : k * 100, muted, 1); });
    c.text(790, 986, `MEAN ${summary.dwellMean.toFixed(1)} MS  CV ${summary.dwellCV.toFixed(2)}`, muted, 1);
  } else {
    c.text(790, 270, 'NOT RUN - NO CHAIN TRIALS', muted, 2);
    c.text(78, 745, 'NOT RUN - NO TRANSITION DATA', muted, 2);
    c.text(790, 745, 'NOT RUN - NO BRANCH OR DWELL DATA', muted, 2);
  }
  fs.mkdirSync(atPath('figures'), { recursive: true }); c.save(atPath('figures', 'fig13-chain.png'));
  return atPath('figures', 'fig13-chain.png');
}

export function finalizeResults(stopReason = null) {
  const result = readResults();
  result.stopReason = stopReason; result.checks = acceptanceChecks(result);
  result.outcome = result.checks.every(c => c.ok) ? 'passed' : stopReason ? 'stopped' : 'partial';
  jsonWrite(atPath('figures', 'chain-results.json'), result); drawFigure(result); return result;
}

/** 无缓存时严格按阶段门槛重做；失败时只保存结果，不自行修改设计。 */
export function runExperiment(cfg = environmentConfig()) {
  const ruleFile = atPath('figures', 'rate-rule-results.json');
  if (!fs.existsSync(ruleFile)) execFileSync(process.execPath, [atPath('demo', 'rate-rule.mjs')], { cwd: ROOT, stdio: 'inherit', env: { ...process.env, MODE: 'all' } });
  const ruleAcceptance = jsonRead(ruleFile);
  updateResults({ ruleAcceptance, phases: { ...readResults().phases, rateRule: ruleAcceptance.ok ? 'passed' : 'failed' } });
  if (!ruleAcceptance.ok) return finalizeResults('R3 网络层规则验收未通过，禁止进入演化链');
  if (!fs.existsSync(cacheFile('attractors', cfg)) && !fs.existsSync(cacheFile('attractor-progress', cfg))) {
    const before = trainAttractors(cfg, { baselineOnly: true });
    // 初始 28 Hz 已实测偏低，30 Hz 是计划范围内的首个调整。
    if (!before.ok && before.baseline.rE < 2 && cfg.bg === 28) cfg = { ...cfg, bg: 30 };
  }
  const attractors = trainAttractors(cfg);
  if (!attractors.measurements.checks.every(c => c.ok)) return finalizeResults('S1 吸引子效能或补全未达标；按停止条件结束实验');
  const gain = ensureLearningGain(attractors);
  if (!gain.selected) return finalizeResults('S1b 六个条件中没有兼顾静息与补全的学习增益');
  const main = ensureSequence(cfg);
  if (!main.measurements.checks.every(c => c.ok)) return finalizeResults('S2 两次补救用尽后仍未达标');
  for (const G of G_GRID) for (const b of B_GRID) calibrateCondition(main, G, b);
  const { workingPoint } = collectGrid(main);
  if (!workingPoint) {
    return finalizeResults('S3 完整网格没有工作点；组内加密备选已在 v2 配置中使用');
  }
  mainTest(main);
  for (const kind of ['noSequence', 'shuffled', 'noAdaptation', 'lowNoise', 'originalRule']) runControl(main, workingPoint, kind);
  collectControls(main); return finalizeResults();
}

function main() {
  const mode = process.env.MODE ?? 'attractors', cfg = environmentConfig();
  if (mode === 'attractors') {
    const r = trainAttractors(cfg, { baselineOnly: process.env.BASELINE_ONLY === '1' });
    process.exitCode = r.measurements ? (r.measurements.checks.every(x => x.ok) ? 0 : 2) : (r.ok ? 0 : 2);
  } else if (mode === 'learn-gain') {
    const file = cacheFile('attractors', cfg);
    if (!fs.existsSync(file)) throw new Error('先完成 S1 的吸引子训练');
    const state = jsonRead(file);
    if (process.env.G && process.env.COLLECT_ONLY !== '1') learningGainCondition(state, Number(process.env.G));
    else {
      if (process.env.COLLECT_ONLY !== '1') for (const G of LEARN_GAIN_GRID) learningGainCondition(state, G);
      const r = collectLearningGain(state); reportChecks(r.checks); process.exitCode = r.selected ? 0 : 2;
    }
  } else if (mode === 'sequence') {
    const state = ensureSequence(cfg); process.exitCode = state.measurements.checks.every(x => x.ok) ? 0 : 2;
  } else if (mode === 'calib' || mode === 'test' || mode === 'controls') {
    // 并行工作进程只写各自缓存；完整网格与对照由单独的聚合调用写结果。
    const file = cacheFile('main', cfg);
    if (!fs.existsSync(file)) throw new Error('尚无主序列缓存，请先 MODE=sequence');
    const state = jsonRead(file);
    if (state.configHash !== hash(cfg) || !state.measurements.checks.every(x => x.ok)) throw new Error('序列尚未通过或配置不符');
    if (mode === 'calib') {
      const gs = process.env.G ? [Number(process.env.G)] : G_GRID, bs = process.env.B ? [Number(process.env.B)] : B_GRID;
      if (gs.some(g => !G_GRID.includes(g)) || bs.some(b => !B_GRID.includes(b))) throw new Error('条件不在计划的网格中');
      if (process.env.COLLECT_ONLY !== '1') for (const G of gs) for (const b of bs) calibrateCondition(state, G, b);
      if (!process.env.G && !process.env.B || process.env.COLLECT_ONLY === '1') collectGrid(state);
    } else if (mode === 'test') {
      const r = mainTest(state); process.exitCode = r.checks.every(c => c.ok) ? 0 : 2;
    } else {
      const workingPoint = readResults().workingPoint;
      if (!workingPoint) throw new Error('尚未聚合完整标定网格');
      if (process.env.COLLECT_ONLY !== '1') for (const kind of process.env.CONTROL ? [process.env.CONTROL] : ['noSequence', 'shuffled', 'noAdaptation', 'lowNoise', 'originalRule']) runControl(state, workingPoint, kind);
      if (!process.env.CONTROL || process.env.COLLECT_ONLY === '1') {
        const r = collectControls(state); process.exitCode = r.checks.every(c => c.ok) ? 0 : 2;
      }
    }
  } else if (mode === 'fig') {
    finalizeResults(readResults().stopReason ?? null);
  } else if (mode === 'all') {
    process.exitCode = runExperiment(cfg).outcome === 'passed' ? 0 : 2;
  } else throw new Error(`未知 MODE=${mode}`);
}
if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  try { main(); } catch (error) { console.error(error.stack); process.exitCode = 1; }
}
