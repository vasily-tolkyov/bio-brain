/** 闭环证据层 E1：从 v2 提取的吸引子构建、刺激期学习和精确续算工具。 */
import crypto from 'node:crypto';
import { SpikingNetwork } from '../L4-network/spiking-network.mjs';
import { GB_NMDA, GB_RATE } from '../L3-plasticity/calcium-rule.mjs';
import { mammalianCosts } from '../L1-neuron/hh.mjs';
export const E1_DEFAULTS = Object.freeze({
  version: 2, rule: 'GB_RATE', initialGain: 1, seed: 1, NE: 720, NI: 180, K: 8, M: 80, dt: 0.1,
  pIn: 0.25, pOut: 0.083, contacts: 10, sMin: 0.1, sMax: 9, rho0: 0.1,
  accel: 5, budget: 0.2, bg: 30, bgI: 12, stim: 52, seqStim: 58,
  b: 20, tauW: 300, pEI: 0.2, sEI: 1.5, pIE: 0.4, sIE: 4,
  pII: 0.4, sII: 4, eta: 0.02, target: 3, ieMax: 40,
});

export const mean = xs => xs.reduce((s, x) => s + x, 0) / (xs.length || 1);

const hash = value => crypto.createHash('sha256').update(JSON.stringify(value)).digest('hex');
export const RULES = { GB_RATE, GB_NMDA, GB_NMDA_ETA0: { ...GB_NMDA, eta: 0 } };
export const groupOf = (i, cfg) => i < cfg.K * cfg.M ? Math.floor(i / cfg.M) : -1;
export const prototype = (g, cfg) => Array.from({ length: cfg.M }, (_, k) => g * cfg.M + k);
let costs;

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

/** 默认关闭，供闭环构建器共享第 4.6 节的 v2 入度换算。 */
export function e1Configuration(cfg = E1_DEFAULTS, inDegree = null) {
  if (inDegree === null || inDegree === false) return cfg;
  if (inDegree !== 'v2' || cfg.M !== 80) throw new Error('E1 入度选项须为 v2，且每组为 80 个神经元');
  const NE = Math.max(720, 80 * cfg.K + 80), NI = NE / 4;
  return { ...cfg, NE, NI, pIn: 0.25, pOut: 0.083 * (640 / (NE - 80)),
    pEI: 0.2 * (720 / NE), pIE: 0.4 * (180 / NI), pII: 0.4 * (180 / NI) };
}

export function buildE1(cfg = E1_DEFAULTS, state = null, { G = state?.learnGain ?? cfg.initialGain ?? 1, b = 20, noiseSeed = 1001, inDegree = null } = {}) {
  cfg = e1Configuration(cfg, inDegree);
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

export function captureRuntime(S) {
  return { ...S.net.exportState(), e1: { G: S.G, b: S.b } };
}
export function restoreRuntime(S, runtime) {
  S.net.importState(runtime);
  if (runtime.e1) { S.G = runtime.e1.G; S.b = runtime.e1.b; }
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

export function variant(S, rng) {
  const g = rng.pick(S.cfg.K), proto = prototype(g, S.cfg), outside = S.E.filter(i => groupOf(i, S.cfg) !== g);
  const ids = [...rng.shuffle(proto.slice()).slice(0, 64), ...rng.shuffle(outside).slice(0, 16)];
  run(S, 200, { ids, learn: true }); run(S, 300, { learn: true }); clearSpikes(S);
  return g;
}

