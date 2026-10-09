// 只读诊断：复现 attempt/2/attractor-chain 的网络（不改仓库），检验"静息期自发活动的噪声学习"假设。
// node diag.mjs rest   —— 未训练网络，只有背景，学习开着 30 s，每 5 s 记一次效能
// node diag.mjs gated  —— 300 次变体，钙规则只在 200 ms 刺激期间学习，静息期冻结（抑制性可塑性照常）
import fs from 'fs';
import { SpikingNetwork } from 'file:///D:/kimi_kairos/bio-brain/src/L4-network/spiking-network.mjs';
import { GB_NMDA } from 'file:///D:/kimi_kairos/bio-brain/src/L3-plasticity/calcium-rule.mjs';
import { mammalianCosts } from 'file:///D:/kimi_kairos/bio-brain/src/L1-neuron/hh.mjs';
import { makeRng } from 'file:///D:/kimi_kairos/bio-brain/src/core/rng.mjs';
import { SpikingNetwork as SpikingNetworkRG } from './spiking-network-rg.mjs';

// 用法：node diag.mjs <mode> [base|rg] [eta] [post2Tau]；rg = 同时检测钙乘突触后慢速放电迹的规则变体
const [, , MODE, RULE = 'base', ETA = '0.5', TAU2 = '100', CPRE, CPOST] = process.argv;
const RG = RULE === 'rg';
const PLASTICITY = RG ? { ...GB_NMDA, eta: Number(ETA), post2Tau: Number(TAU2), ...(CPRE ? { Cpre: Number(CPRE) } : {}), ...(CPOST ? { Cpost: Number(CPOST) } : {}) } : GB_NMDA;
const NetClass = RG ? SpikingNetworkRG : SpikingNetwork;
const TAG = (RG ? `rg-e${ETA}-t${TAU2}${CPRE ? `-cpre${CPRE}` : ''}${CPOST ? `-cpost${CPOST}` : ''}` : 'base') + (process.env.PIN ? `-pin${process.env.PIN}-pout${process.env.POUT}` : '');

const cfg = { seed: 1, NE: 720, NI: 180, K: 8, M: 80, dt: 0.1, pIn: Number(process.env.PIN ?? 0.1), pOut: Number(process.env.POUT ?? 0.1), contacts: 10, sMin: 0.1, sMax: 9, rho0: 0.1, accel: 5, budget: 0.2,
  bg: 30, bgI: 12, stim: 52, b: 20, tauW: 300, pEI: 0.2, sEI: 1.5, pIE: 0.4, sIE: 4, pII: 0.4, sII: 4, eta: 0.02, target: 3, ieMax: 40 };
const mean = (xs) => xs.reduce((s, x) => s + x, 0) / (xs.length || 1);
const groupOf = (i) => (i < cfg.K * cfg.M ? Math.floor(i / cfg.M) : -1);
const prototype = (g) => Array.from({ length: cfg.M }, (_, k) => g * cfg.M + k);
const costs = mammalianCosts();

function resumableRng(initial = 1) {
  let s = initial >>> 0;
  const uniform = () => { s = (s + 0x6d2b79f5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  return { uniform, pick: (n) => Math.floor(uniform() * n), shuffle: (a) => { for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(uniform() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; } };
}

function build(noiseSeed) {
  const net = new NetClass({ costs, seed: cfg.seed, dt: cfg.dt, plasticity: PLASTICITY, pops: [{ name: 'E', n: cfg.NE, p: { b: cfg.b, tauW: cfg.tauW } }, { name: 'I', n: cfg.NI, p: { C: 100, tRef: 1 } }] });
  const pairs = [];
  for (let a = 0; a < cfg.NE; a++) for (let z = 0; z < cfg.NE; z++) {
    if (a === z) continue;
    const ga = groupOf(a), gz = groupOf(z);
    if (net.rng.uniform() < (ga >= 0 && ga === gz ? cfg.pIn : cfg.pOut)) pairs.push([a, z]);
  }
  const EE = net.connect({ pairs, contacts: cfg.contacts, pRelease: 0.3, tauRec: 100, plastic: { sMin: cfg.sMin, sMax: cfg.sMax, rho0: cfg.rho0, accel: cfg.accel, budget: cfg.budget } });
  net.connect({ from: 'E', to: 'I', p: cfg.pEI, contacts: 10, pRelease: 0.3, tauRec: 100, scale: cfg.sEI });
  const IE = net.connect({ from: 'I', to: 'E', p: cfg.pIE, contacts: 10, pRelease: 0.5, tauRec: 300, kind: 'inhFast', scale: cfg.sIE, istdp: { eta: cfg.eta, target: cfg.target, wMax: cfg.ieMax } });
  net.connect({ from: 'I', to: 'I', p: cfg.pII, contacts: 10, pRelease: 0.5, tauRec: 300, kind: 'inhFast', scale: cfg.sII });
  net.rng = resumableRng(noiseSeed);
  return { net, EE, IE, E: net.idx('E'), I: net.idx('I') };
}
const rateFn = (ids, hz) => { const r = new Float64Array(cfg.NE + cfg.NI); r.fill(cfg.bg, 0, cfg.NE); r.fill(cfg.bgI, cfg.NE); for (const i of ids) r[i] = hz; return (i) => r[i]; };
const run = (S, ms, { ids = [], hz = cfg.stim, learn = false } = {}) => S.net.run(ms, { rate: rateFn(ids, hz), learn });
const clearSpikes = (S) => { for (const s of S.net.spikes) s.length = 0; };
const freq = (S, ids, t0, ms) => (mean(ids.map((i) => S.net.spikes[i].filter((t) => t >= t0 && t < t0 + ms).length)) * 1000) / ms;

function rhoStats(S) {
  let w = 0, nw = 0, b = 0, nb = 0, u = 0, nu = 0;
  for (let k = 0; k < S.EE.rho.length; k++) {
    const a = groupOf(S.EE.pre[k]), z = groupOf(S.EE.post[k]), r = S.EE.rho[k];
    if (a >= 0 && z >= 0) { if (a === z) { w += r; nw++; } else { b += r; nb++; } } else { u += r; nu++; }
  }
  return { within: +(w / nw).toFixed(4), between: +(b / nb).toFixed(4), ungrouped: +(u / nu).toFixed(4), ratio: +((w / nw) / (b / nb)).toFixed(3), ie: +mean(Array.from(S.IE.scale)).toFixed(3) };
}
function groupRatios(S) {
  const sum = Array.from({ length: cfg.K }, () => Array(cfg.K).fill(0)), cnt = sum.map((r) => r.slice());
  for (let k = 0; k < S.EE.rho.length; k++) { const a = groupOf(S.EE.pre[k]), z = groupOf(S.EE.post[k]); if (a >= 0 && z >= 0) { sum[a][z] += S.EE.rho[k]; cnt[a][z]++; } }
  let os = 0, oc = 0; for (let a = 0; a < cfg.K; a++) for (let z = 0; z < cfg.K; z++) if (a !== z) { os += sum[a][z]; oc += cnt[a][z]; }
  const off = os / oc;
  return { off: +off.toFixed(4), ratios: sum.map((r, a) => +((r[a] / cnt[a][a]) / off).toFixed(3)) };
}
function cloneFrom(S0, noiseSeed) {
  const T = build(noiseSeed);
  T.EE.rho.set(S0.EE.rho); for (let k = 0; k < T.EE.rho.length; k++) T.EE.scale[k] = cfg.sMin + (cfg.sMax - cfg.sMin) * T.EE.rho[k];
  T.IE.scale.set(S0.IE.scale);
  return T;
}
function completion(S0) {
  const S = cloneFrom(S0, 2001), rng = makeRng(2002), rows = [];
  run(S, 500);
  for (let g = 0; g < cfg.K; g++) {
    const d = [];
    for (let rep = 0; rep < 6; rep++) {
      run(S, 300); clearSpikes(S);
      const proto = prototype(g), cue = rng.shuffle(proto.slice()).slice(0, 32), unseen = proto.filter((i) => !cue.includes(i)), outside = S.E.filter((i) => groupOf(i) !== g), t0 = S.net.t;
      run(S, 150, { ids: cue });
      d.push(freq(S, unseen, t0, 150) - freq(S, outside, t0, 150));
    }
    rows.push(+mean(d).toFixed(3));
  }
  return { perGroup: rows, meanDelta: +mean(rows).toFixed(3), positive: rows.filter((x) => x > 0).length };
}
function restIgnition(S0) {
  const S = cloneFrom(S0, 3001); run(S, 500); clearSpikes(S);
  const t0 = S.net.t, ms = 3000; run(S, ms);
  let hot = 0; const nb = ms / 20;
  for (let b = 0; b < nb; b++) if (Array.from({ length: cfg.K }, (_, g) => freq(S, prototype(g), t0 + b * 20, 20)).some((r) => r > 10)) hot++;
  return { rE: +freq(S, S.E, t0, ms).toFixed(2), rI: +freq(S, S.I, t0, ms).toFixed(2), ignition: +(hot / nb).toFixed(3) };
}

const mode = MODE, out = { mode, rule: TAG, plasticity: PLASTICITY, cfg };
const t00 = Date.now();
if (mode === 'rest') {
  const S = build(1001); run(S, 500);
  out.trace = [{ s: 0, ...rhoStats(S) }]; console.log('0 s', JSON.stringify(out.trace[0]));
  for (let k = 1; k <= 6; k++) {
    clearSpikes(S); const t0 = S.net.t; run(S, 5000, { learn: true });
    const row = { s: 5 * k, rE: +freq(S, S.E, t0, 5000).toFixed(2), rI: +freq(S, S.I, t0, 5000).toFixed(2), ...rhoStats(S) };
    out.trace.push(row); console.log(`${5 * k} s`, JSON.stringify(row), `${((Date.now() - t00) / 1000).toFixed(0)} s wall`);
  }
} else if (mode === 'gated' || mode === 'ungated') {
  const S = build(1001), rng = resumableRng(3); run(S, 500);
  const restFrozen = mode === 'gated';
  const counts = Array(cfg.K).fill(0);
  for (let n = 0; n < 300; n++) {
    const g = rng.pick(cfg.K), proto = prototype(g), outside = S.E.filter((i) => groupOf(i) !== g);
    const ids = [...rng.shuffle(proto.slice()).slice(0, 64), ...rng.shuffle(outside).slice(0, 16)];
    S.EE.frozen = false; run(S, 200, { ids, learn: true });
    if (restFrozen) { S.EE.frozen = true; S.EE.z.fill(0); }
    run(S, 300, { learn: true });
    clearSpikes(S); counts[g]++;
    if (n % 50 === 49) console.log(`variant ${n + 1}/300`, JSON.stringify(rhoStats(S)), `${((Date.now() - t00) / 1000).toFixed(0)} s wall`);
  }
  S.EE.frozen = false;
  out.counts = counts; out.stats = rhoStats(S); out.groups = groupRatios(S);
  console.log('weights', JSON.stringify(out.stats), JSON.stringify(out.groups));
  out.completion = completion(S); console.log('completion', JSON.stringify(out.completion));
  out.rest = restIgnition(S); console.log('rest', JSON.stringify(out.rest));
}
if (mode === 'classes') {
  // 只反复刺激 A 组（每次 64 个成员 + 16 个组外），看四类突触各涨多少；预算开 / 关各一次
  const classOf = (k, S) => { const a = groupOf(S.EE.pre[k]) === 0, z = groupOf(S.EE.post[k]) === 0; return a && z ? 'AtoA' : a ? 'AtoOut' : z ? 'OutToA' : 'OutToOut'; };
  for (const budgetOn of [true, false]) {
    const S = build(1001), rng = resumableRng(3);
    if (!budgetOn) S.EE.plastic.budget = undefined;
    run(S, 500);
    const rho0 = S.EE.rho.slice(), rates = { A: [], out: [] };
    for (let n = 0; n < 20; n++) {
      const proto = prototype(0), outside = S.E.filter((i) => groupOf(i) !== 0);
      const ids = [...rng.shuffle(proto.slice()).slice(0, 64), ...rng.shuffle(outside).slice(0, 16)], stim = new Set(ids);
      clearSpikes(S); const t0 = S.net.t;
      S.EE.frozen = false; run(S, 200, { ids, learn: true });
      rates.A.push(freq(S, proto.filter((i) => stim.has(i)), t0, 200)); rates.out.push(freq(S, outside.filter((i) => !stim.has(i)), t0, 200));
      S.EE.frozen = true; S.EE.z.fill(0); run(S, 300, { learn: true });
    }
    const acc = {};
    for (let k = 0; k < S.EE.rho.length; k++) { const c = classOf(k, S); (acc[c] ??= { d: 0, n: 0, up: 0 }); acc[c].d += S.EE.rho[k] - rho0[k]; acc[c].n++; if (S.EE.rho[k] > rho0[k] + 0.05) acc[c].up++; }
    const res = Object.fromEntries(Object.entries(acc).map(([c, v]) => [c, { meanDelta: +(v.d / v.n).toFixed(4), fracUp: +(v.up / v.n).toFixed(3), n: v.n }]));
    out[budgetOn ? 'budgetOn' : 'budgetOff'] = { classes: res, rateA: +mean(rates.A).toFixed(1), rateOutside: +mean(rates.out).toFixed(2) };
    console.log(budgetOn ? 'budget on ' : 'budget off', `A 被刺激成员 ${mean(rates.A).toFixed(1)} Hz，组外未刺激 ${mean(rates.out).toFixed(2)} Hz`, JSON.stringify(res), `${((Date.now() - t00) / 1000).toFixed(0)} s wall`);
  }
}
if (mode === 'direction') {
  // A 40 ms → B 40 ms → 休息 400 ms，40 遍（与演化链计划的序列强度相同，58 Hz），看 A→B 与 B→A
  const S = build(1001); run(S, 500);
  const rho0 = S.EE.rho.slice(), A = prototype(0), B = prototype(1);
  for (let n = 0; n < 40; n++) {
    run(S, 40, { ids: A, hz: 58, learn: true }); run(S, 40, { ids: B, hz: 58, learn: true }); run(S, 400, { learn: true });
    clearSpikes(S);
  }
  const cls = (k) => {
    const a = groupOf(S.EE.pre[k]), z = groupOf(S.EE.post[k]), inA = (g) => g === 0 || g === 1;
    if (a === 0 && z === 1) return 'AtoB'; if (a === 1 && z === 0) return 'BtoA'; if (a === 0 && z === 0) return 'AtoA'; if (a === 1 && z === 1) return 'BtoB';
    return inA(a) && !inA(z) ? 'ABtoOut' : !inA(a) && inA(z) ? 'OutToAB' : 'OutToOut';
  };
  const acc = {};
  for (let k = 0; k < S.EE.rho.length; k++) { const c = cls(k); (acc[c] ??= { d: 0, n: 0 }); acc[c].d += S.EE.rho[k] - rho0[k]; acc[c].n++; }
  out.direction = Object.fromEntries(Object.entries(acc).map(([c, v]) => [c, +(v.d / v.n).toFixed(4)]));
  console.log('direction', JSON.stringify(out.direction));
}
out.seconds = (Date.now() - t00) / 1000;
fs.writeFileSync(new URL(`./diag-${mode}-${TAG}.json`, import.meta.url), JSON.stringify(out, null, 1));
console.log(`done in ${out.seconds.toFixed(0)} s`);
