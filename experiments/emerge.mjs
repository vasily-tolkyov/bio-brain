import { SpikingNetwork } from '../src/L4-network/spiking-network.mjs';
import { mammalianCosts } from '../src/L1-neuron/hh.mjs';
import { makeRng } from '../src/core/rng.mjs';
import { GB_DP, GB_NMDA } from '../src/L3-plasticity/calcium-rule.mjs';

const env = (k, d) => Number(process.env[k] ?? d);
const costs = mammalianCosts();
const NE = 100, NI = 25, K = 4, M = 20;
const gb = process.env.RULE === 'DP' ? GB_DP : { ...GB_NMDA, eta: env('ETA_N', GB_NMDA.eta), Cpost: env('CPOST', GB_NMDA.Cpost), gammaP: env('GAMMAP', GB_NMDA.gammaP) };
const net = new SpikingNetwork({ costs, seed: 1, plasticity: gb, pops: [{ name: 'E', n: NE, p: { b: env('ADB', 20), tauW: 300 } }, { name: 'I', n: NI, p: { C: 100, tRef: 1 } }] });
const EE = net.connect({ from: 'E', to: 'E', p: env('PCONN', 0.5), contacts: env('CONTACTS', 10), pRelease: 0.3, tauRec: 100, plastic: { sMin: 0.1, sMax: env('SMAX', 6), rho0: 0.1, accel: 5, ...(process.env.BUDGET ? { budget: env('BUDGET', 0.25) } : {}) } });
net.connect({ from: 'E', to: 'I', p: 0.5, contacts: 10, pRelease: 0.3, scale: env('SEI', 1.5) });
const IE = net.connect({ from: 'I', to: 'E', p: 0.5, contacts: 10, pRelease: 0.5, tauRec: 300, kind: 'inh', scale: env('SIE', 1), istdp: env('ETA', 0) > 0 ? { eta: env('ETA', 0), target: env('TARGET', 5) } : null });
const rng = makeRng(3);
const proto = Array.from({ length: K }, (_, k) => Array.from({ length: M }, (_, i) => k * M + i));
const variant = (k) => { const members = rng.shuffle(proto[k].slice()).slice(0, 16); const others = rng.shuffle([...Array(NE).keys()].filter((i) => !proto[k].includes(i))).slice(0, 4); return new Set([...members, ...others]); };
const BG = env('BG', 2), STIM = env('STIM', 30);
const present = (set, ms, learn, stimRate = STIM) => net.run(ms, { rate: (i) => (i < NE ? (set.has(i) ? stimRate : BG) : 0), learn });
const quiet = (ms, learn) => net.run(ms, { rate: (i) => (i < NE ? BG : 0), learn });
function blockStats() {
  const g = EE; const s = { in: [0, 0], out: [0, 0] };
  const pk = (i) => (i < K * M ? Math.floor(i / M) : -1);
  const up = { in: [0, 0], out: [0, 0] };
  for (let k = 0; k < g.pre.length; k++) { const a = pk(g.pre[k]), b = pk(g.post[k]); const key = a >= 0 && a === b ? 'in' : 'out'; s[key][0] += g.scale[k]; s[key][1]++; up[key][0] += g.rho[k] > 0.5 ? 1 : 0; up[key][1]++; }
  return { within: s.in[0] / s.in[1], between: s.out[0] / s.out[1], upIn: up.in[0] / up.in[1], upOut: up.out[0] / up.out[1] };
}
function test(label) {
  const res = [];
  for (let k = 0; k < K; k++) for (let r = 0; r < 3; r++) {
    const v = variant(k), t0 = net.t;
    present(v, 200, false);
    const cnt = net.counts(t0, net.t, [...Array(NE).keys()]);
    quiet(300, false);
    const unstim = proto[k].filter((i) => !v.has(i)), noise = [...v].filter((i) => !proto[k].includes(i));
    res.push({ k, cnt, comp: unstim.filter((i) => cnt[i] >= 2).length / unstim.length, noiseRate: noise.reduce((s, i) => s + cnt[i], 0) / noise.length / 0.2, memRate: [...v].filter((i) => proto[k].includes(i)).reduce((s, i) => s + cnt[i], 0) / 16 / 0.2, others: cnt.reduce((s, c, i) => s + (v.has(i) || proto[k].includes(i) ? 0 : c), 0) });
  }
  const corr = (a, b) => { const n = a.length, ma = a.reduce((s, v) => s + v, 0) / n, mb = b.reduce((s, v) => s + v, 0) / n; let x = 0, y = 0, z = 0; for (let i = 0; i < n; i++) { x += (a[i] - ma) * (b[i] - mb); y += (a[i] - ma) ** 2; z += (b[i] - mb) ** 2; } return x / Math.sqrt(y * z || 1); };
  let wi = [], bw = [];
  for (let a = 0; a < res.length; a++) for (let b = a + 1; b < res.length; b++) (res[a].k === res[b].k ? wi : bw).push(corr(res[a].cnt, res[b].cnt));
  const m = (x) => x.reduce((s, v) => s + v, 0) / x.length;
  console.log(`${label}: 同组相关 ${m(wi).toFixed(2)} 异组 ${m(bw).toFixed(2)} | 未受刺激的原型成员被拉起 ${m(res.map((r) => r.comp)).toFixed(2)} | 成员 ${m(res.map((r) => r.memRate)).toFixed(0)} Hz 噪声神经元 ${m(res.map((r) => r.noiseRate)).toFixed(0)} Hz 其余总放电 ${m(res.map((r) => r.others)).toFixed(1)} | 突触 组内 ${blockStats().within.toFixed(2)} 组间 ${blockStats().between.toFixed(2)}`);
}
const t0w = Date.now();
quiet(500, false);
test('学习前');
for (let n = 0; n < env('NPRES', 100); n++) {
  const ts = net.t;
  present(variant(rng.pick(K)), 200, true); quiet(env('GAP', 300), true);
  if (n % 25 === 24) { const c = net.counts(ts, net.t, [...Array(NE).keys()]); const b = blockStats(); console.log(`  第 ${n + 1} 次呈现：E 平均 ${(c.reduce((s, v) => s + v, 0) / NE / 0.5).toFixed(1)} Hz，突触 组内 ${b.within.toFixed(2)}（强态 ${(b.upIn * 100).toFixed(0)}%） 组间 ${b.between.toFixed(2)}（强态 ${(b.upOut * 100).toFixed(0)}%），抑制均值 ${(IE.scale.reduce((s, v) => s + v, 0) / IE.scale.length).toFixed(2)}`); }
}
test('学习后');
console.log(`墙钟 ${((Date.now() - t0w) / 1000).toFixed(0)} s，模拟 ${(net.t / 1000).toFixed(0)} s，ATP 分布`, Object.entries(net.ledger.shares()).map(([k, v]) => k + ' ' + (v * 100).toFixed(0) + '%').join(' '));
