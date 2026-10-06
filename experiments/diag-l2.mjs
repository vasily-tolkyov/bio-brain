// 诊断：第二层（全新网络 seed 2）在不同驱动方式下组内 / 组间突触效能怎么变
// SCHEME=graded（按激活度绘图）| binary（物体两部件满强度）| l1（第一层的变体训练方式）
import { buildNetwork, CFG as cfg } from '../demo/fig2-network.mjs';
import { makeRng } from '../src/core/rng.mjs';

const scheme = process.env.SCHEME ?? 'graded', R_MAX = Number(process.env.RMAX ?? 30);
const { NE, M } = cfg;
const G = [0, 1, 2, 3].map((a) => Array.from({ length: M }, (_, i) => a * M + i));
const gOf = (i) => (i < 80 ? Math.floor(i / M) : -1);
const { net, EE } = buildNetwork({ ...cfg, seed: 2 });
const ie = net.groups.find((g) => g.istdp);
const rng = makeRng(31), choice = (arr) => arr[rng.pick(arr.length)];
const LV = { X: [0.65, 0.13, 0.42, 0.11], Y: [0.02, 0.65, 0.32, 0.40] };
const PARTS = { X: [0, 2], Y: [1, 3] };
function drives() {
  if (scheme === 'l1') { const k = rng.pick(4); return new Map(rng.shuffle(G[k].slice()).slice(0, 16).map((i) => [i, cfg.stim])); }
  const o = choice(['X', 'Y']);
  if (scheme === 'binary') return new Map(PARTS[o].flatMap((a) => G[a].map((i) => [i, cfg.stim])));
  return new Map(G.flatMap((g, a) => g.map((i) => [i, cfg.bg + LV[o][a] * (R_MAX - cfg.bg)])));
}
const stats = () => {
  const s = Array.from({ length: 4 }, () => new Float64Array(4)), n = Array.from({ length: 4 }, () => new Float64Array(4));
  let other = 0, no = 0;
  for (let k = 0; k < EE.pre.length; k++) { const a = gOf(EE.pre[k]), b = gOf(EE.post[k]); if (EE.pre[k] >= NE || EE.post[k] >= NE) continue; if (a >= 0 && b >= 0) { s[a][b] += EE.rho[k]; n[a][b]++; } else { other += EE.rho[k]; no++; } }
  const m = (a, b) => (s[a][b] / n[a][b]).toFixed(2);
  return `组内 A ${m(0, 0)} B ${m(1, 1)} C ${m(2, 2)} D ${m(3, 3)} | A↔C ${m(0, 2)}/${m(2, 0)} B↔D ${m(1, 3)}/${m(3, 1)} A↔B ${m(0, 1)}/${m(1, 0)} | 其他 ${(other / no).toFixed(2)} | 抑制 ${(ie.scale.reduce((x, y) => x + y, 0) / ie.scale.length).toFixed(2)}`;
};
const bg = (i) => (i < NE ? cfg.bg : 0);
net.run(500, { rate: bg, learn: false });
console.log(`[${scheme}] 0: ${stats()}`);
let fire = [0, 0, 0, 0], nf = 0;
for (let n = 0; n < 100; n++) {
  const d = drives(), t0 = net.t;
  net.run(200, { rate: (i) => (i < NE ? d.get(i) ?? cfg.bg : 0), learn: true });
  const c = net.counts(t0, net.t, [...Array(80).keys()]);
  G.forEach((g, a) => (fire[a] += g.reduce((x, i) => x + c[i], 0) / g.length / 0.2)); nf++;
  net.run(300, { rate: bg, learn: true });
  if (n % 25 === 24) { console.log(`[${scheme}] ${n + 1}: ${stats()}  刺激期各组平均放电 ${fire.map((f) => (f / nf).toFixed(1)).join('/')} Hz`); fire = [0, 0, 0, 0]; nf = 0; }
}
