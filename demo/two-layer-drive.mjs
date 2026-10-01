/** 第二层（全新网络）在不同的驱动方式下，组内 / 组间突触效能怎么长：画得多强决定绑得多紧。
 * SCHEME=graded（默认）：按第一层激活度绘图（X: A 0.65 B 0.13 C 0.42 D 0.11；Y: A 0.02 B 0.65 C 0.32 D 0.40，取自 fig8 的实测平均）
 * SCHEME=binary：物体的两个部件满强度驱动，其余背景
 * SCHEME=l1：第一层的变体训练方式（每次一组的 16 个成员），作为"这个网络本身能学"的对照
 * 用法：node demo/two-layer-drive.mjs（约 80 秒），每学 25 次打印一次 */
import { buildNetwork, CFG as cfg } from './fig2-network.mjs';
import { makeRng } from '../src/core/rng.mjs';

const scheme = process.env.SCHEME ?? 'graded', R_MAX = 30;
const { NE, M } = cfg;
const G = [0, 1, 2, 3].map((a) => Array.from({ length: M }, (_, i) => a * M + i));
const gOf = (i) => (i < 4 * M ? Math.floor(i / M) : -1);
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
  for (let k = 0; k < EE.pre.length; k++) {
    if (EE.pre[k] >= NE || EE.post[k] >= NE) continue;
    const a = gOf(EE.pre[k]), b = gOf(EE.post[k]);
    if (a >= 0 && b >= 0) { s[a][b] += EE.rho[k]; n[a][b]++; } else { other += EE.rho[k]; no++; }
  }
  const m = (a, b) => (s[a][b] / n[a][b]).toFixed(2);
  return `组内 A ${m(0, 0)} B ${m(1, 1)} C ${m(2, 2)} D ${m(3, 3)} | A→C ${m(0, 2)} C→A ${m(2, 0)} | B→D ${m(1, 3)} D→B ${m(3, 1)} | A→B ${m(0, 1)} B→A ${m(1, 0)} | 其他 ${(other / no).toFixed(2)} | 抑制 ${(ie.scale.reduce((x, y) => x + y, 0) / ie.scale.length).toFixed(2)}`;
};
const bg = (i) => (i < NE ? cfg.bg : 0);
net.run(500, { rate: bg, learn: false });
console.log(`[${scheme}] 学习前：${stats()}`);
let fire = [0, 0, 0, 0], nf = 0;
for (let n = 0; n < 100; n++) {
  const d = drives(), t0 = net.t;
  net.run(200, { rate: (i) => (i < NE ? d.get(i) ?? cfg.bg : 0), learn: true });
  const c = net.counts(t0, net.t, [...Array(4 * M).keys()]);
  G.forEach((g, a) => (fire[a] += g.reduce((x, i) => x + c[i], 0) / g.length / 0.2)); nf++;
  net.run(300, { rate: bg, learn: true });
  if (n % 25 === 24) { console.log(`[${scheme}] 学习 ${n + 1} 次：${stats()}  刺激期各组平均放电 ${fire.map((f) => (f / nf).toFixed(1)).join(' / ')} Hz`); fire = [0, 0, 0, 0]; nf = 0; }
}
