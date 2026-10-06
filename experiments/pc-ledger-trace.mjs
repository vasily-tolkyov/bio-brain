// 「X 的中继 → Y 的误差」抑制的增减账：默认 20 ms 配对窗口 vs 增强一侧用长的突触前痕迹（TAUPRE ms）
import fs from 'fs';
import { PC2, grp } from '../demo/predictive-hierarchy.mjs';
import { STIM, protoB } from '../demo/balanced-layer.mjs';
import { buildLayered, loadState, learnOnly, inputRate } from '../src/L4-network/layered-network.mjs';
import { mean } from '../demo/lib-hierarchy.mjs';

const TAUPRE = Number(process.env.TAUPRE ?? 0), log = console.log;
const S = buildLayered(2, { ...PC2, pcOver: { ...PC2.pcOver, ...(TAUPRE ? { predTauPre: TAUPRE } : {}) } });
loadState(S, JSON.parse(fs.readFileSync('cache/pc-2.json', 'utf8')).state); learnOnly(S, []);
const [L1] = S.layers, g = L1.PRED;
const set = (ids) => new Set(ids);
const sX = set([...grp(S, 0, 'S', 'A'), ...grp(S, 0, 'S', 'D')]), sY = set([...grp(S, 0, 'S', 'C'), ...grp(S, 0, 'S', 'B')]);
const nY = set([...grp(S, 0, 'N', 'C'), ...grp(S, 0, 'N', 'B')]), nX = set([...grp(S, 0, 'N', 'A'), ...grp(S, 0, 'N', 'D')]);
const idx = (preSet, postSet) => { const k = []; for (let s = 0; s < g.pre.length; s++) if (preSet.has(g.pre[s]) && postSet.has(g.post[s])) k.push(s); return k; };
const K = { XY: idx(sX, nY), XX: idx(sX, nX), YX: idx(sY, nX), YY: idx(sY, nY) }, avgW = (ks) => mean(ks.map((s) => g.scale[s]));
const at = (parts) => new Map([...parts].flatMap((p) => protoB(p).map((a) => [a, STIM])));
function ledger(segments, reps = 12) {
  const keep = g.scale.slice(), acc = segments.map(() => ({ XY: 0, XX: 0, YX: 0, YY: 0 }));
  learnOnly(S, [g]);
  for (let r = 0; r < reps; r++) {
    S.net.run(300, { rate: inputRate(S), learn: false });
    segments.forEach(([parts, ms], i) => {
      const before = Object.fromEntries(Object.entries(K).map(([n, ks]) => [n, avgW(ks)]));
      S.net.run(ms, { rate: inputRate(S, at(parts)), learn: true });
      for (const n of Object.keys(K)) acc[i][n] += (avgW(K[n]) - before[n]) / reps;
    });
  }
  g.scale.set(keep); learnOnly(S, []);
  return acc;
}
const fmt = (v) => ((v >= 0 ? '+' : '') + (v * 1000).toFixed(1)).padStart(7);
log(`突触前痕迹 ${TAUPRE || 20} ms（×10⁻³ / 次；当前强度 X→Y误差 ${avgW(K.XY).toFixed(3)}，X→X误差 ${avgW(K.XX).toFixed(3)}，Y→X误差 ${avgW(K.YX).toFixed(3)}，Y→Y误差 ${avgW(K.YY).toFixed(3)}）`);
log('                 X中继→Y误差  X中继→X误差  Y中继→X误差  Y中继→Y误差');
const show = (title, segs) => {
  const a = ledger(segs.map(([p, ms]) => [p, ms]));
  log(`  ${title}`);
  segs.forEach(([, , lab], i) => log(`    ${lab.padEnd(12)} ${fmt(a[i].XY)}      ${fmt(a[i].XX)}      ${fmt(a[i].YX)}      ${fmt(a[i].YY)}`));
  const t = (n) => a.reduce((s, v) => s + v[n], 0);
  log(`    ${'合计'.padEnd(12)} ${fmt(t('XY'))}      ${fmt(t('XX'))}      ${fmt(t('YX'))}      ${fmt(t('YY'))}`);
};
show('序列 X → Y：', [['AD', 150, 'X 0–150 ms'], ['CB', 30, 'Y 0–30 ms'], ['CB', 120, 'Y 30–150 ms'], ['', 300, '休息 300 ms']]);
show('单独的物体 X：', [['AD', 200, 'X 0–200 ms'], ['', 300, '休息 300 ms']]);
