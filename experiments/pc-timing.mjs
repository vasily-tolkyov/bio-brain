// 时间对不上的机理：Y 出现前后各群的细时间进程 + 「X 的中继 → Y 的误差细胞」抑制在各阶段的增减账
import { pcStack, sequenceCourse, grp } from '../demo/predictive-hierarchy.mjs';
import { STIM, protoB } from '../demo/balanced-layer.mjs';
import { learnOnly, inputRate } from '../src/L4-network/layered-network.mjs';
import { mean } from '../demo/lib-hierarchy.mjs';

const log = console.log, row = (a) => [...a].map((v) => v.toFixed(1).padStart(5)).join('');
const S = pcStack(), [L1] = S.layers;

// ① 细时间进程：X 150 ms → Y 150 ms，Y 出现前 30 ms 到后 70 ms，每 5 ms
const tc = sequenceCourse(S, [['AD', 150], ['CB', 150], ['', 200]], { bin: 5, reps: 16 });
const pr = (p, q) => [...tc[p]].map((v, b) => (v + tc[q][b]) / 2).slice(24, 44);
log('Y 出现前 30 ms 到后 70 ms，每 5 ms（第 7 列 = Y 的头 5 ms）：');
log(`  上层 X 的组（E2 A、D）       ${row(pr('E2A', 'E2D'))}`);
log(`  上层 Y 的组（E2 C、B）       ${row(pr('E2C', 'E2B'))}`);
log(`  X 的中继（S1 A、D）           ${row(pr('S1A', 'S1D'))}`);
log(`  Y 的中继（S1 C、B）           ${row(pr('S1C', 'S1B'))}`);
log(`  第一层 Y 的表征（E1 C、B）   ${row(pr('E1C', 'E1B'))}`);
log(`  第一层 Y 的误差（N1 C、B）   ${row(pr('N1C', 'N1B'))}`);
log(`  第一层 X 的误差（N1 A、D）   ${row(pr('N1A', 'N1D'))}`);

// ② 增减账：只让第一层的预测抑制学习，记录各阶段「X 的中继 → Y 的误差」与「X 的中继 → X 的误差」的平均强度变化
const set = (ids) => new Set(ids);
const sX = set([...grp(S, 0, 'S', 'A'), ...grp(S, 0, 'S', 'D')]), nY = set([...grp(S, 0, 'N', 'C'), ...grp(S, 0, 'N', 'B')]), nX = set([...grp(S, 0, 'N', 'A'), ...grp(S, 0, 'N', 'D')]);
const g = L1.PRED, idx = (post) => { const k = []; for (let s = 0; s < g.pre.length; s++) if (sX.has(g.pre[s]) && post.has(g.post[s])) k.push(s); return k; };
const kXY = idx(nY), kXX = idx(nX), avgW = (ks) => mean(ks.map((s) => g.scale[s]));
const at = (parts) => new Map([...parts].flatMap((p) => protoB(p).map((a) => [a, STIM])));
function ledger(segments, reps = 12) {
  const keep = g.scale.slice(), acc = segments.map(() => ({ xy: 0, xx: 0 }));
  learnOnly(S, [g]);
  for (let r = 0; r < reps; r++) {
    S.net.run(300, { rate: inputRate(S), learn: false });
    segments.forEach(([parts, ms], i) => {
      const a = avgW(kXY), b = avgW(kXX);
      S.net.run(ms, { rate: inputRate(S, at(parts)), learn: true });
      acc[i].xy += (avgW(kXY) - a) / reps; acc[i].xx += (avgW(kXX) - b) / reps;
    });
  }
  g.scale.set(keep); learnOnly(S, []);
  return acc;
}
const fmt = (v) => (v >= 0 ? '+' : '') + (v * 1000).toFixed(1);
log('\n每次经历的平均强度变化（×10⁻³；当前平均强度：X 的中继 → Y 的误差 ' + avgW(kXY).toFixed(3) + '，→ X 的误差 ' + avgW(kXX).toFixed(3) + '）：');
const seqSeg = [['AD', 150, 'X 0–150 ms'], ['CB', 30, 'Y 0–30 ms'], ['CB', 120, 'Y 30–150 ms'], ['', 300, '休息 300 ms']];
const a = ledger(seqSeg.map(([p, ms]) => [p, ms]));
log('  序列 X → Y：');
seqSeg.forEach(([, , lab], i) => log(`    ${lab.padEnd(12)} X 中继 → Y 误差 ${fmt(a[i].xy).padStart(7)}    X 中继 → X 误差 ${fmt(a[i].xx).padStart(7)}`));
log(`    合计         X 中继 → Y 误差 ${fmt(a.reduce((s, v) => s + v.xy, 0)).padStart(7)}    X 中继 → X 误差 ${fmt(a.reduce((s, v) => s + v.xx, 0)).padStart(7)}`);
const objSeg = [['AD', 200, 'X 0–200 ms'], ['', 300, '休息 300 ms']];
const b = ledger(objSeg.map(([p, ms]) => [p, ms]));
log('  单独的物体 X：');
objSeg.forEach(([, , lab], i) => log(`    ${lab.padEnd(12)} X 中继 → Y 误差 ${fmt(b[i].xy).padStart(7)}    X 中继 → X 误差 ${fmt(b[i].xx).padStart(7)}`));
log(`    合计         X 中继 → Y 误差 ${fmt(b.reduce((s, v) => s + v.xy, 0)).padStart(7)}    X 中继 → X 误差 ${fmt(b.reduce((s, v) => s + v.xx, 0)).padStart(7)}`);
