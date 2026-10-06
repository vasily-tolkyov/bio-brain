// 序列期待的干净对照：同一个网络、同一段 X → Y，只把「X 的中继 → Y 的误差」这一组突触置零，看 Y 出现后 Y 的误差差多少
import fs from 'fs';
import { PC2, grp, sequenceCourse } from '../demo/predictive-hierarchy.mjs';
import { buildLayered, loadState, learnOnly } from '../src/L4-network/layered-network.mjs';
import { mean } from '../demo/lib-hierarchy.mjs';

const log = console.log, base = JSON.parse(fs.readFileSync('cache/pc-2.json', 'utf8')).state;
const nets = {
  '只学物体': () => { const S = buildLayered(2, PC2); loadState(S, base); return S; },
  '序列，20 ms 窗口': () => { const S = buildLayered(2, PC2); loadState(S, JSON.parse(fs.readFileSync('cache/pc-2-seq.json', 'utf8')).state); return S; },
  '序列，200 ms 痕迹': () => { const S = buildLayered(2, PC2); loadState(S, base); const w = JSON.parse(fs.readFileSync('cache/pc-2-seq-trace200.json', 'utf8')); loadState(S, [{ PRED: w.L1PRED }, { EE: w.L2EE, IE: w.L2IE, PRED: w.L2PRED }]); return S; },
};
const f = (v) => v.toFixed(1).padStart(5);
log('Y 紧跟 X（X 150 ms → Y 150 ms），第一层 Y 的误差：头 50 ms 平均 / 10 ms 分箱的峰值 / 整段 Y 平均（Hz，16 次）');
for (const [name, make] of Object.entries(nets)) {
  const S = make(); learnOnly(S, []);
  const [L1] = S.layers, g = L1.PRED;
  const sX = new Set([...grp(S, 0, 'S', 'A'), ...grp(S, 0, 'S', 'D')]), nY = new Set([...grp(S, 0, 'N', 'C'), ...grp(S, 0, 'N', 'B')]);
  const kXY = []; for (let s = 0; s < g.pre.length; s++) if (sX.has(g.pre[s]) && nY.has(g.post[s])) kXY.push(s);
  const measure = () => { const tc = sequenceCourse(S, [['AD', 150], ['CB', 150], ['', 200]], { reps: 16 }); const y = [...tc.N1C].map((v, b) => (v + tc.N1B[b]) / 2); return { first: mean(y.slice(15, 20)), peak: Math.max(...y.slice(15, 25)), whole: mean(y.slice(15, 30)), w: mean(kXY.map((s) => g.scale[s])) }; };
  const on = measure();
  const keep = g.scale.slice(); for (const s of kXY) g.scale[s] = 0;
  const off = measure(); g.scale.set(keep);
  log(`  ${name.padEnd(14)} X 中继 → Y 误差 平均强度 ${on.w.toFixed(2)}：完整 ${f(on.first)} / ${f(on.peak)} / ${f(on.whole)}；置零 ${f(off.first)} / ${f(off.peak)} / ${f(off.whole)}；差 ${f(off.first - on.first)} / ${f(off.peak - on.peak)} / ${f(off.whole - on.whole)}`);
}
