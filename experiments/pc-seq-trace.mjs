// 只在序列阶段用长的突触前痕迹（TAUPRE ms）学预测抑制：从只学过物体的网络（cache/pc-2.json）出发，
// 重复 demo/predictive-hierarchy.mjs 的序列训练，再做序列检验与物体层面的误差；与只学物体的网络对照
import fs from 'fs';
import { PC2, OBJ, SEQ_ITEM, grp, sequenceTests, timeCourse, spontaneous } from '../demo/predictive-hierarchy.mjs';
import { STIM, protoB } from '../demo/balanced-layer.mjs';
import { buildLayered, loadState, learnOnly, inputRate } from '../src/L4-network/layered-network.mjs';
import { makeRng } from '../src/core/rng.mjs';
import { mean } from '../demo/lib-hierarchy.mjs';

const TAUPRE = Number(process.env.TAUPRE ?? 200), log = console.log, t0 = Date.now();
const cfg = { ...PC2, pcOver: { ...PC2.pcOver, predTauPre: TAUPRE } };
const base = JSON.parse(fs.readFileSync('cache/pc-2.json', 'utf8')).state;
const S = buildLayered(2, cfg), [L1, L2] = S.layers, rng = makeRng(PC2.seed + 30);
loadState(S, base);
const at = (ids, hz) => new Map(ids.map((i) => [i, hz]));
const run = (ms, drives, learn) => S.net.run(ms, { rate: inputRate(S, drives), learn });
const object = () => { const parts = OBJ[rng.pick(2) ? 'Y' : 'X']; run(200, at(parts.flatMap((p) => rng.shuffle(protoB(p).slice()).slice(0, 48)), STIM), true); run(300, new Map(), true); };
const G = 'ABCD', avgW = (pre, post) => { const a = new Set(grp(S, 0, 'S', pre)), b = new Set(grp(S, 0, 'N', post)), v = []; const g = L1.PRED; for (let k = 0; k < g.pre.length; k++) if (a.has(g.pre[k]) && b.has(g.post[k])) v.push(g.scale[k]); return mean(v); };
const matrix = (title) => { log(`  ${title}（行 = 中继组，列 = 误差组 A B C D）：`); for (const p of G) log(`    S${p} [${G.split('').map((q) => avgW(p, q).toFixed(2).padStart(6)).join('')}]`); };
matrix('序列训练前的预测抑制');
learnOnly(S, [L2.EE, L2.IE, L1.PRED, L2.PRED]);
run(500, new Map(), false);
for (let n = 0; n < 80; n++) {
  run(SEQ_ITEM, at(OBJ.X.flatMap(protoB), STIM), true);
  run(SEQ_ITEM, at(OBJ.Y.flatMap(protoB), STIM), true);
  run(300, new Map(), true);
  if (n % 2) object();
  if (n % 20 === 19) log(`  序列 ${n + 1}/80（${((Date.now() - t0) / 1000).toFixed(0)} s）`);
}
learnOnly(S, []);
matrix('序列训练后的预测抑制');
fs.writeFileSync(`cache/pc-2-seq-trace${TAUPRE}.json`, JSON.stringify({ L1PRED: [...L1.PRED.scale], L2PRED: [...L2.PRED.scale], L2EE: [...L2.EE.rho], L2IE: [...L2.IE.scale] }));
const r = sequenceTests(S, `经历序列，痕迹 ${TAUPRE} ms`);
// 物体层面是否还在：熟悉的 X、Y 的误差（50–200 ms）与自发
const sp = spontaneous(S)[0], errOf = (tc, parts) => mean(parts.map((p) => mean([...tc[`N1${p}`]].slice(2))));
const fx = timeCourse(S, OBJ.X), fy = timeCourse(S, OBJ.Y), oa = timeCourse(S, ['A']);
log(`  物体层面：第一层误差自发 ${sp.N.toFixed(1)} Hz；熟悉的 X ${errOf(fx, OBJ.X).toFixed(1)}、Y ${errOf(fy, OBJ.Y).toFixed(1)} Hz；X 呈现时 Y 的误差 ${errOf(fx, OBJ.Y).toFixed(1)} Hz；只有 A 时缺失的 D ${errOf(oa, ['D']).toFixed(1)} Hz`);
log(`  完成（${((Date.now() - t0) / 1000).toFixed(0)} s）`);
