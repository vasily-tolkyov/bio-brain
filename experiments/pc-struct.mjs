// 预测编码两层：学到的第二层递归结构与预测抑制矩阵
import fs from 'fs';
import { PC2, grp } from '../demo/predictive-hierarchy.mjs';
import { buildLayered, loadState } from '../src/L4-network/layered-network.mjs';
import { mean } from '../demo/lib-hierarchy.mjs';

const G = 'ABCD';
const files = process.argv.slice(2).length ? process.argv.slice(2) : ['cache/pc-2.json', 'cache/pc-2-seq.json'];
for (const f of files) {
  const S = buildLayered(2, PC2); loadState(S, JSON.parse(fs.readFileSync(f, 'utf8')).state);
  const [L1, L2] = S.layers;
  const between = (g, key, pk, pp, qk, qp, val) => { const a = new Set(grp(S, pk, pp[0], pp[1])), b = new Set(grp(S, qk, qp[0], qp[1])), v = []; for (let k = 0; k < g.pre.length; k++) if (a.has(g.pre[k]) && b.has(g.post[k])) v.push(val(g, k)); return mean(v); };
  const rho = (g, k) => g.rho[k], sc = (g, k) => g.scale[k];
  const ee = (p, q) => between(L2.EE, 'EE', 1, ['E', p], 1, ['E', q], rho);
  console.log(`${f}：第二层 物体内 A↔D ${((ee('A', 'D') + ee('D', 'A')) / 2).toFixed(2)}，C↔B ${((ee('C', 'B') + ee('B', 'C')) / 2).toFixed(2)}；X→Y ${mean(['A', 'D'].flatMap((p) => ['C', 'B'].map((q) => ee(p, q)))).toFixed(2)}，Y→X ${mean(['C', 'B'].flatMap((p) => ['A', 'D'].map((q) => ee(p, q)))).toFixed(2)}`);
  for (const [name, L] of [['第一层', L1], ['第二层', L2]]) {
    console.log(`  ${name}预测抑制（行 = 中继组，列 = 误差组 A B C D）：`);
    for (const p of G) console.log(`    S${p} [${G.split('').map((q) => between(L.PRED, 'PRED', L.k - 1, ['S', p], L.k - 1, ['N', q], sc).toFixed(2).padStart(6)).join('')}]`);
  }
}
