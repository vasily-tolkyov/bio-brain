// 第一层各组之间的平均效能（行 = 突触前组，列 = 突触后组）
import fs from 'fs';
import { buildLayered, loadState } from '../src/L4-network/layered-network.mjs';
import { cfg } from '../demo/lib-hierarchy.mjs';

const file = process.env.F ?? 'cache/hier-1.json';
const S = buildLayered(1, { ...cfg, seed: Number(process.env.SEED ?? cfg.seed) }); loadState(S, JSON.parse(fs.readFileSync(file, 'utf8')));
const L = S.layers[0], blk = (a) => (a < 80 ? Math.floor(a / 20) : 4);
const s = Array.from({ length: 5 }, () => new Float64Array(5)), n = Array.from({ length: 5 }, () => new Float64Array(5));
for (let k = 0; k < L.EE.pre.length; k++) { const a = blk(L.EE.pre[k] - L.E[0]), b = blk(L.EE.post[k] - L.E[0]); s[a][b] += L.EE.rho[k]; n[a][b]++; }
console.log('      ' + ['A', 'B', 'C', 'D', '其他'].map((x) => x.padStart(6)).join(''));
s.forEach((row, a) => console.log(['A', 'B', 'C', 'D', 'O'][a].padEnd(4) + Array.from(row).map((v, b) => (v / n[a][b]).toFixed(2).padStart(6)).join('')));
