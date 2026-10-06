// 第二层物体序列的时间压缩：每个物体 ITEM ms，看 X→Y 通道和短 X 时各层 Y 组的活动
import fs from 'fs';
import { buildLayered, learnOnly, saveState, loadState, inputRate, spikesIn } from '../src/L4-network/layered-network.mjs';
import { cfg, proto, OBJ, mean, trainedState } from '../demo/lib-hierarchy.mjs';
import { makeRng } from '../src/core/rng.mjs';

const ITEM = Number(process.env.ITEM ?? 20), REPS = Number(process.env.REPS ?? 120);
const at = (ids, hz) => new Map(ids.map((i) => [i, hz]));
const run = (S, ms, d, learn) => S.net.run(ms, { rate: inputRate(S, d), learn });
const S = buildLayered(2, cfg), L = S.layers[1], rng = makeRng(cfg.seed + 20);
loadState(S, trainedState(1));
// 物体阶段与 lib-hierarchy 相同；只取它训练好的第二层状态，省时间
const st2 = JSON.parse(fs.readFileSync('cache/hier-2-noseq.json', 'utf8'));
loadState(S, st2);
learnOnly(S, [L.EE]);
for (let n = 0; n < REPS; n++) { run(S, ITEM, at(OBJ.X.flatMap(proto), cfg.seqStim), true); run(S, ITEM, at(OBJ.Y.flatMap(proto), cfg.seqStim), true); run(S, cfg.seqRestMs, new Map(), true); }
learnOnly(S, []);
const XB = OBJ.X.map((p) => 'ABCD'.indexOf(p)), YB = OBJ.Y.map((p) => 'ABCD'.indexOf(p)), blk = (a) => (a < 80 ? Math.floor(a / 20) : -1);
const m = (f, t) => { let s = 0, n = 0; for (let k = 0; k < L.EE.pre.length; k++) { const a = L.E.indexOf(L.EE.pre[k]), b = L.E.indexOf(L.EE.post[k]); if (f.includes(blk(a)) && t.includes(blk(b))) { s += L.EE.rho[k]; n++; } } return s / n; };
const blocks = ['A', 'B', 'C', 'D'].map(proto);
const course = (segs, reps = 12) => { const total = segs.reduce((s, [ms]) => s + ms, 0), out = S.layers.map(() => blocks.map(() => new Float64Array(total / 5))); for (let r = 0; r < reps; r++) { run(S, 400, new Map(), false); const ts = S.net.t; for (const [ms, d] of segs) run(S, ms, d, false); S.layers.forEach((Ly, k) => blocks.forEach((b, q) => spikesIn(S.net, b.map((a) => Ly.E[a]), ts, ts + total).forEach((s) => s.forEach((t) => (out[k][q][Math.floor((t - ts) / 5)] += 1 / (20 * 0.005) / reps))))); } return out; };
const c = course([[40, at(OBJ.X.flatMap(proto), cfg.seqStim)], [60, new Map()]]);
console.log(`物体时长 ${ITEM} ms × ${REPS} 遍：第二层 X→Y ${m(XB, YB).toFixed(3)}，Y→X ${m(YB, XB).toFixed(3)}，物体内 ${((m([XB[0]], [XB[1]]) + m([YB[0]], [YB[1]])) / 2).toFixed(2)}`);
console.log(`  短 X 时 Y 的组（5 ms 一格）：${S.layers.map((_, k) => `第${k + 1}层 ` + YB.map((q) => 'ABCD'[q] + ':' + Array.from(c[k][q]).slice(0, 12).map((v) => v.toFixed(0).padStart(3)).join('')).join(' ')).join('   ')}`);
fs.writeFileSync(`scratch/seq-item-${ITEM}.json`, JSON.stringify(saveState(S)));
