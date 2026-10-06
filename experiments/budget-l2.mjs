// 上层突触预算对物体级通道的影响：重新训练第二层（物体 + 序列），看结构与短 X 时各层 Y 组的活动（完整 / 切断反馈）
import { buildLayered, learnOnly, loadState, inputRate, spikesIn } from '../src/L4-network/layered-network.mjs';
import { cfg, proto, OBJ, trainedState } from '../demo/lib-hierarchy.mjs';
import { makeRng } from '../src/core/rng.mjs';

const B = Number(process.env.BUDGET2 ?? 0.3), ITEM = Number(process.env.ITEM ?? 40), REPS = Number(process.env.REPS ?? 120);
const c2 = { ...cfg, upperBudget: B };
const at = (ids, hz) => new Map(ids.map((i) => [i, hz]));
const run = (S, ms, d, learn) => S.net.run(ms, { rate: inputRate(S, d), learn });
const S = buildLayered(2, c2), L = S.layers[1], rng = makeRng(cfg.seed + 20);
loadState(S, trainedState(1));
learnOnly(S, [L.EE, L.IE, L.FB, S.layers[0].IE]);
run(S, 500, new Map(), false);
for (let n = 0; n < 100; n++) { const parts = OBJ[rng.pick(2) ? 'Y' : 'X']; run(S, 200, at(parts.flatMap((p) => rng.shuffle(proto(p).slice()).slice(0, 12)), cfg.stim), true); run(S, 300, new Map(), true); }
const XB = OBJ.X.map((p) => 'ABCD'.indexOf(p)), YB = OBJ.Y.map((p) => 'ABCD'.indexOf(p)), blk = (a) => (a < 80 ? Math.floor(a / 20) : -1);
const m = (g, f, t, preL, postL) => { let s = 0, n = 0; for (let k = 0; k < g.pre.length; k++) { const a = preL.indexOf(g.pre[k]), b = postL.indexOf(g.post[k]); if (a >= 0 && b >= 0 && f.includes(blk(a)) && t.includes(blk(b))) { s += g.rho[k]; n++; } } return s / n; };
const show = (tag) => console.log(`[预算 ${B}，物体 ${ITEM} ms] ${tag}：第二层 物体内 ${((m(L.EE, [XB[0]], [XB[1]], L.E, L.E) + m(L.EE, [YB[0]], [YB[1]], L.E, L.E)) / 2).toFixed(2)}  X→Y ${m(L.EE, XB, YB, L.E, L.E).toFixed(3)}  Y→X ${m(L.EE, YB, XB, L.E, L.E).toFixed(3)}  反馈 同物体 ${((m(L.FB, YB, YB, L.E, S.layers[0].E) + m(L.FB, XB, XB, L.E, S.layers[0].E)) / 2).toFixed(2)} 跨物体 ${((m(L.FB, YB, XB, L.E, S.layers[0].E) + m(L.FB, XB, YB, L.E, S.layers[0].E)) / 2).toFixed(2)}`);
show('物体阶段后');
learnOnly(S, [L.EE]);
for (let n = 0; n < REPS; n++) { run(S, ITEM, at(OBJ.X.flatMap(proto), cfg.seqStim), true); run(S, ITEM, at(OBJ.Y.flatMap(proto), cfg.seqStim), true); run(S, cfg.seqRestMs, new Map(), true); }
learnOnly(S, []);
show('序列阶段后');
const blocks = ['A', 'B', 'C', 'D'].map(proto);
const course = (reps = 12) => { const total = 100, out = S.layers.map(() => blocks.map(() => new Float64Array(total / 5))); for (let r = 0; r < reps; r++) { run(S, 400, new Map(), false); const ts = S.net.t; run(S, 40, at(OBJ.X.flatMap(proto), cfg.seqStim), false); run(S, 60, new Map(), false); S.layers.forEach((Ly, k) => blocks.forEach((b, q) => spikesIn(S.net, b.map((a) => Ly.E[a]), ts, ts + total).forEach((s) => s.forEach((t) => (out[k][q][Math.floor((t - ts) / 5)] += 1 / (20 * 0.005) / reps))))); } return out; };
const fmt = (c) => S.layers.map((_, k) => `第${k + 1}层 ` + YB.map((q) => 'ABCD'[q] + ':' + Array.from(c[k][q]).slice(0, 10).map((v) => v.toFixed(0).padStart(3)).join('')).join(' ')).join('   ');
console.log(`  短 X 时 Y 的组（5 ms 一格）完整：${fmt(course())}`);
const keep = Float64Array.from(L.FB.scale); L.FB.scale.fill(0);
console.log(`  短 X 时 Y 的组（5 ms 一格）切断反馈：${fmt(course())}`);
L.FB.scale.set(keep);
