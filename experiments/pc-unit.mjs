// 单层"表征 + 误差"最小版本：上层的预测用外部驱动中继群 S 代替
// MODE=cal：标定（未学习）；MODE=train：学习 + 测试
import { BAL, trainBal, protoB, STIM } from '../demo/balanced-layer.mjs';
import { buildLayered, learnOnly, inputRate, spikesIn, loadState, groupIds } from '../src/L4-network/layered-network.mjs';
import { mean } from '../demo/lib-hierarchy.mjs';
import { makeRng } from '../src/core/rng.mjs';

const over = JSON.parse(process.env.PCO ?? '{}'), CTX = Number(process.env.CTX ?? 20), NTR = Number(process.env.NTR ?? 100);
const S = buildLayered(1, { ...BAL, pc: true, pcOver: over });
loadState(S, trainBal()); learnOnly(S, []);
const L = S.layers[0], pc = S.cfg.pcP, log = console.log;
const G = 'ABCD', gN = (k) => groupIds(S, L.N, G.indexOf(k)), gS = (k) => groupIds(S, L.S, G.indexOf(k)), gE = (k) => protoB(k);
const rate = (ids, t0, t1) => mean(spikesIn(S.net, ids, t0, t1).map((s) => s.length)) / ((t1 - t0) / 1000);
function trial(ev, ctx, ms, learn = false) {
  const drives = new Map(); for (const k of ev) for (const a of gE(k)) drives.set(a, STIM);
  const extra = new Map(); for (const k of ctx) for (const i of gS(k)) extra.set(i, pc.bgS + CTX);
  const t0 = S.net.t; S.net.run(ms, { rate: inputRate(S, drives, extra), learn }); return t0;
}
const fmt = (x) => x.toFixed(1).padStart(5);
function probe(ev, ctx, reps = 6, label = '') {
  const acc = { E: [0, 0, 0, 0], N: [0, 0, 0, 0], S: [0, 0, 0, 0], Ne: [0, 0, 0, 0], Nl: [0, 0, 0, 0] };
  for (let r = 0; r < reps; r++) {
    const t0 = trial(ev, ctx, 200);
    G.split('').forEach((k, q) => {
      acc.E[q] += rate(gE(k), t0, t0 + 200) / reps; acc.N[q] += rate(gN(k), t0, t0 + 200) / reps; acc.S[q] += rate(gS(k), t0, t0 + 200) / reps;
      acc.Ne[q] += rate(gN(k), t0, t0 + 50) / reps; acc.Nl[q] += rate(gN(k), t0 + 100, t0 + 200) / reps;
    });
    trial('', '', 300);
  }
  log(`  ${label.padEnd(22)} E[${acc.E.map(fmt).join('')}]  S[${acc.S.map(fmt).join('')}]  N[${acc.N.map(fmt).join('')}]  N早[${acc.Ne.map(fmt).join('')}] N晚[${acc.Nl.map(fmt).join('')}]`);
  return acc;
}
function base(ms = 2000) {
  trial('', '', 500); const t0 = trial('', '', ms);
  log(`  自发：E ${rate(L.E, t0, t0 + ms).toFixed(2)}  I ${rate(L.I, t0, t0 + ms).toFixed(2)}  N ${rate(L.N, t0, t0 + ms).toFixed(2)}  S ${rate(L.S, t0, t0 + ms).toFixed(2)} Hz`);
}
const meanScale = (g, f) => { let s = 0, n = 0; for (let k = 0; k < g.pre.length; k++) if (f(g.pre[k], g.post[k])) { s += g.scale[k]; n++; } return n ? s / n : 0; };
const inG = (ids) => { const set = new Set(ids); return (i) => set.has(i); };
function predMatrix() {
  const rows = G.split('').map((p) => G.split('').map((q) => meanScale(L.PRED, (a, b) => inG(gS(p))(a) && inG(gN(q))(b))));
  log('  预测抑制强度（行 = 中继组，列 = 误差组）：'); rows.forEach((r, q) => log(`    S${G[q]} [${r.map((v) => v.toFixed(2).padStart(6)).join('')}]`));
}
log(`配置：${JSON.stringify(over)} CTX ${CTX}`);
base();
log('  时窗：全程 0–200 ms；早 0–50；晚 100–200（列 = A B C D 组）');
probe('AD', '', 6, 'X 证据，无预测');
probe('AD', 'AD', 6, 'X 证据 + X 预测');
if (!process.env.QUICK) { probe('', 'AD', 6, '只有 X 预测'); probe('CB', 'AD', 6, 'Y 证据 + X 预测'); }
if (process.env.MODE === 'train') {
  const rng = makeRng(7), t0 = Date.now();
  learnOnly(S, [L.PRED]);
  for (let n = 0; n < NTR; n++) {
    const o = rng.uniform() < 0.5 ? 'AD' : 'CB';
    trial(o, o, 200, true); trial('', '', 300, true);
    if (n % 25 === 24) { log(`  学习 ${n + 1}/${NTR}（${((Date.now() - t0) / 1000).toFixed(0)} s）`); }
  }
  learnOnly(S, []);
  predMatrix();
  base();
  probe('AD', '', 6, 'X 证据，无预测');
  probe('AD', 'AD', 6, 'X 证据 + X 预测');
  probe('', 'AD', 6, '只有 X 预测');
  probe('CB', 'AD', 6, 'Y 证据 + X 预测');
  probe('A', 'AD', 6, '只有 A + X 预测');
}
