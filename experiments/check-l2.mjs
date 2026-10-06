// 第 2 层的物体级通道够不够强：结构 + 短 X（40 ms）之后 Y 本该出现的时段里各层 C、D 的活动（完整 / 没经历序列 / 切断反馈）
import { cfg, proto, OBJ, mean, trainedStack } from '../demo/lib-hierarchy.mjs';
import { inputRate, spikesIn } from '../src/L4-network/layered-network.mjs';

const K = Number(process.env.NL ?? 2);
const blocks = ['A', 'B', 'C', 'D'].map(proto);
const course = (S, segs, reps = 12) => {
  const total = segs.reduce((s, [ms]) => s + ms, 0), nb = total / 10;
  const out = S.layers.map(() => blocks.map(() => new Float64Array(nb)));
  for (let r = 0; r < reps; r++) {
    S.net.run(400, { rate: inputRate(S), learn: false });
    const ts = S.net.t;
    for (const [ms, d] of segs) S.net.run(ms, { rate: inputRate(S, d), learn: false });
    S.layers.forEach((L, k) => blocks.forEach((b, q) => spikesIn(S.net, b.map((a) => L.E[a]), ts, ts + total).forEach((s) => s.forEach((t) => (out[k][q][Math.floor((t - ts) / 10)] += 1 / (20 * 0.01) / reps)))));
  }
  return out;
};
const X = new Map(OBJ.X.flatMap(proto).map((i) => [i, cfg.seqStim])), Y = new Map(OBJ.Y.flatMap(proto).map((i) => [i, cfg.seqStim]));
const fmt = (c) => c.map((L, k) => `第${k + 1}层 ` + L.map((b, q) => 'ABCD'[q] + ':' + Array.from(b).map((v) => v.toFixed(0).padStart(3)).join('')).join('  ')).join('\n    ');
const blk = (a) => (a < 80 ? Math.floor(a / 20) : -1);
for (const [name, S] of [['完整', trainedStack(K)], ['没经历序列', trainedStack(2, { variant: 'noseq' })], ['切断反馈', (() => { const s = trainedStack(K); s.layers.slice(1).forEach((L) => L.FB.scale.fill(0)); return s; })()]]) {
  const XB = OBJ.X.map((p) => 'ABCD'.indexOf(p)), YB2 = OBJ.Y.map((p) => 'ABCD'.indexOf(p));
  const st = S.layers.map((L) => { const m = (f, t) => { let s = 0, n = 0; for (let k = 0; k < L.EE.pre.length; k++) { const a = L.E.indexOf(L.EE.pre[k]), b = L.E.indexOf(L.EE.post[k]); if (f.includes(blk(a)) && t.includes(blk(b))) { s += L.EE.rho[k]; n++; } } return s / n; }; return `物体内 ${((m([XB[0]], [XB[1]]) + m([XB[1]], [XB[0]]) + m([YB2[0]], [YB2[1]]) + m([YB2[1]], [YB2[0]])) / 4).toFixed(2)} X→Y ${m(XB, YB2).toFixed(2)} Y→X ${m(YB2, XB).toFixed(2)}`; });
  console.log(`【${name}】结构：${st.join('；')}`);
  const c1 = course(S, [[40, X], [160, new Map()]]);
  const YB = OBJ.Y.map((p) => 'ABCD'.indexOf(p));
  const win = (k, a, b) => mean(YB.map((q) => mean(Array.from(c1[k][q]).slice(a, b))));
  console.log(`  短 X（40 ms）期间 Y 的组（${OBJ.Y.join('、')}）平均放电：${S.layers.map((_, k) => `第${k + 1}层 ${win(k, 0, 4).toFixed(2)} Hz`).join('，')}；40–120 ms：${S.layers.map((_, k) => `第${k + 1}层 ${win(k, 4, 12).toFixed(2)} Hz`).join('，')}`);
  console.log(`  时间进程（10 ms 一格，Hz）：\n    ${fmt(c1)}`);
  if (name === '完整') console.log(`  序列 X→Y 的时间进程：\n    ${fmt(course(S, [[40, X], [40, Y], [120, new Map()]], 8))}`);
}
