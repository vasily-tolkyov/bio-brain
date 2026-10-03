/** 平衡态的两层（只有浅层、没有深层）：每层都是 demo/balanced-layer.mjs 的平衡层，神经元带顶树突；
 * 前馈 ↑：第一层兴奋 → 第二层兴奋，按组对应（同组 ffSame、异组 ffOther），强度固定，落在胞体；另有第一层 → 第二层抑制的前馈抑制；
 * 反馈 ↓：第二层兴奋 → 第一层兴奋的顶树突，按钙规则学（慢、预算按比例缩小）。
 * 世界与旧的多层实验相同：物体 X = A + D、Y = C + B，X 之后是 Y；第一层用训练好的平衡层（兴奋性连接冻结），第二层学物体、物体序列与反馈。
 * 用法：CALIB=1 node demo/balanced-hierarchy.mjs（未训练的两层标定）；node demo/balanced-hierarchy.mjs [noseq]（训练，结果缓存到 cache/balh-2*.json）；
 *       EVAL=1 node demo/balanced-hierarchy.mjs（评估，两份缓存都在时） */
import fs from 'fs';
import { BAL, STIM, SEQ_STIM, protoB, trainBal, spontaneous } from './balanced-layer.mjs';
import { cfg, mean } from './lib-hierarchy.mjs';
import { buildLayered, learnOnly, inputRate, spikesIn, saveState, loadState } from '../src/L4-network/layered-network.mjs';
import { makeRng } from '../src/core/rng.mjs';

export const OBJ = { X: ['A', 'D'], Y: ['C', 'B'] };
// 标定（第一层训练好、第二层未训练）：第二层自发 E 2.6 Hz、I 12.4 Hz；第一层 A 组 +8 / +24 Hz 输入时第二层 A 组 7.1 / 13.5 Hz（分级），
//   B 组 0.6 / 0.3 Hz（按组对应），D 组约 3 Hz（第一层学到的 A→D 经前馈传上来）；前馈 3、上层背景 14–20 时第二层自发只有 0.2–1.7 Hz
export const BAL2 = {
  ...BAL, apical: true, ffSame: Number(process.env.FF_SAME ?? 0.1), ffOther: 0.005, ffScale: Number(process.env.FF_SCALE ?? 6),
  ffiP: 0.1, ffiScale: 1.5, fbP: 0.1, fbAccel: 1, fbBudgetMode: 'mult', bgUpper: Number(process.env.BG_UP ?? 18), upperBudget: 0.3,
};
// 序列每一项的时长：旧网络里 40 ms 一项恰好让整组齐放一次；平衡态里同样 40 ms 第二层每个神经元只放 0.16 个脉冲，
//   紧接着的 Y 只有 0.10 个，"X 先、Y 后"的配对几乎不出现，序列学不到（X→Y 与没经历序列时相同）；150 ms 时 X、Y 各约 0.8–0.9 个
export const SEQ_ITEM = Number(process.env.SEQ_ITEM ?? 150);
// 压缩重放：150 ms 一项时方向仍学不到（X→Y 29.0、Y→X 28.7）——钙规则只在几十毫秒内把前后放电配对，
//   跨过项边界能配上的只有零星脉冲，同一项内 150 ms 的共同激活经预算把权重吸到组内。
//   海马在尖波涟漪期间以 10–20 倍的速度重放刚经历的序列（Lee & Wilson 2002），皮层同步重放（Ji & Wilson 2007）：
//   每遍序列后的休息里插入 n 次 X 20 ms → Y 20 ms 的强驱动（第一层 +d1、第二层直达 +d2，模拟海马经内嗅皮层直达高级皮层）。
//   标定：+120 / +80 时第二层 X 每个神经元约 1.0 个脉冲、Y 约 0.6 个且平均晚 24 ms；只驱动第一层时 +200 也只有 0.38 / 0.13
export const REPLAY = { n: Number(process.env.REPLAY_N ?? 2), item: 20, d1: 120, d2: 80 };
const log = console.log, at = (ids, hz) => new Map(ids.map((i) => [i, hz]));
const run = (S, ms, drives, learn) => S.net.run(ms, { rate: inputRate(S, drives), learn });
const rate = (S, ids, t0, t1) => mean(spikesIn(S.net, ids, t0, t1).map((s) => s.length)) / ((t1 - t0) / 1000);
const grp = (L, k) => protoB(k).map((a) => L.E[a]);

function trainUpperBal(seq) {
  const file = `cache/balh-2-${seq ? `i${SEQ_ITEM}${REPLAY.n ? `-r${REPLAY.n}` : ''}` : 'noseq'}.json`;
  if (fs.existsSync(file)) return JSON.parse(fs.readFileSync(file, 'utf8'));
  const S = buildLayered(2, BAL2), [L1, L2] = S.layers, rng = makeRng(BAL2.seed + 20), t0 = Date.now();
  loadState(S, trainBal());
  learnOnly(S, [L2.EE, L2.IE, L2.FB, L1.IE]);
  run(S, 500, new Map(), false);
  const object = (learn) => { const parts = OBJ[rng.pick(2) ? 'Y' : 'X']; run(S, 200, at(parts.flatMap((p) => rng.shuffle(protoB(p).slice()).slice(0, 48)), STIM), learn); run(S, 300, new Map(), learn); };
  for (let n = 0; n < 100; n++) { object(true); if (n % 25 === 24) log(`  第 2 层：物体 ${n + 1}/100（${((Date.now() - t0) / 1000).toFixed(0)} s）`); }
  if (seq) {
    learnOnly(S, [L2.EE]);
    const replayDrive = (o) => ({ rate: inputRate(S, at(OBJ[o].flatMap(protoB), BAL.bg + REPLAY.d1), new Map(OBJ[o].flatMap((p) => grp(L2, p)).map((i) => [i, BAL2.bgUpper + REPLAY.d2]))), learn: true });
    for (let n = 0; n < 80; n++) {
      run(S, SEQ_ITEM, at(OBJ.X.flatMap(protoB), SEQ_STIM), true);
      run(S, SEQ_ITEM, at(OBJ.Y.flatMap(protoB), SEQ_STIM), true);
      // 休息（总长不变）：每次重放 = X 20 ms → Y 20 ms，前后各隔 80 ms
      let rest = cfg.seqRestMs;
      for (let r = 0; r < REPLAY.n; r++) {
        run(S, 80, new Map(), true);
        S.net.run(REPLAY.item, replayDrive('X'));
        S.net.run(REPLAY.item, replayDrive('Y'));
        rest -= 80 + 2 * REPLAY.item;
      }
      run(S, rest, new Map(), true);
      if (n % 2) object(true); // 序列与物体交错：只学序列时后继项的吸引子被 X→Y 挤掉
      if (n % 20 === 19) log(`  第 2 层：序列 ${n + 1}/80（${((Date.now() - t0) / 1000).toFixed(0)} s）`);
    }
  }
  log(`  第 2 层训练完成（${((Date.now() - t0) / 1000).toFixed(0)} s）`);
  const state = saveState(S);
  fs.writeFileSync(file, JSON.stringify(state));
  return state;
}
export function balHier({ trained = true, seq = true } = {}) {
  const S = buildLayered(2, BAL2);
  loadState(S, trained ? trainUpperBal(seq) : trainBal());
  learnOnly(S, []);
  return S;
}

/** 未训练（第一层训练好、第二层未训练）的标定：两层的自发活动；驱动第一层 A 组时第二层各组的频率 */
function calibrate() {
  const S = balHier({ trained: false }), [L1, L2] = S.layers;
  S.net.run(500, { rate: inputRate(S), learn: false });
  const t0 = S.net.t; S.net.run(1500, { rate: inputRate(S), learn: false });
  const sp = [L1, L2].map((L) => ({ E: rate(S, L.E, t0, t0 + 1500), I: rate(S, L.I, t0, t0 + 1500) }));
  const res = [];
  for (const d of [8, 24]) {
    const acc = { A1: 0, A2: 0, B2: 0, D2: 0 };
    for (let k = 0; k < 4; k++) {
      S.net.run(300, { rate: inputRate(S), learn: false });
      const ts = S.net.t; run(S, 200, at(protoB('A'), BAL.bg + d), false);
      acc.A1 += rate(S, grp(L1, 'A'), ts + 50, ts + 200) / 4; acc.A2 += rate(S, grp(L2, 'A'), ts + 50, ts + 200) / 4;
      acc.B2 += rate(S, grp(L2, 'B'), ts + 50, ts + 200) / 4; acc.D2 += rate(S, grp(L2, 'D'), ts + 50, ts + 200) / 4;
    }
    res.push({ d, ...acc });
  }
  log(`ffSame ${BAL2.ffSame}、ffScale ${BAL2.ffScale}、上层背景 ${BAL2.bgUpper} | 自发：第一层 E ${sp[0].E.toFixed(1)} / I ${sp[0].I.toFixed(1)}，第二层 E ${sp[1].E.toFixed(1)} / I ${sp[1].I.toFixed(1)} Hz | ` +
    res.map((r) => `第一层 A +${r.d}：第一层 A ${r.A1.toFixed(1)}、第二层 A ${r.A2.toFixed(1)}、B ${r.B2.toFixed(1)}、D ${r.D2.toFixed(1)}`).join('；'));
}

// ---------- 评估 ----------
const ctrl = (L) => L.E.slice(BAL.K * BAL.M); // 不属于任何组的 80 个
const objIds = (L, o) => OBJ[o].flatMap((p) => grp(L, p));
const objOfIdx = (a) => { const b = Math.floor(a / BAL.M); return OBJ.X.includes('ABCD'[b]) ? 'X' : OBJ.Y.includes('ABCD'[b]) ? 'Y' : null; };

/** 每个突触后神经元从某物体收到的 scale 之和，按突触后物体平均 */
function objDrive(g, preL, postL) {
  const pre = new Map(preL.E.map((i, a) => [i, objOfIdx(a)])), post = new Map(postL.E.map((i, a) => [i, objOfIdx(a)])), sum = {}, n = { X: 0, Y: 0 };
  for (const o of post.values()) if (o) n[o]++;
  for (let s = 0; s < g.pre.length; s++) { const a = pre.get(g.pre[s]), b = post.get(g.post[s]); if (a && b) sum[a + b] = (sum[a + b] ?? 0) + g.scale[s]; }
  return Object.fromEntries(['XX', 'XY', 'YY', 'YX'].map((k) => [k, +((sum[k] ?? 0) / n[k[1]]).toFixed(1)]));
}

/** 一段试验：segments 依次运行，返回 [t0, t0+总长) 内的时间进程（20 ms 分箱的频率）与顶树突平均电位（1 ms） */
function episode(S, segments, sets, vsets, reps) {
  const total = segments.reduce((s, [ms]) => s + ms, 0), nb = Math.ceil(total / 20);
  const out = Object.fromEntries(Object.keys(sets).map((k) => [k, new Float64Array(nb)])), vd = Object.fromEntries(Object.keys(vsets).map((k) => [k, new Float64Array(total)]));
  for (let r = 0; r < reps; r++) {
    S.net.run(400, { rate: inputRate(S), learn: false });
    const ts = S.net.t; let step = 0;
    const onStep = () => { if (step++ % 10) return; const ms = (step / 10) | 0; if (ms < total) for (const [k, ids] of Object.entries(vsets)) vd[k][ms] += mean(ids.map((i) => S.net.Vd[i])) / reps; };
    for (const [ms, d] of segments) S.net.run(ms, { rate: inputRate(S, d), learn: false, onStep });
    for (const [k, ids] of Object.entries(sets)) spikesIn(S.net, ids, ts, ts + total).forEach((s) => s.forEach((t) => (out[k][Math.floor((t - ts) / 20)] += 1000 / (20 * ids.length * reps))));
  }
  return { out, vd };
}
const win = (arr, a, b, bin = 20) => mean(Array.from(arr).slice(a / bin, b / bin));

function evaluate(S, name) {
  const [L1, L2] = S.layers, T = SEQ_ITEM, r = { name };
  // 自发
  S.net.run(500, { rate: inputRate(S), learn: false });
  const t0 = S.net.t; S.net.run(1500, { rate: inputRate(S), learn: false });
  r.spont = [L1, L2].map((L) => +rate(S, L.E, t0, t0 + 1500).toFixed(2));
  // 学到的连接
  r.EE2 = objDrive(L2.EE, L2, L2); r.FB = objDrive(L2.FB, L2, L1);
  // 跨层补全：只驱动第一层 A
  const C = episode(S, [[100, new Map()], [200, at(protoB('A'), BAL.bg + 24)]], { D1: grp(L1, 'D'), B1: grp(L1, 'B'), D2: grp(L2, 'D'), B2: grp(L2, 'B') }, { D1: grp(L1, 'D'), B1: grp(L1, 'B') }, 8);
  r.completion = { D2: win(C.out.D2, 150, 300), B2: win(C.out.B2, 150, 300), D1: win(C.out.D1, 150, 300), B1: win(C.out.B1, 150, 300),
    apicalD1minusB1: win(C.vd.D1, 150, 300, 1) - win(C.vd.B1, 150, 300, 1) - (win(C.vd.D1, 0, 100, 1) - win(C.vd.B1, 0, 100, 1)) };
  // 序列预测：只给 X（与训练时一样长），看 X 结束后 0–160 ms
  const P = episode(S, [[100, new Map()], [T, at(OBJ.X.flatMap(protoB), SEQ_STIM)], [200, new Map()]], { Y2: objIds(L2, 'Y'), C2: ctrl(L2), Y1: objIds(L1, 'Y'), C1: ctrl(L1) }, { Y1: objIds(L1, 'Y'), C1: ctrl(L1) }, 10);
  const base = (k) => win(P.out[k], 0, 100), w0 = 100 + T, w1 = 100 + T + 160;
  r.predict = { Y2: win(P.out.Y2, w0, w1) - base('Y2'), C2: win(P.out.C2, w0, w1) - base('C2'), Y1: win(P.out.Y1, w0, w1) - base('Y1'), C1: win(P.out.C1, w0, w1) - base('C1'),
    Y2during: win(P.out.Y2, 100, w0) - base('Y2'), C2during: win(P.out.C2, 100, w0) - base('C2'),
    apicalY1minusC1: win(P.vd.Y1, w0, w1, 1) - win(P.vd.C1, w0, w1, 1) - (win(P.vd.Y1, 0, 100, 1) - win(P.vd.C1, 0, 100, 1)),
    courseY2: Array.from(P.out.Y2, (v) => +v.toFixed(1)), courseC2: Array.from(P.out.C2, (v) => +v.toFixed(1)) };
  // 真正的 Y 在 X 开始后 T（紧跟 X，与训练时一样）/ T + 80 ms 出现（0 = 前面没有 X）：头 20、40 ms 的放电减去出现前 40 ms 的基线
  r.facil = [0, T, T + 80].map((on) => {
    const segs = on ? [[100, new Map()], [T, at(OBJ.X.flatMap(protoB), SEQ_STIM)], [on - T, new Map()], [40, at(OBJ.Y.flatMap(protoB), SEQ_STIM)], [60, new Map()]]
      : [[100 + 80, new Map()], [40, at(OBJ.Y.flatMap(protoB), SEQ_STIM)], [60, new Map()]];
    const onset = on ? 100 + on : 180;
    const E = episode(S, segs, { Y1: objIds(L1, 'Y'), Y2: objIds(L2, 'Y') }, {}, 8);
    const f = (k) => ({ pre: win(E.out[k], onset - 40, onset), e20: win(E.out[k], onset, onset + 20) - win(E.out[k], onset - 40, onset), e40: win(E.out[k], onset, onset + 40) - win(E.out[k], onset - 40, onset) });
    return { on, Y1: f('Y1'), Y2: f('Y2') };
  });
  return r;
}

function report(r) {
  log(`\n【${r.name}】自发：第一层 ${r.spont[0]} Hz、第二层 ${r.spont[1]} Hz`);
  log(`  第二层学到的物体级连接：X→X ${r.EE2.XX}  X→Y ${r.EE2.XY}  Y→Y ${r.EE2.YY}  Y→X ${r.EE2.YX}；反馈（第二层 → 第一层顶树突）：X→X ${r.FB.XX}  X→Y ${r.FB.XY}  Y→Y ${r.FB.YY}  Y→X ${r.FB.YX}`);
  const c = r.completion;
  log(`  只驱动第一层 A（50–200 ms）：第二层 D ${c.D2.toFixed(1)} / 对照 B ${c.B2.toFixed(1)} Hz；第一层 D ${c.D1.toFixed(1)} / B ${c.B1.toFixed(1)} Hz；第一层 D 的顶树突比 B 高 ${c.apicalD1minusB1.toFixed(2)} mV（减去刺激前）`);
  const p = r.predict;
  log(`  只给 X ${SEQ_ITEM} ms（减去刺激前）：X 期间第二层 Y ${p.Y2during.toFixed(1)} / 不成组 ${p.C2during.toFixed(1)} Hz；X 结束后 0–160 ms 第二层 Y ${p.Y2.toFixed(1)} / 不成组 ${p.C2.toFixed(1)} Hz；第一层 Y ${p.Y1.toFixed(1)} / 不成组 ${p.C1.toFixed(1)} Hz；第一层 Y 的顶树突比不成组高 ${p.apicalY1minusC1.toFixed(2)} mV`);
  log(`    第二层 Y 时间进程（20 ms 分箱，X 在 100–${100 + SEQ_ITEM} ms）：${p.courseY2.join(' ')}`);
  log(`    第二层不成组 时间进程：${p.courseC2.join(' ')}`);
  for (const f of r.facil) log(`  Y ${f.on ? `在 X 开始后 ${f.on} ms（X 结束后 ${f.on - SEQ_ITEM} ms）` : '前面没有 X'}出现：第一层 Y 头 20 / 40 ms 比基线高 ${f.Y1.e20.toFixed(1)} / ${f.Y1.e40.toFixed(1)} Hz（基线 ${f.Y1.pre.toFixed(1)}）；第二层 Y ${f.Y2.e20.toFixed(1)} / ${f.Y2.e40.toFixed(1)} Hz（基线 ${f.Y2.pre.toFixed(1)}）`);
}

if (import.meta.url === `file:///${process.argv[1].replace(/\\/g, '/')}`) {
  if (process.env.CALIB) calibrate();
  else if (process.env.EVAL) {
    const nets = { two: () => balHier({ seq: true }), noseq: () => balHier({ seq: false }), cut: () => { const S = balHier({ seq: true }); S.layers[1].FB.scale.fill(0); return S; } };
    const names = { two: '2 层', noseq: '2 层，第二层没经历物体序列', cut: '2 层，切断反馈' };
    for (const k of (process.env.NET ?? 'two,noseq,cut').split(',')) {
      const r = evaluate(nets[k](), names[k]);
      report(r);
      fs.writeFileSync(`scratch/balh-eval-${k}.json`, JSON.stringify(r));
    }
  } else trainUpperBal(process.argv[2] !== 'noseq');
}
