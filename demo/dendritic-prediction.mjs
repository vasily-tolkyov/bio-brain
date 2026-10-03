/** 预测由树突承载：自上而下的预测落在锥体细胞的顶树突（阈下），自下而上的证据落在胞体，两者在同一个细胞的不同区室相遇，
 * 匹配时产生平台与爆发（Larkum 2013）。预测不需要任何神经元提前放电，所以不需要靠持续放电去"保持"——
 * 之前所有版本（深层保持、θ、γ、易化、AMPA 拷贝）都卡在"保持要靠小群体的强递归、增益略大就失控略小就保持不住"。
 *
 * 结构（每层相同，平衡态，见 demo/balanced-layer.mjs）：400 个两区室锥体细胞（A B C D 四组各 80 + 80 个不成组）+ PV 60 / SOM 25 / VIP 15；
 *   PV 快抑制打胞体（γ、增益、慢稳态），SOM 打顶树突（树突的门），VIP 抑制 SOM（开门）。
 * 层间：前馈 ↑ 打胞体（按组对应、固定）+ 前馈抑制打 PV；反馈 ↓ 从上层兴奋性细胞打下层顶树突，按平台 × 资格迹学习
 *   （行为时间尺度的可塑性：Bittner 等 2017；新皮层 Gambino 等 2014）；上层另投射到下层 VIP（固定）。
 * 神经调质 net.mod（模拟乙酰胆碱）：1 = 学习态（VIP 背景升高、顶树突去极化 250 pA、平台规则打开），0 = 回忆态。
 *   标定（未训练单层，A 组强驱动 150 ms）：学习态 57% 的 A 出现平台、其他神经元 4%、自发几乎没有；回忆态都没有（SOM 15 Hz 把顶树突压低 1.6 mV）。
 * 同一条反馈规则学到两件事：X 在上层活跃时下层 A、D 在放电（物体的补全），随后 C、B 在放电（序列的预测）。
 * 物体存在上层递归里（钙规则，只在物体阶段学），序列存在反馈里——两者不再争同一份预算。
 *
 * 用法：node demo/dendritic-prediction.mjs [noseq]（训练，缓存 cache/dend-*.json）；EVAL=1 NET=two|noseq|cut node demo/dendritic-prediction.mjs（评估） */
import fs from 'fs';
import { BAL, STIM, SEQ_STIM, protoB } from './balanced-layer.mjs';
import { cfg, mean } from './lib-hierarchy.mjs';
import { buildLayered, learnOnly, inputRate, spikesIn, saveState, loadState } from '../src/L4-network/layered-network.mjs';
import { makeRng } from '../src/core/rng.mjs';

export const INH_BAL = {
  nPV: 60, nSOM: 25, nVIP: 15, pEPV: 0.2, sEPV: 1.5, pPVE: 0.6, sPVE: 4, pPVPV: 0.6, sPVPV: 4,
  pESOM: 0.1, sESOM: 3, pSOME: 0.4, sSOME: 10, pSOMPV: 0.3, sSOMPV: 3, pSOMVIP: 0.3, sSOMVIP: 3, pVIPSOM: 0.6, sVIPSOM: 8,
  pEVIP: 0.05, sEVIP: 1, somBg: 12, vipBg: 9, vipMod: 8, pFBV: Number(process.env.P_FBV ?? 0.05), sFBV: 2,
};
// 第一版（整个训练都处在学习态、平台规则只增强、反馈在学习态全效传递）反馈学得完全不分对象：送到不成组神经元的反馈与送到物体部件的一样强
//   （X→X 14.5、X→Y 13.8、X→不成组 12.6）——休息期间上层的自发放电（资格迹均值约 1）与下层的自发平台在全部突触上随机相遇；
//   反馈一旦变强还会在学习态里自己引出平台、把自己越写越强（单元测试：效能 0.30 → 0.79，平台出现在突触后被驱动之前）。三处修正：
//   神经调质是相位性的——乙酰胆碱只在显著事件时短暂释放（Parikh 等 2007；Hangya 等 2015）：刺激呈现时学习态，休息时回忆态；
//   平台规则取协方差形式（BTSP 是双向的，Milstein 等 2021）：资格迹减去它自己的滑动平均，随机相遇平均下来互相抵消；
//   学习态里反馈的传递减弱 80%（Hasselmo 的编码 / 提取切换）：学习时按自下而上的输入学，已学到的反馈不会自我强化
export const DEND = {
  ...BAL, inhTypes: true, apical: true, inh: INH_BAL, apicalP: { modI: 250 },
  ffSame: 0.1, ffOther: 0.005, ffScale: 6, ffiP: 0.1, bgUpper: Number(process.env.BG_UP ?? 18), upperBudget: 0.3,
  fbP: 0.1, fbMax: 6, fbRho0: 0.05, fbBudget: 0.15, fbBudgetMode: 'mult', fbRule: 'plateau', fbModSup: 0.8,
  fbPlateau: { tauE: 600, etaP: 0.002, thetaPl: 0.3, cov: true, tauBar: 5000 },
};
const TAG = 'v2';
export const OBJ = { X: ['A', 'D'], Y: ['C', 'B'] };
export const ITEM = 150; // 序列每一项（平衡态里 40 ms 太短：第二层每个神经元只放 0.16 个脉冲）
const log = console.log, at = (ids, hz) => new Map(ids.map((i) => [i, hz]));
const run = (S, ms, drives, learn) => S.net.run(ms, { rate: inputRate(S, drives), learn });
// 相位性神经调质：刺激呈现时学习态，休息时回忆态
const stim = (S, ms, drives) => { S.net.mod = 1; run(S, ms, drives, true); };
const rest = (S, ms) => { S.net.mod = 0; run(S, ms, new Map(), true); };
const grp = (L, k) => protoB(k).map((a) => L.E[a]);
const objIds = (L, o) => OBJ[o].flatMap((p) => grp(L, p));
const ctrl = (L) => L.E.slice(BAL.K * BAL.M);

function cached(file, make) {
  if (fs.existsSync(file)) return JSON.parse(fs.readFileSync(file, 'utf8'));
  const s = make(); fs.mkdirSync('cache', { recursive: true }); fs.writeFileSync(file, JSON.stringify(s)); return s;
}

/** 第一层：与平衡层相同的流程（变体 150 次 → 吸引子；A→D、C→B → 有向通道），学习态 */
export const trainL1 = () => cached(`cache/dend-${TAG}-1.json`, () => {
  const S = buildLayered(1, DEND), L = S.layers[0], rng = makeRng(DEND.seed + 2), t0 = Date.now();
  const others = (k) => [...Array(BAL.NE).keys()].filter((i) => !protoB(k).includes(i));
  learnOnly(S, [L.EE, L.IE]);
  S.net.mod = 0; run(S, 500, new Map(), false);
  for (let n = 0; n < BAL.nLearn; n++) {
    const k = 'ABCD'[rng.pick(4)];
    stim(S, 200, at([...rng.shuffle(protoB(k).slice()).slice(0, 64), ...rng.shuffle(others(k)).slice(0, 16)], STIM));
    rest(S, 300);
    if (n % 50 === 49) log(`  第一层：变体 ${n + 1}/${BAL.nLearn}（${((Date.now() - t0) / 1000).toFixed(0)} s）`);
  }
  for (let n = 0; n < 80; n++) {
    const [p, q] = n % 2 ? ['C', 'B'] : ['A', 'D'];
    stim(S, cfg.seqItemMs, at(protoB(p), SEQ_STIM)); stim(S, cfg.seqItemMs, at(protoB(q), SEQ_STIM)); rest(S, cfg.seqRestMs);
  }
  log(`  第一层训练完成（${((Date.now() - t0) / 1000).toFixed(0)} s）`);
  return saveState(S);
});

/** 第二层：物体阶段（学习态；第二层递归、PV 稳态、反馈都学）→ 序列阶段（只学反馈，第二层递归冻结）。项间隔开，免得前一项的资格迹混进下一项 */
export function trainL2(seq) {
  return cached(`cache/dend-${TAG}-2-${seq ? 'seq' : 'noseq'}.json`, () => {
    const S = buildLayered(2, DEND), [L1, L2] = S.layers, rng = makeRng(DEND.seed + 20), t0 = Date.now();
    loadState(S, trainL1());
    learnOnly(S, [L2.EE, L2.IE, L2.FB, L1.IE]);
    S.net.mod = 0; run(S, 5500, new Map(), false); // 让资格迹的滑动平均先适应自发活动
    for (let n = 0; n < 80; n++) {
      const parts = OBJ[rng.pick(2) ? 'Y' : 'X'];
      stim(S, 200, at(parts.flatMap((p) => rng.shuffle(protoB(p).slice()).slice(0, 48)), STIM));
      rest(S, 800);
      if (n % 20 === 19) log(`  第二层：物体 ${n + 1}/80（${((Date.now() - t0) / 1000).toFixed(0)} s）`);
    }
    if (seq) {
      learnOnly(S, [L2.FB, L1.IE, L2.IE]);
      for (let n = 0; n < 60; n++) {
        stim(S, ITEM, at(OBJ.X.flatMap(protoB), SEQ_STIM));
        stim(S, ITEM, at(OBJ.Y.flatMap(protoB), SEQ_STIM));
        rest(S, 1000);
        if (n % 20 === 19) log(`  第二层：序列 ${n + 1}/60（${((Date.now() - t0) / 1000).toFixed(0)} s）`);
      }
    }
    log(`  第二层训练完成（${((Date.now() - t0) / 1000).toFixed(0)} s）`);
    return saveState(S);
  });
}

export function dendStack({ layers = 2, seq = true, trained = true } = {}) {
  const S = buildLayered(layers, DEND);
  loadState(S, layers === 1 || !trained ? trainL1() : trainL2(seq));
  learnOnly(S, []);
  return S;
}

// ---------- 标定（第一层训练好、第二层未训练） ----------
function calibrate() {
  const S = dendStack({ trained: false }), [L1, L2] = S.layers;
  for (const mod of [0, 1]) {
    S.net.mod = mod;
    S.net.run(500, { rate: inputRate(S), learn: false });
    const t0 = S.net.t; S.net.run(1500, { rate: inputRate(S), learn: false });
    const r = (ids, a = t0, b = t0 + 1500) => mean(spikesIn(S.net, ids, a, b).map((s) => s.length)) / ((b - a) / 1000);
    const sp = `第一层 E ${r(L1.E).toFixed(1)} / SOM ${r(L1.SOM).toFixed(1)} / VIP ${r(L1.VIP).toFixed(1)}，第二层 E ${r(L2.E).toFixed(1)} / PV ${r(L2.PV).toFixed(1)} / SOM ${r(L2.SOM).toFixed(1)} / VIP ${r(L2.VIP).toFixed(1)} Hz`;
    const acc = { A1: 0, A2: 0, B2: 0, vip1: 0 };
    for (let k = 0; k < 4; k++) {
      S.net.run(300, { rate: inputRate(S), learn: false });
      const ts = S.net.t; run(S, 200, at(protoB('A'), STIM), false);
      acc.A1 += r(grp(L1, 'A'), ts + 50, ts + 200) / 4; acc.A2 += r(grp(L2, 'A'), ts + 50, ts + 200) / 4; acc.B2 += r(grp(L2, 'B'), ts + 50, ts + 200) / 4; acc.vip1 += r(L1.VIP, ts + 50, ts + 200) / 4;
    }
    log(`${mod ? '学习态' : '回忆态'} | 自发：${sp} | 驱动第一层 A：第一层 A ${acc.A1.toFixed(1)}、第二层 A ${acc.A2.toFixed(1)}、B ${acc.B2.toFixed(1)}、第一层 VIP ${acc.vip1.toFixed(1)} Hz`);
  }
}

// ---------- 评估（回忆态） ----------
const objOfIdx = (a) => { const b = Math.floor(a / BAL.M); return OBJ.X.includes('ABCD'[b]) ? 'X' : OBJ.Y.includes('ABCD'[b]) ? 'Y' : null; };
/** 反馈的物体级强度：每个第一层神经元从第二层某物体收到的 scale 之和，按第一层物体平均 */
function fbMatrix(S) {
  const [L1, L2] = S.layers, g = L2.FB, pre = new Map(L2.E.map((i, a) => [i, objOfIdx(a)])), post = new Map(L1.E.map((i, a) => [i, objOfIdx(a) ?? 'o']));
  const sum = {}, n = { X: 0, Y: 0, o: 0 };
  for (const o of post.values()) n[o]++;
  for (let s = 0; s < g.pre.length; s++) { const a = pre.get(g.pre[s]), b = post.get(g.post[s]); if (a) sum[a + b] = (sum[a + b] ?? 0) + g.scale[s]; }
  return Object.fromEntries(['XX', 'XY', 'Xo', 'YY', 'YX', 'Yo'].map((k) => [k, +((sum[k] ?? 0) / n[k[1]]).toFixed(1)]));
}

/** 一段试验：返回 20 ms 分箱的放电率、1 ms 的顶树突平均电位、各窗口里出现平台 / 爆发（间隔 < 6 ms）的神经元比例 */
function episode(S, segments, sets, reps) {
  const total = segments.reduce((s, [ms]) => s + ms, 0), nb = Math.ceil(total / 20);
  const out = Object.fromEntries(Object.keys(sets).map((k) => [k, new Float64Array(nb)])), vd = Object.fromEntries(Object.keys(sets).map((k) => [k, new Float64Array(total)]));
  const plat = Object.fromEntries(Object.keys(sets).map((k) => [k, new Float64Array(nb)]));
  const all = [...new Set(Object.values(sets).flat())];
  for (let r = 0; r < reps; r++) {
    S.net.run(400, { rate: inputRate(S), learn: false });
    const ts = S.net.t; let step = 0; const hit = Object.fromEntries(Object.keys(sets).map((k) => [k, Array.from({ length: nb }, () => new Set())]));
    const onStep = () => {
      if (step++ % 10) return; const ms = (step / 10) | 0; if (ms >= total) return;
      for (const [k, ids] of Object.entries(sets)) { vd[k][ms] += mean(ids.map((i) => S.net.Vd[i])) / reps; for (const i of ids) if (S.net.plat[i] > 0.3) hit[k][(ms / 20) | 0].add(i); }
    };
    for (const [ms, d] of segments) S.net.run(ms, { rate: inputRate(S, d), learn: false, onStep });
    for (const [k, ids] of Object.entries(sets)) {
      spikesIn(S.net, ids, ts, ts + total).forEach((s) => s.forEach((t) => (out[k][Math.floor((t - ts) / 20)] += 1000 / (20 * ids.length * reps))));
      for (let b = 0; b < nb; b++) plat[k][b] += hit[k][b].size / ids.length / reps;
    }
  }
  return { out, vd, plat, all };
}
const win = (arr, a, b, bin = 20) => mean(Array.from(arr).slice(a / bin, b / bin));
const fracAny = (arr, a, b) => Math.max(...Array.from(arr).slice(a / 20, b / 20)); // 窗口内某一箱里出现平台的最大比例

function evaluate(S, name) {
  const [L1, L2] = S.layers, r = { name };
  S.net.mod = 0;
  S.net.run(500, { rate: inputRate(S), learn: false });
  const t0 = S.net.t; S.net.run(1500, { rate: inputRate(S), learn: false });
  r.spont = [L1, L2].map((L) => +(mean(spikesIn(S.net, L.E, t0, t0 + 1500).map((s) => s.length)) / 1.5).toFixed(2));
  r.FB = fbMatrix(S);
  const sets = { Y1: objIds(L1, 'Y'), C1: ctrl(L1), X1: objIds(L1, 'X'), Y2: objIds(L2, 'Y'), X2: objIds(L2, 'X') };
  // 只给 X：预测落在第一层 Y 的顶树突（阈下）
  const P = episode(S, [[100, new Map()], [ITEM, at(OBJ.X.flatMap(protoB), SEQ_STIM)], [300, new Map()]], sets, 10);
  const vbase = (k) => win(P.vd[k], 0, 100, 1);
  r.predict = {
    apicalY1minusC1_during: win(P.vd.Y1, 100, 100 + ITEM, 1) - vbase('Y1') - (win(P.vd.C1, 100, 100 + ITEM, 1) - vbase('C1')),
    apicalY1minusC1_after: win(P.vd.Y1, 100 + ITEM, 100 + ITEM + 200, 1) - vbase('Y1') - (win(P.vd.C1, 100 + ITEM, 100 + ITEM + 200, 1) - vbase('C1')),
    rateY1_after: win(P.out.Y1, 100 + ITEM, 100 + ITEM + 200) - win(P.out.Y1, 0, 100), rateC1_after: win(P.out.C1, 100 + ITEM, 100 + ITEM + 200) - win(P.out.C1, 0, 100),
    X2: win(P.out.X2, 100, 100 + ITEM), Y2: win(P.out.Y2, 100, 100 + ITEM),
    vdTraceY1: Array.from({ length: 28 }, (_, b) => +(win(P.vd.Y1, b * 20, b * 20 + 20, 1) - vbase('Y1') - (win(P.vd.C1, b * 20, b * 20 + 20, 1) - vbase('C1'))).toFixed(2)),
  };
  // 匹配：Y 紧跟 X（与训练时一样）/ X 结束后 300 ms / 前面没有 X；看第一层 Y 出现平台的比例、头 20 / 40 ms 的放电
  r.match = [0, ITEM, ITEM + 300].map((on) => {
    const segs = on ? [[100, new Map()], [ITEM, at(OBJ.X.flatMap(protoB), SEQ_STIM)], [on - ITEM, new Map()], [ITEM, at(OBJ.Y.flatMap(protoB), SEQ_STIM)], [100, new Map()]]
      : [[100 + ITEM, new Map()], [ITEM, at(OBJ.Y.flatMap(protoB), SEQ_STIM)], [100, new Map()]];
    const onset = on ? 100 + on : 100 + ITEM;
    const E = episode(S, segs, { Y1: objIds(L1, 'Y') }, 10);
    return { on, plat: fracAny(E.plat.Y1, onset, onset + ITEM), e20: win(E.out.Y1, onset, onset + 20) - win(E.out.Y1, onset - 40, onset), e40: win(E.out.Y1, onset, onset + 40) - win(E.out.Y1, onset - 40, onset), rate: win(E.out.Y1, onset, onset + ITEM) };
  });
  return r;
}

function report(r) {
  log(`\n【${r.name}】自发：第一层 ${r.spont[0]} Hz、第二层 ${r.spont[1]} Hz`);
  log(`  反馈（第二层物体 → 第一层）：X→X ${r.FB.XX}  X→Y ${r.FB.XY}  X→不成组 ${r.FB.Xo}；Y→Y ${r.FB.YY}  Y→X ${r.FB.YX}  Y→不成组 ${r.FB.Yo}`);
  const p = r.predict;
  log(`  只给 X ${ITEM} ms：第二层 X ${p.X2.toFixed(1)} / Y ${p.Y2.toFixed(1)} Hz；第一层 Y 的顶树突比不成组高 X 期间 ${p.apicalY1minusC1_during.toFixed(2)} mV、X 后 0–200 ms ${p.apicalY1minusC1_after.toFixed(2)} mV；X 后第一层 Y 放电变化 ${p.rateY1_after.toFixed(1)} Hz（不成组 ${p.rateC1_after.toFixed(1)}）`);
  log(`    第一层 Y 顶树突减不成组（20 ms 分箱，X 在 100–${100 + ITEM} ms）：${p.vdTraceY1.join(' ')}`);
  for (const m of r.match) log(`  Y ${m.on ? `在 X 开始后 ${m.on} ms` : '前面没有 X'}出现：第一层 Y 出现平台的比例 ${(m.plat * 100).toFixed(1)}%、头 20 / 40 ms 比基线高 ${m.e20.toFixed(1)} / ${m.e40.toFixed(1)} Hz、整段 ${m.rate.toFixed(1)} Hz`);
}

if (import.meta.url === `file:///${process.argv[1].replace(/\\/g, '/')}`) {
  if (process.env.CALIB) calibrate();
  else if (process.env.EVAL) {
    const nets = { two: () => dendStack({ seq: true }), noseq: () => dendStack({ seq: false }), cut: () => { const S = dendStack({ seq: true }); S.layers[1].FB.scale.fill(0); return S; } };
    const names = { two: '2 层', noseq: '2 层，没经历序列', cut: '2 层，切断反馈' };
    for (const k of (process.env.NET ?? 'two,noseq,cut').split(',')) { const r = evaluate(nets[k](), names[k]); report(r); fs.writeFileSync(`scratch/dend-eval-${k}.json`, JSON.stringify(r)); }
  } else if (process.argv[2] === 'l1') trainL1();
  else trainL2(process.argv[2] !== 'noseq');
}
