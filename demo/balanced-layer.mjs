/** 平衡态的单层：网络有自发活动作底、放电异步、对输入的反应分级——旧网络静息为 0、对输入全或无（整组一起齐放 0、1 或 2 次），
 * 一切"保持"都只能靠小群体里很强的递归兴奋，增益略大就失控、略小就保持不住。
 * 结构：400 个兴奋性（A B C D 四组各 80 个 + 80 个不成组，带适应的点神经元）+ 100 个抑制性（快速 GABA_A，彼此互相抑制）；
 *   兴奋性递归连接概率 0.1（每个神经元约 40 个递归输入）；兴奋性与抑制性神经元都有背景输入。
 * 标定（未训练）：自发 E 3.0 Hz、I 9.1 Hz，群体计数 Fano 1.3（异步），A 组随额外输入 +0 → +12 Hz 从 3.2 平滑升到 8.3 Hz，组外从 3.0 降到 2.3（竞争）。
 * 训练与旧第一层相同：变体 150 次 → 吸引子；A→D、C→B 各 40 遍 → 物体内部的有向通道。抑制性可塑性的目标频率 3 Hz（自发水平）。
 * 用法：node demo/balanced-layer.mjs（首次运行先训练约 9 分钟，结果缓存到 cache/bal-1-s<sMax>-e<抑制性可塑性学习率>.json；
 *   BAL_SMAX=6、BAL_ETA=0.05 复现之前的设置） */
import fs from 'fs';
import { cfg, mean } from './lib-hierarchy.mjs';
import { buildLayered, learnOnly, inputRate, spikesIn, saveState, loadState } from '../src/L4-network/layered-network.mjs';
import { makeRng } from '../src/core/rng.mjs';

// sMax：兴奋性递归可学到的上限。6（与旧网络相同）时学到的结构几乎不参与动力学：输入-输出曲线与未训练时相同、补全与方向都不到 1 Hz；
//   测试时把学到的递归放大 1.5 倍出现可用的工作点（补全 +1.9 Hz、方向 +2.5 Hz，自发与异步保住），2 倍时整网变活跃、方向失去特异，2.5 倍才失控
export const BAL = {
  ...cfg, NE: 400, NI: 100, K: 4, M: 80, pConn: 0.1, pEI: 0.2, pIE: 0.4, pII: 0.4, ieKind: 'inhFast',
  sEI: 1.5, sie: 4, sII: 4, bg: 28, bgI: 12, eTarget: 3, ieMax: 40, apical: false,
  sMax: Number(process.env.BAL_SMAX ?? 9),
  // 抑制性可塑性的学习率（上限 9 时扫描；补全 = 没给输入的成员比组外高，方向 = 给 A 后 D 组比对照 B 组高）：
  //   0.05（与旧网络相同）补全 +1.1 Hz、方向 +1.9 Hz——稳态在同一段训练里同步加强抑制，抵消一部分学到的递归增益；
  //   0.03 补全 +1.2、方向 +1.9；0.02 补全 +0.8、方向 +2.9（A→D 0.31、D→A 0.20）；
  //   0.01 网络开始自发地在各吸引子之间游走（放电间隔 CV 0.57 → 1.10，群体 Fano 3.65），补全只剩 +0.3，
  //   训练时各组自发地一起亮，方向被对称地增强抹平（A→D 0.33、D→A 0.31）
  istdpEta: Number(process.env.BAL_ETA ?? 0.02),
};
export const STIM = BAL.bg + 24, SEQ_STIM = BAL.bg + 30;
export const protoB = (L) => Array.from({ length: BAL.M }, (_, i) => 'ABCD'.indexOf(L) * BAL.M + i);
const log = console.log, sd = (x) => { const m = mean(x); return Math.sqrt(mean(x.map((v) => (v - m) ** 2))); };
const at = (ids, hz) => new Map(ids.map((i) => [i, hz]));
const run = (S, ms, drives, learn) => S.net.run(ms, { rate: inputRate(S, drives), learn });

export function trainBal() {
  const file = `cache/bal-1-s${BAL.sMax}-e${BAL.istdpEta}.json`;
  if (fs.existsSync(file)) return JSON.parse(fs.readFileSync(file, 'utf8'));
  const S = buildLayered(1, BAL), L = S.layers[0], rng = makeRng(BAL.seed + 2), t0 = Date.now();
  const others = (k) => [...Array(BAL.NE).keys()].filter((i) => !protoB(k).includes(i));
  learnOnly(S, [L.EE, L.IE]);
  run(S, 500, new Map(), false);
  for (let n = 0; n < BAL.nLearn; n++) {
    const k = 'ABCD'[rng.pick(4)];
    run(S, 200, at([...rng.shuffle(protoB(k).slice()).slice(0, 64), ...rng.shuffle(others(k)).slice(0, 16)], STIM), true);
    run(S, 300, new Map(), true);
    if (n % 50 === 49) log(`  平衡层：变体 ${n + 1}/${BAL.nLearn}（${((Date.now() - t0) / 1000).toFixed(0)} s）`);
  }
  for (let n = 0; n < 80; n++) {
    const [p, q] = n % 2 ? ['C', 'B'] : ['A', 'D'];
    run(S, cfg.seqItemMs, at(protoB(p), SEQ_STIM), true);
    run(S, cfg.seqItemMs, at(protoB(q), SEQ_STIM), true);
    run(S, cfg.seqRestMs, new Map(), true);
  }
  log(`  平衡层训练完成（${((Date.now() - t0) / 1000).toFixed(0)} s）`);
  const state = saveState(S);
  fs.mkdirSync('cache', { recursive: true });
  fs.writeFileSync(file, JSON.stringify(state));
  return state;
}
export function balStack(trained = true) { const S = buildLayered(1, BAL); if (trained) loadState(S, trainBal()); learnOnly(S, []); return S; }

export function spontaneous(S, ms = 2000) {
  const L = S.layers[0];
  S.net.run(500, { rate: inputRate(S), learn: false });
  const t0 = S.net.t; S.net.run(ms, { rate: inputRate(S), learn: false });
  const sE = spikesIn(S.net, L.E, t0, t0 + ms), sI = spikesIn(S.net, L.I, t0, t0 + ms);
  const cvs = sE.filter((s) => s.length >= 4).map((s) => { const isi = s.slice(1).map((t, k) => t - s[k]); return sd(isi) / mean(isi); });
  const bins = new Float64Array(ms / 5); for (const s of sE) for (const t of s) bins[Math.floor((t - t0) / 5)]++;
  const b = Array.from(bins);
  return { rE: mean(sE.map((s) => s.length)) / (ms / 1000), rI: mean(sI.map((s) => s.length)) / (ms / 1000), cv: cvs.length ? mean(cvs) : null, fano: mean(b) > 0 ? sd(b) ** 2 / mean(b) : 0 };
}

/** A 组全部 80 个额外输入 d Hz：50–200 ms 的 A 组频率（逐次试验）与组外 */
export function transfer(S, deltas = [0, 4, 8, 16, 24], reps = 4) {
  const L = S.layers[0], A = protoB('A').map((a) => L.E[a]), others = L.E.filter((i) => !A.includes(i));
  return deltas.map((d) => {
    const tr = []; let out = 0;
    for (let k = 0; k < reps; k++) {
      S.net.run(300, { rate: inputRate(S), learn: false });
      const ts = S.net.t;
      S.net.run(200, { rate: inputRate(S, at(protoB('A'), BAL.bg + d)), learn: false });
      tr.push(mean(spikesIn(S.net, A, ts + 50, ts + 200).map((s) => s.length)) / 0.15);
      out += mean(spikesIn(S.net, others, ts + 50, ts + 200).map((s) => s.length)) / 0.15 / reps;
    }
    return { d, A: mean(tr), sdA: sd(tr), out };
  });
}

/** 补全：只给 A 的 32 个（40%）150 ms；没给输入的 48 个、组外的频率；撤掉后 0–150 ms A 组还剩多少 */
export function completion(S, reps = 6) {
  const L = S.layers[0], rng = makeRng(4), A = protoB('A'), others = L.E.filter((i) => !A.map((a) => L.E[a]).includes(i));
  const r = { uncued: 0, out: 0, after: 0 };
  for (let k = 0; k < reps; k++) {
    S.net.run(300, { rate: inputRate(S), learn: false });
    const cue = rng.shuffle(A.slice()).slice(0, 32), un = A.filter((a) => !cue.includes(a)).map((a) => L.E[a]), ts = S.net.t;
    S.net.run(150, { rate: inputRate(S, at(cue, STIM)), learn: false });
    S.net.run(150, { rate: inputRate(S), learn: false });
    const rt = (ids, a, b) => mean(spikesIn(S.net, ids, ts + a, ts + b).map((s) => s.length)) / ((b - a) / 1000) / reps;
    r.uncued += rt(un, 0, 150); r.out += rt(others, 0, 150); r.after += rt(A.map((a) => L.E[a]), 150, 300);
  }
  return r;
}

/** 有向通道：A 组 40 ms；40–160 ms D 组与对照 B 组的频率 */
export function channel(S, reps = 8) {
  const L = S.layers[0], ids = (k) => protoB(k).map((a) => L.E[a]);
  let D = 0, B = 0;
  for (let k = 0; k < reps; k++) {
    S.net.run(300, { rate: inputRate(S), learn: false });
    const ts = S.net.t;
    S.net.run(cfg.seqItemMs, { rate: inputRate(S, at(protoB('A'), SEQ_STIM)), learn: false });
    S.net.run(120, { rate: inputRate(S), learn: false });
    D += mean(spikesIn(S.net, ids('D'), ts + 40, ts + 160).map((s) => s.length)) / 0.12 / reps;
    B += mean(spikesIn(S.net, ids('B'), ts + 40, ts + 160).map((s) => s.length)) / 0.12 / reps;
  }
  return { D, B };
}

function weights(S) {
  const L = S.layers[0], g = L.EE, blk = (a) => Math.min(Math.floor(a / BAL.M), BAL.K), loc = new Map(L.E.map((i, a) => [i, blk(a)])), acc = {};
  for (let s = 0; s < g.pre.length; s++) { const a = loc.get(g.pre[s]), b = loc.get(g.post[s]); if (a > 3 || b > 3) continue; const key = a === b ? 'in' : `${'ABCD'[a]}${'ABCD'[b]}`; (acc[key] ??= []).push(g.rho[s]); }
  return { within: mean(acc.in), between: mean(Object.entries(acc).filter(([k]) => k !== 'in').flatMap(([, v]) => v)), AD: mean(acc.AD), DA: mean(acc.DA), CB: mean(acc.CB), BC: mean(acc.BC) };
}

if (import.meta.url === `file:///${process.argv[1].replace(/\\/g, '/')}`) {
  for (const [name, trained] of [['未训练', false], ['训练后', true]]) {
    const S = balStack(trained), sp = spontaneous(S), tf = transfer(S), c = completion(S), ch = channel(S), w = weights(S);
    log(`\n【平衡层，${name}】`);
    log(`  连接效能 ρ：组内 ${w.within.toFixed(3)}、组间 ${w.between.toFixed(3)}；A→D ${w.AD.toFixed(3)} / D→A ${w.DA.toFixed(3)}，C→B ${w.CB.toFixed(3)} / B→C ${w.BC.toFixed(3)}；抑制→兴奋平均强度 ${mean(Array.from(S.layers[0].IE.scale)).toFixed(2)}（初值 ${BAL.sie}）`);
    log(`  静息：E ${sp.rE.toFixed(2)} Hz、I ${sp.rI.toFixed(1)} Hz，放电间隔 CV ${sp.cv?.toFixed(2)}，群体 Fano ${sp.fano.toFixed(2)}`);
    log(`  输入-输出（A 组额外输入 +0 / +4 / +8 / +16 / +24 Hz）：A ${tf.map((x) => `${x.A.toFixed(1)}±${x.sdA.toFixed(1)}`).join(' / ')} Hz；组外 ${tf.map((x) => x.out.toFixed(1)).join(' / ')} Hz`);
    log(`  补全（只给 A 的 32 个）：没给输入的 48 个 ${c.uncued.toFixed(1)} Hz，组外 ${c.out.toFixed(1)} Hz；撤掉后 150 ms 内 A 组 ${c.after.toFixed(1)} Hz（自发 ${sp.rE.toFixed(1)}）`);
    log(`  有向通道（A 40 ms，40–160 ms）：D 组 ${ch.D.toFixed(1)} Hz，对照 B 组 ${ch.B.toFixed(1)} Hz`);
  }
}
