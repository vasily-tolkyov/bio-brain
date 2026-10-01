/** 外界刺激下的脉冲网络（第 1–3 层的神经元、突触、钙规则可塑性 + 抑制性可塑性 + 异突触竞争）：
 * 一、4 组"相似但不同"的刺激（原型 = 20 个神经元，每次只呈现变体：原型里随机 16 个 + 4 个无关神经元，原型本身从不呈现）
 *     → 特征性激活、吸引子是否涌现（补全、维持各模式所需的最少外部驱动 = 能量）
 * 二、相邻刺激组按固定顺序连续出现 A→B→C→D → 组间是否长出有向通道（正/反向突触、只给 A 时活动是否顺序传下去）
 * 用法：node demo/fig2-network.mjs → figures/fig2-*.png，并打印全部数字 */
import fs from 'fs';
import { SpikingNetwork } from '../src/L4-network/spiking-network.mjs';
import { GB_NMDA } from '../src/L3-plasticity/calcium-rule.mjs';
import { mammalianCosts } from '../src/L1-neuron/hh.mjs';
import { makeRng } from '../src/core/rng.mjs';
import { Canvas, heat, diverging, PALETTE } from '../src/core/png.mjs';

export const CFG = {
  NE: 100, NI: 25, K: 4, M: 20, stim: 18, bg: Number(process.env.BG ?? 6), pConn: 0.5, contacts: 10, sMax: 6,
  rho0: Number(process.env.RHO0 ?? 0.1), budget: Number(process.env.BUDGET ?? 0.2), sie: 8, istdpEta: 0.05,
  rule: GB_NMDA, nLearn: 150, nSeq: 40, nEnergy: Number(process.env.NENERGY ?? 4), seed: 1,
  seqItemMs: Number(process.env.SEQ_ITEM ?? 40), seqStim: Number(process.env.SEQ_STIM ?? 30), seqRestMs: 400,
};

export function buildNetwork(cfg = CFG) {
  const costs = mammalianCosts();
  const net = new SpikingNetwork({ costs, seed: cfg.seed, plasticity: cfg.rule, pops: [{ name: 'E', n: cfg.NE, p: { b: 20, tauW: 300 } }, { name: 'I', n: cfg.NI, p: { C: 100, tRef: 1 } }] });
  const EE = net.connect({ from: 'E', to: 'E', p: cfg.pConn, contacts: cfg.contacts, pRelease: 0.3, tauRec: 100, plastic: { sMin: 0.1, sMax: cfg.sMax, rho0: cfg.rho0, accel: 5, budget: cfg.budget } });
  net.connect({ from: 'E', to: 'I', p: 0.5, contacts: 10, pRelease: 0.3, scale: 1.5 });
  const IE = net.connect({ from: 'I', to: 'E', p: 0.5, contacts: 10, pRelease: 0.5, tauRec: 300, kind: 'inh', scale: cfg.sie, istdp: { eta: cfg.istdpEta, target: 5 } });
  return { net, EE, IE, costs };
}

export function runExperiment(cfg = CFG, { figures = true, log = console.log } = {}) {
  const { net, EE } = buildNetwork(cfg);
  const { NE, K, M } = cfg;
  const rng = makeRng(cfg.seed + 2);
  const proto = Array.from({ length: K }, (_, k) => Array.from({ length: M }, (_, i) => k * M + i));
  const groupOf = (i) => (i < K * M ? Math.floor(i / M) : -1);
  const variant = (k) => new Set([...rng.shuffle(proto[k].slice()).slice(0, 16), ...rng.shuffle([...Array(NE).keys()].filter((i) => groupOf(i) !== k)).slice(0, 4)]);
  const drive = (set, r) => (i) => (i < NE ? (set.has(i) ? r : cfg.bg) : 0);
  const quiet = (ms, learn = false) => net.run(ms, { rate: drive(new Set(), 0), learn });
  const mean = (x) => x.reduce((s, v) => s + v, 0) / (x.length || 1);

  const snapshot = () => { const W = Array.from({ length: NE }, () => new Float64Array(NE)); for (let k = 0; k < EE.pre.length; k++) W[EE.post[k]][EE.pre[k]] = EE.rho[k]; return W; };
  const blockMean = (W, a, b) => { const v = []; for (const i of proto[b]) for (const j of proto[a]) if (i !== j && EE.pre.length) v.push(W[i][j]); return mean(v); };
  const connected = new Set(); for (let k = 0; k < EE.pre.length; k++) connected.add(EE.post[k] * NE + EE.pre[k]);
  const blockMeanC = (W, a, b) => { const v = []; for (const i of proto[b]) for (const j of proto[a]) if (connected.has(i * NE + j)) v.push(W[i][j]); return mean(v); };

  /** 一组测试呈现：返回每次的放电计数与补全率；raster 收集放电时刻 */
  function respond(label, nPer = 2) {
    const out = [], raster = [];
    for (let k = 0; k < K; k++) for (let r = 0; r < nPer; r++) {
      const v = variant(k), t0 = net.t;
      net.run(200, { rate: drive(v, cfg.stim), learn: false });
      const cnt = net.counts(t0, net.t, [...Array(NE).keys()]);
      raster.push({ k, v: [...v], t0, spikes: [...Array(NE).keys()].map((i) => net.spikes[i].filter((s) => s >= t0 && s < t0 + 200).map((s) => s - t0)) });
      quiet(300);
      const unstim = proto[k].filter((i) => !v.has(i));
      out.push({ k, cnt, comp: unstim.filter((i) => cnt[i] >= 2).length / unstim.length, others: cnt.reduce((s, c, i) => s + (v.has(i) || groupOf(i) === k ? 0 : c), 0) });
    }
    const corr = (a, b) => { const n = a.length, ma = mean(a), mb = mean(b); let x = 0, y = 0, z = 0; for (let i = 0; i < n; i++) { x += (a[i] - ma) * (b[i] - mb); y += (a[i] - ma) ** 2; z += (b[i] - mb) ** 2; } return x / Math.sqrt(y * z || 1); };
    const wi = [], bw = [];
    for (let a = 0; a < out.length; a++) for (let b = a + 1; b < out.length; b++) (out[a].k === out[b].k ? wi : bw).push(corr(out[a].cnt, out[b].cnt));
    const res = { within: mean(wi), between: mean(bw), completion: mean(out.map((o) => o.comp)), others: mean(out.map((o) => o.others)), raster };
    log(`${label}：同组变体响应相关 ${res.within.toFixed(2)}，异组 ${res.between.toFixed(2)}；没被刺激的原型成员被拉起 ${(res.completion * 100).toFixed(0)}%；组外神经元放电 ${res.others.toFixed(1)} 次/呈现`);
    return res;
  }

  /** 只给部分线索（原型里 8 个神经元），看原型其余 12 个是否被拉起 */
  function partialCue(label) {
    const fr = [];
    for (let k = 0; k < K; k++) {
      const cue = new Set(rng.shuffle(proto[k].slice()).slice(0, 8)), t0 = net.t;
      net.run(200, { rate: drive(cue, cfg.stim), learn: false });
      const cnt = net.counts(t0, net.t, [...Array(NE).keys()]);
      quiet(300);
      fr.push(proto[k].filter((i) => !cue.has(i)).filter((i) => cnt[i] >= 2).length / 12);
    }
    log(`${label}：只刺激原型的 8 个神经元，其余 12 个被拉起 ${(mean(fr) * 100).toFixed(0)}%`);
    return mean(fr);
  }

  /** 维持一个模式所需的最少外部驱动（每条输入的放电率，Hz）与此时整个系统的 ATP——动力学账与代谢账 */
  function holdCost(set) {
    const ids = [...set];
    const meanRate = (r) => { quiet(200); net.ledger.reset(); const t0 = net.t; net.run(400, { rate: drive(set, r), learn: false }); const c = net.counts(t0 + 100, net.t, ids); return { rate: mean(c) / 0.3, atp: net.ledger.total() / 0.4 }; };
    let lo = 0, hi = 40;
    for (let k = 0; k < 6; k++) { const mid = (lo + hi) / 2; if (meanRate(mid).rate >= 10) hi = mid; else lo = mid; }
    return { drive: hi, atp: meanRate(hi).atp };
  }
  const learnSeq = Array.from({ length: cfg.nLearn }, () => { const k = rng.pick(K); return { k, v: variant(k) }; });
  function energyTable(label) {
    const rows = { 原型: [], 经历过的变体: [], 随机: [] };
    for (let k = 0; k < cfg.nEnergy; k++) {
      rows['原型'].push(holdCost(new Set(proto[k])));
      rows['经历过的变体'].push(holdCost(learnSeq.find((x) => x.k === k).v));
      rows['随机'].push(holdCost(new Set(rng.shuffle([...Array(NE).keys()]).slice(0, M))));
    }
    const summary = Object.fromEntries(Object.entries(rows).map(([n, v]) => [n, { drive: mean(v.map((x) => x.drive)), atp: mean(v.map((x) => x.atp)) }]));
    log(`${label}：维持 20 个神经元放到 10 Hz 所需的最少外部驱动 —— ${Object.entries(summary).map(([n, s]) => `${n} ${s.drive.toFixed(1)} Hz（${s.atp.toExponential(2)} ATP/s）`).join('，')}`);
    return summary;
  }

  /** 只给 A 组一个序列项的时长（与训练时相同），之后撤掉，按 10 ms 分箱记录各组平均放电率；重复 reps 次取平均 */
  /** 有向通道的功能检验：持续刺激某一组 150 ms，量它的下一组与上一组被拉起多少（Hz）。
   * 有向通道 = 只拉下一组（A→B、B→C、C→D），不拉上一组（B→A、C→B、D→C）。栅格取刺激 A 的那一次 */
  function sequenceProbe(label, reps = 3) {
    const rateOf = (k, t0, t1) => mean(net.counts(t0, t1, proto[k])) / ((t1 - t0) / 1000);
    const fwd = [], bwd = [], chain = [];
    let raster = null;
    for (let r = 0; r < reps; r++) for (let k = 0; k < K; k++) {
      quiet(400);
      const t0 = net.t;
      net.run(150, { rate: drive(new Set(proto[k]), cfg.stim), learn: false });
      if (k + 1 < K) fwd.push(rateOf(k + 1, t0, net.t));
      if (k - 1 >= 0) bwd.push(rateOf(k - 1, t0, net.t));
      if (k === 0) chain.push(rateOf(2, t0, net.t), rateOf(3, t0, net.t));
      if (k === 0 && r === 0) { net.run(100, { rate: drive(new Set(), 0), learn: false }); raster = [...Array(NE).keys()].map((i) => net.spikes[i].filter((s) => s >= t0 && s < t0 + 250).map((s) => s - t0)); }
    }
    const res = { fwd: mean(fwd), bwd: mean(bwd), skip: mean(chain), raster };
    log(`${label}：持续刺激一组 150 ms 时，下一组被拉起 ${res.fwd.toFixed(1)} Hz，上一组 ${res.bwd.toFixed(1)} Hz（刺激 A 时 C、D 平均 ${res.skip.toFixed(1)} Hz）`);
    return res;
  }

  const tWall = Date.now();
  const IEg = net.groups.find((g) => g.istdp);
  let W0, before, W1, after1;
  if (process.env.LOAD && fs.existsSync(process.env.LOAD)) {
    const s = JSON.parse(fs.readFileSync(process.env.LOAD, 'utf8'));
    EE.rho.set(s.rho); for (let k = 0; k < EE.rho.length; k++) EE.scale[k] = 0.1 + (cfg.sMax - 0.1) * EE.rho[k];
    IEg.scale.set(s.ie); W0 = s.W0; before = s.before; after1 = s.after1;
    quiet(500);
    W1 = snapshot();
    log(`（读入第一阶段结束时的突触状态：${process.env.LOAD}）`);
    after1.seq = sequenceProbe('学习后、序列训练前');
  } else {
    quiet(500);
    W0 = snapshot();
    before = { resp: respond('学习前'), cue: partialCue('学习前'), energy: energyTable('学习前'), seq: sequenceProbe('学习前（序列）') };
    // 第一阶段：变体随机顺序反复出现
    for (let n = 0; n < cfg.nLearn; n++) { net.run(200, { rate: drive(learnSeq[n].v, cfg.stim), learn: true }); quiet(300, true); if (n % 50 === 49) log(`  第一阶段第 ${n + 1} 次呈现，墙钟 ${((Date.now() - tWall) / 1000).toFixed(0)} s`); }
    W1 = snapshot();
    after1 = { resp: respond('学习后'), cue: partialCue('学习后'), energy: energyTable('学习后'), seq: sequenceProbe('学习后、序列训练前') };
    if (process.env.SAVE) fs.writeFileSync(process.env.SAVE, JSON.stringify({ rho: [...EE.rho], ie: [...IEg.scale], W0: W0.map((r) => [...r]), before, after1 }));
  }

  // 第二阶段：相邻刺激组按 A→B→C→D 紧接着出现（时间压缩：每组 seqItemMs，像海马 θ 周期里依次放电的位置细胞）
  for (let n = 0; n < cfg.nSeq; n++) { for (let k = 0; k < K; k++) net.run(cfg.seqItemMs, { rate: drive(new Set(proto[k]), cfg.seqStim), learn: true }); quiet(cfg.seqRestMs, true); if (n % 20 === 19) log(`  第二阶段第 ${n + 1} 遍序列，墙钟 ${((Date.now() - tWall) / 1000).toFixed(0)} s`); }
  const W2 = snapshot();
  const seqAfter = sequenceProbe('序列训练后');

  const blocks = (W) => Array.from({ length: K }, (_, a) => Array.from({ length: K }, (_, b) => blockMeanC(W, a, b)));
  const B0 = blocks(W0), B1 = blocks(W1), B2 = blocks(W2);
  const fmtB = (B) => B.map((row, a) => row.map((v, b) => `${'ABCD'[a]}→${'ABCD'[b]} ${v.toFixed(2)}`).join(' ')).join(' | ');
  log(`组块平均效能 ρ（突触前组→突触后组）\n  学习前：${fmtB(B0)}\n  第一阶段后：${fmtB(B1)}\n  第二阶段后：${fmtB(B2)}`);
  const fwd = [B2[0][1], B2[1][2], B2[2][3]], bwd = [B2[1][0], B2[2][1], B2[3][2]];
  const fwd1 = [B1[0][1], B1[1][2], B1[2][3]], bwd1 = [B1[1][0], B1[2][1], B1[3][2]];
  log(`相邻组之间：序列训练前 正向 ${mean(fwd1).toFixed(3)} 反向 ${mean(bwd1).toFixed(3)}；序列训练后 正向 ${mean(fwd).toFixed(3)} 反向 ${mean(bwd).toFixed(3)}`);
  log(`墙钟 ${((Date.now() - tWall) / 1000).toFixed(0)} s，模拟 ${(net.t / 1000).toFixed(0)} s`);

  if (figures) drawFigures({ cfg, proto, groupOf, W0, W1, W2, before, after1, seqAfter });
  return { before, after1, seqAfter, B0, B1, B2, fwd, bwd, fwd1, bwd1 };
}

function drawFigures({ cfg, groupOf, W0, W1, W2, before, after1, seqAfter }) {
  fs.mkdirSync('figures', { recursive: true });
  const { NE } = cfg;
  const colorOf = (i) => (groupOf(i) >= 0 ? PALETTE[groupOf(i)] : [90, 90, 90]);
  // 图 2：对各组变体的放电栅格（学习前 / 学习后），每个变体 200 ms，被刺激的神经元底色标黄
  const rasterFig = (r1, r2, file) => {
    const cw = 2, rowH = 3, gap = 30, panelW = r1.raster.length * (200 * cw / 2 + 10), cv = new Canvas(panelW + 120, NE * rowH * 2 + gap * 3 + 40);
    [r1, r2].forEach((res, pi) => {
      const y0 = 40 + pi * (NE * rowH + gap);
      cv.text(10, y0 - 26, pi === 0 ? 'BEFORE LEARNING' : 'AFTER LEARNING', [0, 0, 0], 2);
      res.raster.forEach((pres, n) => {
        const x0 = 100 + n * (200 * cw / 2 + 10);
        for (const i of pres.v) cv.rect(x0, y0 + i * rowH, 200, rowH, [255, 240, 180]);
        for (let i = 0; i < NE; i++) for (const s of pres.spikes[i]) cv.rect(x0 + s, y0 + i * rowH, 2, rowH, colorOf(i));
        cv.rect(x0, y0 + NE * rowH + 2, 200, 4, PALETTE[pres.k]);
      });
      for (let g = 0; g < 4; g++) cv.text(70, y0 + g * cfg.M * rowH + 20, 'ABCD'[g], PALETTE[g], 2);
    });
    cv.save(file);
  };
  rasterFig(before.resp, after1.resp, 'figures/fig2-responses.png');
  // 图 3：兴奋性突触效能矩阵（行 = 突触后，列 = 突触前，按组排序）；第三张是第二阶段后减去第一阶段后（红 = 变弱，蓝 = 变强）
  const px = 3, cv = new Canvas(3 * (NE * px + 40) + 40, NE * px + 80);
  [W0, W1].forEach((W, m) => { const x0 = 20 + m * (NE * px + 40); cv.text(x0, 10, m === 0 ? 'START' : 'AFTER VARIANTS', [0, 0, 0], 2); for (let i = 0; i < NE; i++) for (let j = 0; j < NE; j++) cv.rect(x0 + j * px, 40 + i * px, px, px, heat(W[i][j])); });
  const x2 = 20 + 2 * (NE * px + 40);
  cv.text(x2, 10, 'SEQUENCE - BEFORE', [0, 0, 0], 2);
  for (let i = 0; i < NE; i++) for (let j = 0; j < NE; j++) cv.rect(x2 + j * px, 40 + i * px, px, px, diverging((W2[i][j] - W1[i][j]) * 2));
  cv.save('figures/fig3-weights.png');
  // 图 4：只给 A 后的栅格（序列训练前 / 后）
  const cv4 = new Canvas(620, NE * 3 * 2 + 110), sx = 2;
  [after1.seq, seqAfter].forEach((res, pi) => {
    const y0 = 40 + pi * (NE * 3 + 40);
    cv4.text(10, y0 - 26, pi === 0 ? 'DRIVE A 150 MS - BEFORE SEQUENCE' : 'DRIVE A 150 MS - AFTER SEQUENCE', [0, 0, 0], 2);
    cv4.rect(60, y0, 150 * sx, NE * 3, [255, 240, 180]);
    for (let i = 0; i < NE; i++) for (const s of res.raster[i]) cv4.rect(60 + s * sx, y0 + i * 3, 2, 3, colorOf(i));
    for (let g = 0; g < 4; g++) cv4.text(30, y0 + g * cfg.M * 3 + 20, 'ABCD'[g], PALETTE[g], 2);
  });
  cv4.save('figures/fig4-sequence.png');
}

if (import.meta.url === `file:///${process.argv[1].replace(/\\/g, '/')}`) runExperiment();
