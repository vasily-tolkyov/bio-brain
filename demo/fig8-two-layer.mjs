/** 两层脉冲网络的迭代激活：后一层以前一层被激活的吸引子"按激活强度绘图"作为刺激。
 * 第一层 = 训练好的网络（吸引子 A B C D + 有向通道 A→B→C→D，见 demo/fig2-network.mjs）。
 * 第一层对刺激的响应 → 读出各吸引子的激活度（0–1）→ 画成第二层的刺激：每个第一层吸引子对应第二层的 20 个神经元，
 *   它们的输入速率 = 背景 + 激活度 × (最大 − 背景)。
 * 第二层 = 全新的同构网络（同样的神经元、突触、学习规则），只经历两个"物体"：X = A + C，Y = B + D。
 * 看：第二层是否长出"吸引子的组合"这一级吸引子；只给物体的一个部件时能否补全另一个；没学过的组合（A + D）怎样；
 *   第一层内部的细节（具体哪些神经元被刺激）在"绘图"这一步被丢弃。
 * 用法：node demo/fig8-two-layer.mjs → figures/fig8-two-layer.png；verify/L4-two-layer.mjs 调用 runTwoLayer */
import fs from 'fs';
import { buildNetwork } from './fig2-network.mjs';
import { cfg, NE, mean, proto, trainedNetwork, discoverAttractors, labelAttractors, applyStim } from './lib-trained.mjs';
import { makeRng } from '../src/core/rng.mjs';
import { Canvas, PALETTE, heat } from '../src/core/png.mjs';

const R_MAX = 30, N_TRAIN = 100, POOL = 10;
const clamp01 = (v) => Math.max(0, Math.min(1, v));
const corr = (a, b) => { const n = a.length, ma = mean(a), mb = mean(b); let x = 0, y = 0, z = 0; for (let i = 0; i < n; i++) { x += (a[i] - ma) * (b[i] - mb); y += (a[i] - ma) ** 2; z += (b[i] - mb) ** 2; } return x / Math.sqrt(y * z || 1); };
const TESTS = [
  ['X 完整（A + C）', ['A', 'C'], 'X = A+C'], ['Y 完整（B + D）', ['B', 'D'], 'Y = B+D'],
  ['只给 A', ['A'], 'ONLY A'], ['只给 C', ['C'], 'ONLY C'], ['只给 B', ['B'], 'ONLY B'], ['只给 D', ['D'], 'ONLY D'],
  ['没学过的组合 A + D', ['A', 'D'], 'NEW A+D'], ['没学过的组合 B + C', ['B', 'C'], 'NEW B+C'],
];

export function runTwoLayer({ log = console.log, figures = true } = {}) {
  const t0 = Date.now(), el = () => `${((Date.now() - t0) / 1000).toFixed(0)} s`;

  // ── 第一层：训练好的网络、它自己的吸引子、激活度标定 ──
  const L1 = trainedNetwork(log);
  const att1 = discoverAttractors(L1.EE), names = labelAttractors(att1).map((p) => p.label);
  log(`第一层的吸引子（只看学到的突触找到）：${att1.map((g, a) => `${names[a]}（${g.length} 个神经元）`).join('，')}`);
  const rest1 = applyStim(L1.net, new Map());
  const cal1 = att1.map((g) => { const full = applyStim(L1.net, new Map(g.map((j) => [j, cfg.stim]))); return { rest: mean(g.map((i) => rest1[i])), full: mean(g.map((i) => full[i])) }; });
  const levels1 = (rates) => att1.map((g, a) => (mean(g.map((i) => rates[i])) - cal1[a].rest) / (cal1[a].full - cal1[a].rest));
  const respond1 = (drives, reps = 2) => { const rates = applyStim(L1.net, drives, { reps }); return { rates, lv: levels1(rates) }; };

  // ── "绘图"：第一层吸引子 a 的激活度 → 第二层第 a 组 20 个神经元的输入速率 ──
  const grp2 = att1.map((_, a) => proto[a]);
  const draw = (lv) => new Map(grp2.flatMap((g, a) => g.map((i) => [i, cfg.bg + clamp01(lv[a]) * (R_MAX - cfg.bg)])));

  // ── 物体：两个第一层吸引子同时出现。刺激按实验者定义的原型构造（每个部件随机取 12 个成员） ──
  const rng = makeRng(31), choice = (arr) => arr[rng.pick(arr.length)];
  const P = (L) => proto['ABCD'.indexOf(L)];
  const stimOf = (parts, n = 12) => new Map(parts.flatMap((L) => rng.shuffle(P(L).slice()).slice(0, n).map((i) => [i, cfg.stim])));
  const OBJ = { X: ['A', 'C'], Y: ['B', 'D'] };
  const pool = Object.fromEntries(Object.entries(OBJ).map(([o, parts]) => [o, Array.from({ length: POOL }, () => respond1(stimOf(parts), 1).lv)]));
  const objLevels = Object.fromEntries(Object.keys(OBJ).map((o) => [o, names.map((_, a) => mean(pool[o].map((l) => l[a])))]));
  for (const o of Object.keys(OBJ)) log(`物体 ${o} = ${OBJ[o].join(' + ')}：第一层各吸引子的平均激活度 ${names.map((n, a) => `${n} ${objLevels[o][a].toFixed(2)}`).join('  ')}（${el()}）`);

  // ── 第二层：全新的同构网络，只在物体上学习；同一初始状态、不学习的拷贝作对照 ──
  const L2 = buildNetwork({ ...cfg, seed: 2 }), naive = buildNetwork({ ...cfg, seed: 2 });
  const bg = (i) => (i < NE ? cfg.bg : 0);
  L2.net.run(500, { rate: bg, learn: false });
  for (let n = 0; n < N_TRAIN; n++) {
    const d = draw(choice(pool[choice(['X', 'Y'])]));
    L2.net.run(200, { rate: (i) => (i < NE ? d.get(i) ?? cfg.bg : 0), learn: true });
    L2.net.run(300, { rate: bg, learn: true });
    if (n % 25 === 24) log(`  第二层学习 ${n + 1}/${N_TRAIN}（${el()}）`);
  }
  const att2 = discoverAttractors(L2.EE);
  const comp = att2.map((g) => names.map((_, a) => g.filter((i) => grp2[a].includes(i)).length));
  const otherOf = (c) => att2[c].length - comp[c].reduce((s, k) => s + k, 0);
  log(`第二层学出的吸引子：${att2.map((g, c) => `${g.length} 个神经元 = ${comp[c].map((k, a) => (k ? `${names[a]} 组 ${k}` : '')).filter(Boolean).join(' + ')}${otherOf(c) ? ` + 其他 ${otherOf(c)}` : ''}`).join('；')}`);

  // ── 测试（不再学习）：第一层刺激 → 第一层激活度 → 画成第二层刺激 → 第二层各组放电 ──
  const grpRate = (rates) => grp2.map((g) => mean(g.map((i) => rates[i])));
  const rows = [];
  for (const [name, parts, short] of TESTS) {
    const { lv } = respond1(stimOf(parts));
    const d = draw(lv);
    const rt = grpRate(applyStim(L2.net, d, { reps: 2 })), rn = grpRate(applyStim(naive.net, d, { reps: 2 }));
    rows.push({ name, parts, short, lv, rt, rn });
    log(`\n【${name}】第一层激活度 ${names.map((n, a) => `${n} ${lv[a].toFixed(2)}`).join('  ')}\n  第二层各组放电（Hz，学过 / 没学过）：${names.map((n, a) => `${n} 组 ${rt[a].toFixed(1)} / ${rn[a].toFixed(1)}`).join('  ')}`);
  }

  // ── 细节在"绘图"这一步被丢弃：同一个吸引子的两种不重叠的刺激（A 的前 10 个成员 vs 后 10 个） ──
  const A = P('A'), half1 = new Map(A.slice(0, 10).map((i) => [i, cfg.stim])), half2 = new Map(A.slice(10).map((i) => [i, cfg.stim]));
  const r1 = respond1(half1, 3), r2 = respond1(half2, 3);
  const s1 = applyStim(L2.net, draw(r1.lv), { reps: 3 }), s2 = applyStim(L2.net, draw(r2.lv), { reps: 3 });
  const detail = { stim: corr([...Array(NE).keys()].map((i) => +half1.has(i)), [...Array(NE).keys()].map((i) => +half2.has(i))), L1: corr(r1.rates, r2.rates), drawn: corr(r1.lv, r2.lv), L2: corr(s1, s2) };
  log(`\n同一个吸引子 A 的两种互不重叠的刺激：刺激本身相关 ${detail.stim.toFixed(2)} → 第一层放电相关 ${detail.L1.toFixed(2)} → 画出的激活度相关 ${detail.drawn.toFixed(2)} → 第二层放电相关 ${detail.L2.toFixed(2)}`);

  const result = { names, objLevels, comp, rows, detail };
  if (figures) {
    // 图 8：左 = 第二层学到的 E→E 效能（按 A C B D 组排序）；右 = 每个测试的第一层激活度与第二层各组放电
    fs.mkdirSync('figures', { recursive: true });
    const cv = new Canvas(1500, 640);
    const order = ['A', 'C', 'B', 'D'].flatMap((L) => grp2[names.indexOf(L)]);
    const ord = [...order, ...[...Array(NE).keys()].filter((i) => !order.includes(i))], pos = new Map(ord.map((i, k) => [i, k]));
    const R = Array.from({ length: NE }, () => new Float64Array(NE).fill(-1));
    for (let k = 0; k < L2.EE.pre.length; k++) { const a = L2.EE.pre[k], b = L2.EE.post[k]; if (a < NE && b < NE) R[pos.get(b)][pos.get(a)] = L2.EE.rho[k]; }
    cv.text(20, 15, 'LAYER 2 WEIGHTS  ORDER A C B D', [0, 0, 0], 2);
    for (let i = 0; i < NE; i++) for (let j = 0; j < NE; j++) cv.rect(20 + j * 4, 50 + i * 4, 4, 4, R[i][j] < 0 ? [235, 235, 235] : heat(R[i][j]));
    const maxR = Math.max(...rows.flatMap((r) => [...r.rt, ...r.rn]), 1);
    rows.forEach((r, s) => {
      const x0 = 470 + (s % 4) * 255, y0 = s < 4 ? 40 : 340;
      cv.text(x0, y0, r.short, [0, 0, 0], 2);
      r.lv.forEach((v, a) => { const h = clamp01(v) * 70; cv.rect(x0 + 40 + a * 40, y0 + 90 - h, 26, h, PALETTE[a]); cv.text(x0 + 46 + a * 40, y0 + 96, names[a], PALETTE[a], 2); });
      cv.rect(x0 + 40, y0 + 90, 160, 1, [150, 150, 150]);
      r.rt.forEach((v, a) => {
        const h = (v / maxR) * 140, hn = (r.rn[a] / maxR) * 140, x = x0 + 40 + a * 40, base = y0 + 260;
        cv.rect(x, base - h, 18, h, PALETTE[a]);
        cv.rect(x + 19, base - hn, 10, 1, PALETTE[a]); cv.rect(x + 19, base - hn, 1, hn, PALETTE[a]); cv.rect(x + 28, base - hn, 1, hn, PALETTE[a]);
      });
      cv.rect(x0 + 40, y0 + 260, 160, 1, [150, 150, 150]);
    });
    cv.text(470, 620, 'TOP: LAYER 1 LEVELS (THE DRAWING)   BOTTOM: LAYER 2 GROUP RATES  SOLID = TRAINED  OUTLINE = NAIVE', [0, 0, 0], 1);
    cv.save('figures/fig8-two-layer.png');
    fs.writeFileSync('figures/two-layer-results.json', JSON.stringify(result, null, 1));
    log(`\n写入 figures/fig8-two-layer.png（${el()}）`);
  }
  return result;
}

if (import.meta.url === `file:///${process.argv[1].replace(/\\/g, '/')}`) runTwoLayer();
