/** 从经验里提取规则，并在新场景里按规则预测：前馈的逐层抽象 + 结果到来后的逐层"重新解释"。
 * 世界（像 Minecraft 挖方块）：每次经历 = 一个方块 + 一把工具 → 挖得快还是慢。
 *   方块的可感知特征：颜色、亮度、纹理（外观，3 个）+ 敲击声（硬度的线索，1 个），各用 12 个神经元群体编码。
 *   规则：工具对路就快——锹挖土类、斧砍木类、镐敲石类；不对路就慢。
 *   难点：外观占了大部分输入，且每种材质都有一个"长得几乎一样"的别种材质（泥土 ≈ 深色橡木，砂砾 ≈ 石头，白桦 ≈ 闪长岩）。
 * 网络：特征注意力（每个特征一个增益）→ 第一层 4 个吸引子（方块 → 概念，比方块种类少，必须合并）
 *       → 第二层 12 个吸引子（概念 + 工具 → 情境）→ 读出（每个情境预测"快"的概率）。
 * 每次经历两个相位：
 *   预测相位：只凭看到的方块和工具自下而上沉降，给出预测；
 *   结果相位（只在预测错了时）：实际结果钳住顶层。第二层只允许"会给出实际结果"的情境，于是第一层只剩
 *     "配上这把工具会通向这类情境"的概念，在其中取最像看到的方块的那个——结果逐层改写了下层的解释。
 *   学习：结果相位的概念向这次输入靠拢，预测相位的概念退开；注意力向"支持正确概念"的特征倾斜（同 ALCOVE）；
 *     找不到一致的概念，就是规则本身要改，只调读出。
 * 对照一：只有前馈抽象——两层都只在预测相位做竞争性赫布学习，结果只用来调读出，注意力不变。
 * 对照二：结果逐层传回，但经验缺少变化（每种材质只见过一种外观）：一组里声音碰巧最好分，一组里外观和声音一样好分。
 * 陷阱测试：每种见过的外观配上另一种材质的声音——只有学到"按声音分材质"的网络能答对。
 * 用法：node demo/rule-extraction.mjs（几秒）→ figures/fig9-rules.png */
import fs from 'fs';
import { makeRng } from '../src/core/rng.mjs';
import { Canvas, PALETTE, heat } from '../src/core/png.mjs';

const NB = 12, SIG = 0.07, NF = 4, N1 = 4, N2 = 12;
const FEAT = ['颜色', '亮度', '纹理', '声音'], MAT = ['土类', '木类', '石类'], TOOL = ['锹', '斧', '镐'], SOUND = [0.15, 0.5, 0.85];
const fastOf = (mat, tool) => (mat === tool ? 1 : 0);
// [名字, 材质, 外观 [颜色, 亮度, 纹理], 图上的英文名]
const SEEN = [
  ['泥土', 0, [0.08, 0.35, 0.30]], ['沙子', 0, [0.15, 0.80, 0.15]], ['砂砾', 0, [0.60, 0.50, 0.66]],
  ['橡木', 1, [0.12, 0.60, 0.50]], ['白桦', 1, [0.20, 0.88, 0.40]], ['深色橡木', 1, [0.08, 0.32, 0.36]],
  ['石头', 2, [0.60, 0.52, 0.60]], ['花岗岩', 2, [0.95, 0.55, 0.65]], ['闪长岩', 2, [0.22, 0.85, 0.46]],
];
// 经验缺少变化：每种材质只见过一种外观。一组里声音碰巧比任何单个外观特征都好分，另一组里外观（亮度、纹理）和声音一样好分
const SEEN_POOR = [SEEN[0], SEEN[3], SEEN[6]];
const SEEN_POOR_SALIENT = [['沙子', 0, [0.15, 0.90, 0.10]], ['深色橡木', 1, [0.08, 0.20, 0.50]], ['花岗岩', 2, [0.95, 0.55, 0.90]]];
/** 系统的陷阱：每种见过的外观，配上另一种材质的声音 */
const trapsOf = (seen) => seen.flatMap((b) => [0, 1, 2].filter((m) => m !== b[1]).map((m) => [`长得像${b[0]}的${MAT[m]}`, m, b[2]]));
const NEW = [['红沙', 0, [0.02, 0.65, 0.20], 'RED SAND'], ['绯红木', 1, [0.90, 0.40, 0.50], 'CRIMSON WOOD'], ['诡异木', 1, [0.45, 0.50, 0.50], 'WARPED WOOD'], ['深板岩', 2, [0.70, 0.20, 0.80], 'DEEPSLATE']];
const TRAP = [['灰色的木头（长得像砂砾、石头）', 1, [0.60, 0.51, 0.63], 'GRAY WOOD'], ['白色的软方块（长得像白桦、闪长岩）', 0, [0.21, 0.86, 0.43], 'WHITE SOFT'], ['褐色的硬方块（长得像泥土、深色橡木）', 2, [0.08, 0.34, 0.33], 'BROWN HARD']];

const norm = (v) => { let s = 0; for (const x of v) s += x * x; s = Math.sqrt(s) || 1; return v.map((x) => x / s); };
const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const argmax = (a) => { let b = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[b]) b = i; return b; };
const bumps = (f) => { const u = new Float64Array(NF * NB); f.forEach((v, k) => { for (let i = 0; i < NB; i++) u[k * NB + i] = Math.exp(-((v - i / (NB - 1)) ** 2) / (2 * SIG * SIG)); }); return u; };
const typical = (b) => [...b[2], SOUND[b[1]]];
const sampleOf = (b, rng) => [...b[2].map((v) => v + 0.03 * rng.normal()), SOUND[b[1]] + 0.03 * rng.normal()];

/** 一层吸引子：原型（归一化）、竞争（赢者通吃）、稳态阈值（防止永远不赢的"死"吸引子） */
class Layer {
  constructor(D, N, rng) { this.N = N; this.W = Array.from({ length: N }, () => norm(Float64Array.from({ length: D }, () => rng.uniform()))); this.theta = new Float64Array(N); this.wins = new Float64Array(N); }
  sims(un) { return this.W.map((w, i) => dot(w, un) - this.theta[i]); }
  move(i, un, eta) { const w = this.W[i]; for (let j = 0; j < w.length; j++) w[j] += eta * (un[j] - w[j]); this.W[i] = norm(w); }
  home(win) { for (let i = 0; i < this.N; i++) { this.wins[i] = 0.99 * this.wins[i] + 0.01 * (i === win ? 1 : 0); this.theta[i] += 0.01 * (this.wins[i] - 1 / this.N); } }
}

function makeModel(errorDriven, seed, { eta = 0.05, etaAttn = 0.2 } = {}) {
  const rng = makeRng(seed);
  const L1 = new Layer(NF * NB, N1, rng), L2 = new Layer(N1 + 3, N2, rng), v = new Float64Array(N2).fill(0.5), attn = new Float64Array(NF).fill(1);
  const enc = (f) => { const u = bumps(f); for (let k = 0; k < NF; k++) for (let i = 0; i < NB; i++) u[k * NB + i] *= attn[k]; return norm(u); };
  const in2 = (i, tool) => { const u = new Float64Array(N1 + 3); u[i] = 1; u[N1 + tool] = 1; return norm(u); };
  /** 预测相位：自下而上 */
  const predict = (f, tool) => { const un1 = enc(f), s1 = L1.sims(un1), i = argmax(s1), k = argmax(L2.sims(in2(i, tool))); return { un1, s1, i, k, y: v[k] }; };
  /** 结果相位（只在预测错了时）：顶层被结果钳住，工具不变；只剩"配上这把工具会给出实际结果"的概念，取其中最像看到的 */
  const settleWithOutcome = (p, tool, t) => {
    let i = -1, k = -1;
    for (let q = 0; q < N1; q++) {
      const kq = argmax(L2.sims(in2(q, tool)));
      if (Math.abs(t - v[kq]) < 0.5 && (i < 0 || p.s1[q] > p.s1[i])) { i = q; k = kq; }
    }
    return i < 0 ? p : { i, k };
  };
  function experience(f, tool, t) {
    const p = predict(f, tool);
    v[p.k] += 0.1 * (t - v[p.k]);
    L1.home(p.i); L2.home(p.k);
    if (!errorDriven || Math.abs(t - p.y) < 0.5) { L1.move(p.i, p.un1, eta); L2.move(p.k, in2(p.i, tool), eta); return { p, c: p }; }
    const c = settleWithOutcome(p, tool, t);
    if (c.i !== p.i) {
      // 注意力：哪些特征支持结果相位的概念、哪些把预测相位带错了路
      const raw = bumps(f);
      for (let q = 0; q < NF; q++) { let a = 0, b = 0; for (let j = q * NB; j < (q + 1) * NB; j++) { a += L1.W[c.i][j] * raw[j]; b += L1.W[p.i][j] * raw[j]; } attn[q] = Math.max(0.05, attn[q] + etaAttn * (a - b)); }
      const mean = attn.reduce((s, x) => s + x, 0) / NF; for (let q = 0; q < NF; q++) attn[q] /= mean;
      L1.move(p.i, p.un1, -0.5 * eta);
    }
    L1.move(c.i, p.un1, eta);
    L2.move(c.k, in2(c.i, tool), eta);
    return { p, c };
  }
  return { L1, L2, v, attn, enc, predict, experience };
}

/** 每个第一层概念收了哪些方块（按每种方块的典型样子归类） */
const conceptMembers = (m, blocks) => Array.from({ length: N1 }, (_, i) => blocks.filter((b) => argmax(m.L1.sims(m.enc(typical(b)))) === i));
const describe = (members) => (members.length ? members.map((b) => b[0]).join('、') : '（空）');

function run(name, errorDriven, seen, { seed = 7, epochs = 6, trials = 1500, log = console.log, stories = false } = {}) {
  const m = makeModel(errorDriven, seed), rng = makeRng(seed + 1000);
  const curve = [];
  let told = 0;
  for (let e = 0; e < epochs; e++) {
    let re1 = 0, wrong = 0;
    for (let n = 0; n < trials; n++) {
      const b = seen[rng.pick(seen.length)], tool = rng.pick(3), t = fastOf(b[1], tool);
      const before = stories && told < 4 && e === 0 && n > 200 ? conceptMembers(m, seen) : null;
      const { p, c } = m.experience(sampleOf(b, rng), tool, t);
      if ((p.y > 0.5 ? 1 : 0) !== t) wrong++;
      if (c.i !== p.i) re1++;
      if (before && c.i !== p.i) {
        told++;
        log(`  例：${b[0]} + ${TOOL[tool]}，预测"${p.y > 0.5 ? '快' : '慢'}"（第一层认作概念 #${p.i}：${describe(before[p.i])}），实际"${t ? '快' : '慢'}" → 重新解释为概念 #${c.i}：${describe(before[c.i])}`);
      }
    }
    curve.push({ wrong: wrong / trials, re1: re1 / trials });
  }
  const acc = (blocks, reps = 30) => { let ok = 0, n = 0; const r = makeRng(seed + 2000); for (const b of blocks) for (let tool = 0; tool < 3; tool++) for (let q = 0; q < reps; q++) { ok += (m.predict(sampleOf(b, r), tool).y > 0.5 ? 1 : 0) === fastOf(b[1], tool) ? 1 : 0; n++; } return ok / n; };
  const grid = (blocks) => blocks.map((b) => [0, 1, 2].map((tool) => m.predict(typical(b), tool).y));
  const members = conceptMembers(m, seen);
  const used = members.filter((g) => g.length);
  const purity = used.reduce((s, g) => s + Math.max(...[0, 1, 2].map((k) => g.filter((b) => b[1] === k).length)) / g.length, 0) / used.length;
  const res = { name, curve, attn: [...m.attn], accSeen: acc(seen), accNew: acc(NEW), accTraps: acc(trapsOf(seen)), accTrap: acc(TRAP), members: members.map((g) => g.map((b) => b[0])), memberMats: members.map((g) => g.map((b) => b[1])), purity, gridNew: grid(NEW), gridTrap: grid(TRAP) };
  log(`【${name}】预测错误率（每轮）${curve.map((c) => (c.wrong * 100).toFixed(0) + '%').join(' → ')}${errorDriven ? `；被结果改写的第一层解释 ${curve.map((c) => (c.re1 * 100).toFixed(0) + '%').join(' → ')}` : ''}`);
  log(`  注意力：${FEAT.map((f, q) => `${f} ${m.attn[q].toFixed(2)}`).join('  ')}`);
  log(`  第一层的概念：${members.map((g, i) => (g.length ? `#${i} ${describe(g)}` : '')).filter(Boolean).join('；')}（按材质的纯度 ${(purity * 100).toFixed(0)}%）`);
  log(`  正确率：见过的方块 ${(res.accSeen * 100).toFixed(1)}%，没见过外观的新方块 ${(res.accNew * 100).toFixed(1)}%，陷阱（见过的外观配另一种材质的声音，${trapsOf(seen).length} 种）${(res.accTraps * 100).toFixed(1)}%`);
  return res;
}

const log = console.log;
const A = run('前馈抽象 + 结果逐层传回', true, SEEN, { stories: true });
const B = run('只有前馈抽象（结果只调读出）', false, SEEN);
const C1 = run('结果逐层传回，经验缺少变化，声音碰巧最好分（泥土、橡木、石头）', true, SEEN_POOR);
const C = run('结果逐层传回，经验缺少变化，外观和声音一样好分（沙子、深色橡木、花岗岩各一种外观）', true, SEEN_POOR_SALIENT);
log('\n新场景里按规则预测（"快"的概率，锹 / 斧 / 镐）：');
for (const [blocks, key] of [[NEW, 'gridNew'], [TRAP, 'gridTrap']]) blocks.forEach((b, r) => log(`  ${b[0]}（${MAT[b[1]]}，规则：${TOOL[b[1]]}快）：传回 ${A[key][r].map((y) => y.toFixed(2)).join(' / ')}  只前馈 ${B[key][r].map((y) => y.toFixed(2)).join(' / ')}`));

// ── 图 9：上 = 第一层每个概念收了哪些方块（按材质着色）与注意力；下 = 新场景里的预测 ──
fs.mkdirSync('figures', { recursive: true });
const cv = new Canvas(1500, 800);
const MATCOL = [PALETTE[3], PALETTE[2], PALETTE[1]];
cv.text(20, 12, 'LAYER 1 CONCEPTS: BLOCKS EACH ATTRACTOR TAKES (ORANGE SOIL  GREEN WOOD  BLUE STONE)   BARS = ATTENTION C B T S', [0, 0, 0], 2);
[[A, 'OUTCOME FED BACK'], [B, 'FEEDFORWARD ONLY'], [C, 'FED BACK  POOR VARIETY']].forEach(([res, title], p) => {
  const x0 = 20 + p * 490;
  cv.text(x0, 44, title, [0, 0, 0], 2);
  res.memberMats.forEach((mats, i) => mats.forEach((mat, q) => cv.rect(x0 + i * 50, 290 - (q + 1) * 40, 40, 36, MATCOL[mat])));
  cv.rect(x0, 292, 200, 1, [150, 150, 150]);
  res.attn.forEach((a, q) => { const h = Math.min(2.5, a) / 2.5 * 200; cv.rect(x0 + 260 + q * 45, 290 - h, 34, h, q === 3 ? [200, 30, 30] : [120, 120, 120]); });
  cv.rect(x0 + 260, 292, 180, 1, [150, 150, 150]);
});
cv.text(20, 330, 'NEW SCENES: DARK = PREDICTED FAST (SHOVEL AXE PICK)   RED SQUARE = WHAT THE RULE SAYS', [0, 0, 0], 2);
const rows = [...NEW, ...TRAP];
[[A, 'FED BACK'], [B, 'FEEDFORWARD']].forEach(([res, title], p) => {
  const x0 = 260 + p * 300, all = [...res.gridNew, ...res.gridTrap];
  cv.text(x0, 360, title, [0, 0, 0], 2);
  rows.forEach((b, r) => all[r].forEach((y, tool) => {
    const x = x0 + tool * 60, yy = 390 + r * 50;
    cv.rect(x, yy, 54, 44, heat(y));
    if (b[1] === tool) cv.rect(x + 22, yy + 17, 10, 10, [230, 30, 30]);
  }));
});
rows.forEach((b, r) => cv.text(20, 405 + r * 50, b[3], r < NEW.length ? [0, 0, 0] : [160, 0, 0], 2));
cv.text(20, 760, 'BLACK NAMES = NEVER-SEEN LOOKS   DARK RED NAMES = TRAPS THAT LOOK LIKE ANOTHER MATERIAL', [0, 0, 0], 2);
cv.save('figures/fig9-rules.png');
fs.writeFileSync('figures/rules-results.json', JSON.stringify({ A, B, C1, C }, null, 1));
log('\n写入 figures/fig9-rules.png');
