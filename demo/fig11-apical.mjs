/** 预测能不能传到正在感知的下层：点神经元（反馈落在胞体）vs 带顶树突的神经元（反馈落在顶树突）。
 * 读 demo/fig10-hierarchy.mjs 两种结构的结果（figures/hierarchy-results.json、hierarchy-results-apical.json），对照作图。
 * 用法：先 node demo/fig10-hierarchy.mjs 与 node demo/fig10-hierarchy.mjs apical，再 node demo/fig11-apical.mjs → figures/fig11-apical.png */
import fs from 'fs';
import { Canvas, PALETTE } from '../src/core/png.mjs';

const load = (f) => JSON.parse(fs.readFileSync(f, 'utf8'));
const P = load('figures/hierarchy-results.json'), A = load('figures/hierarchy-results-apical.json');
const pick = (R, name) => R.results.find((r) => r.name === name);
const [p2, pNo, pCut] = ['2 层', '2 层，上层没经历物体序列', '2 层，切断反馈'].map((n) => pick(P, n));
const [a2, aNo, aCut] = ['2 层', '2 层，上层没经历物体序列', '2 层，切断反馈'].map((n) => pick(A, n));
const f1 = (x) => (x === null || x === undefined ? '—' : x.toFixed(1));
const log = console.log;
log('                                  点神经元：2 层 / 没经历序列 / 切断反馈      带顶树突：2 层 / 没经历序列 / 切断反馈');
const row = (label, get, f = f1) => log(`${label.padEnd(28)}${[p2, pNo, pCut].map((r) => f(get(r)).padStart(8)).join('')}        ${[a2, aNo, aCut].map((r) => f(get(r)).padStart(8)).join('')}`);
row('第二层 Y 的组（Hz）', (r) => r.pred[1]);
row('第一层 Y 的组胞体电位（mV）', (r) => r.vMean);
row('第一层 Y 的组顶树突电位（mV）', (r) => r.vdMean);
row('Y 紧跟 X：头 20 ms（Hz）', (r) => r.facil);
row('Y 紧跟 X：头 60 ms（Hz）', (r) => r.facil60);
row('Y 紧跟 X：放电神经元比例（%）', (r) => r.fireExp * 100);
row('Y 紧跟 X：首个脉冲延迟（ms）', (r) => r.latExp);
row('空白之后 Y：头 20 ms（Hz）', (r) => r.noCtx);
row('空白之后 Y：首个脉冲延迟（ms）', (r) => r.latNo);
row('第一层不变性', (r) => r.invariance[0], (x) => x.toFixed(2));
row('第一层补全（Hz）', (r) => r.complete[0]);
row('想象：第一层 X 的组（Hz）', (r) => r.imagery.x1);
row('想象：第一层 Y 的组（Hz）', (r) => r.imagery.y1);
log('\n带顶树突，按层数：' + A.results.slice(0, 4).map((r) => `${r.K} 层 不变性 ${r.invariance[r.K - 1].toFixed(2)}、噪声下辨识 ${r.noisy[r.K - 1].toFixed(2)}、ATP ${(r.atp / 1e9).toFixed(0)}×10⁹、第一层 Y 顶树突 ${r.vdMean.toFixed(2)} mV、Y 紧跟 X 头 60 ms ${r.facil60.toFixed(1)} Hz`).join('；'));
log('学到的结构（带顶树突，4 层）：' + A.structure.map((s, k) => `第${k + 1}层 物体内 ${s.within.toFixed(2)} X→Y ${s.fwd.toFixed(2)} Y→X ${s.bwd.toFixed(2)}${s.fbSame !== undefined ? ` 反馈 ${s.fbSame.toFixed(2)}/${s.fbCross.toFixed(2)}` : ''}`).join('；'));

// ── 图 11 ──
const cv = new Canvas(1500, 820);
const line = (arr, x0, y0, w, h, lo, hi, col, width = 3, dashed = false) => { for (let i = 1; i < arr.length; i++) { if (dashed && i % 4 < 2) continue; const y = (v) => y0 + h - ((Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo)) * h; cv.line(x0 + ((i - 1) / arr.length) * w, y(arr[i - 1]), x0 + (i / arr.length) * w, y(arr[i]), col, width); } };
const panelV = (title, traces, x0) => {
  cv.text(x0, 12, title, [0, 0, 0], 2);
  const all = traces.flatMap(([t]) => t.slice(0, 120)), lo = Math.min(...all) - 0.5, hi = Math.max(...all) + 0.5;
  traces.forEach(([t, col, dashed]) => line(t.slice(0, 120), x0 + 20, 50, 620, 240, lo, hi, col, dashed ? 1 : 3, dashed));
  cv.rect(x0 + 20, 290, 620, 1, [150, 150, 150]); cv.rect(x0 + 20 + 620 * 40 / 120, 50, 1, 240, [200, 200, 200]);
  cv.text(x0 + 20, 300, `${lo.toFixed(0)} TO ${hi.toFixed(0)} MV   0-120 MS   X 0-40 MS`, [0, 0, 0], 1);
};
panelV('POINT NEURONS: L1 Y SOMA POTENTIAL', [[p2.vTrace, PALETTE[1]], [pNo.vTrace, [150, 150, 150]], [pCut.vTrace, [0, 0, 0], true]], 20);
panelV('APICAL: L1 Y DENDRITE POTENTIAL', [[a2.vdTrace, PALETTE[1]], [aNo.vdTrace, [150, 150, 150]], [aCut.vdTrace, [0, 0, 0], true]], 760);
cv.text(20, 320, 'BLUE = 2 LAYERS   GRAY = UPPER LAYER NEVER SAW X THEN Y   DASHED = FEEDBACK CUT', [0, 0, 0], 1);
cv.text(20, 360, 'Y RIGHT AFTER X: LAYER 1 Y GROUPS IN FIRST 60 MS', [0, 0, 0], 2);
const groups = [['POINT', [p2, pNo, pCut]], ['APICAL', [a2, aNo, aCut]]];
const metrics = [['RATE HZ', (r) => r.facil60, 30], ['FIRING %', (r) => r.fireExp * 100, 100], ['FIRST SPIKE MS', (r) => r.latExp, 40]];
metrics.forEach(([lab, get, mx], m) => {
  const x0 = 40 + m * 480;
  cv.text(x0, 392, lab, [0, 0, 0], 2);
  groups.forEach(([g, rs], gi) => {
    rs.forEach((r, j) => { const v = get(r) ?? 0, h = (Math.min(mx, Math.max(0, v)) / mx) * 300; cv.rect(x0 + gi * 210 + j * 60, 740 - h, 50, h, [PALETTE[1], [150, 150, 150], [60, 60, 60]][j]); });
    cv.text(x0 + gi * 210, 752, g, [0, 0, 0], 2);
  });
  cv.rect(x0, 740, 420, 1, [150, 150, 150]);
});
cv.text(40, 790, 'BARS: 2 LAYERS (BLUE)  NO X-THEN-Y (GRAY)  FEEDBACK CUT (DARK)', [0, 0, 0], 1);
cv.save('figures/fig11-apical.png');
log('\n写入 figures/fig11-apical.png');
