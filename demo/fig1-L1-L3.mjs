/** 图 1：外界刺激下的前三层。
 * 上：HH 神经元受 10 µA/cm² 阶跃刺激——膜电位与累计 ATP；
 * 中：突触前 20 Hz 串刺激——突触后 EPSP 逐个变小（短时程压抑），每次释放的囊泡数；
 * 下：一对神经元的三种刺激方式——先后配对（突触前先 8 ms）、只刺激突触后、只刺激突触前——突触效能 ρ 的轨迹（网络版钙规则）。
 * 用法：node demo/fig1-L1-L3.mjs → figures/fig1-L1-L3.png */
import fs from 'fs';
import { HHNeuron, calibrateHH, costsForArea } from '../src/L1-neuron/hh.mjs';
import { LIFNeuron } from '../src/L1-neuron/lif.mjs';
import { Synapse } from '../src/L2-synapse/synapse.mjs';
import { CalciumPlasticity, GB_NMDA } from '../src/L3-plasticity/calcium-rule.mjs';
import { Canvas, PALETTE } from '../src/core/png.mjs';

fs.mkdirSync('figures', { recursive: true });
const W = 1200, H = 960, cv = new Canvas(W, H);
const BLACK = [0, 0, 0], GREY = [170, 170, 170];
const panel = (x, y, w, h, title) => { cv.rect(x, y, w, 1, GREY); cv.rect(x, y + h, w, 1, GREY); cv.rect(x, y, 1, h, GREY); cv.rect(x + w, y, 1, h + 1, GREY); cv.text(x + 6, y - 22, title, BLACK, 2); };
const plot = (xs, ys, { x, y, w, h, x0, x1, y0, y1, color, width = 2 }) => {
  for (let k = 1; k < xs.length; k++) {
    const px = (v) => x + ((v - x0) / (x1 - x0)) * w, py = (v) => y + h - ((v - y0) / (y1 - y0)) * h;
    cv.line(px(xs[k - 1]), py(Math.min(y1, Math.max(y0, ys[k - 1]))), px(xs[k]), py(Math.min(y1, Math.max(y0, ys[k]))), color, width);
  }
};

// ── 上：HH 神经元 ──
const dt = 0.01, hh = new HHNeuron();
for (let t = 0; t < 50; t += dt) hh.step(dt, 0);
const cal = calibrateHH(), area = 20000, q0 = hh.atpPerCm2();
const T1 = [], V1 = [], A1 = [];
let spikes = 0;
for (let t = 0; t < 150; t += dt) {
  if (hh.step(dt, t >= 20 && t < 120 ? 10 : 0)) spikes++;
  if (Math.round(t / dt) % 10 === 0) { T1.push(t); V1.push(hh.V); A1.push((hh.atpPerCm2() - q0) * area * 1e-8); }
}
const p1 = { x: 80, y: 50, w: 1080, h: 230 };
panel(p1.x, p1.y, p1.w, p1.h, 'L1 HH NEURON: STEP STIMULUS 20-120 MS   RED: V (-80..+50 MV)   BLUE: CUMULATIVE ATP');
cv.rect(p1.x + (20 / 150) * p1.w, p1.y + p1.h - 8, (100 / 150) * p1.w, 8, [255, 210, 120]);
plot(T1, V1, { ...p1, x0: 0, x1: 150, y0: -80, y1: 50, color: PALETTE[0] });
plot(T1, A1, { ...p1, x0: 0, x1: 150, y0: 0, y1: A1.at(-1) * 1.05, color: PALETTE[1] });

// ── 中：突触短时程压抑 ──
const syn = new Synapse({ contacts: 10, pRelease: 0.5, tauRec: 500 });
const post = new LIFNeuron({ Vth: 1e9 });
const T2 = [], V2 = [], rel = [];
for (let t = 0, k = 0; t < 700; t += 0.1) {
  if (k < 10 && t >= 20 + 50 * k) { rel.push({ t, v: syn.presynapticSpike() }); k++; }
  post.step(0.1, 0, syn.step(0.1, post.V));
  if (Math.round(t * 10) % 5 === 0) { T2.push(t); V2.push(post.V); }
}
const p2 = { x: 80, y: 370, w: 1080, h: 230 };
panel(p2.x, p2.y, p2.w, p2.h, 'L2 SYNAPSE: PRESYNAPTIC 20 HZ X10   RED: POSTSYNAPTIC V   GREEN BARS: VESICLES RELEASED');
for (const r of rel) cv.rect(p2.x + (r.t / 700) * p2.w - 3, p2.y + p2.h - (r.v / rel[0].v) * 80, 6, (r.v / rel[0].v) * 80, PALETTE[2]);
plot(T2, V2, { ...p2, x0: 0, x1: 700, y0: -65.2, y1: Math.max(...V2) + 0.3, color: PALETTE[0] });

// ── 下：赫布学习的三种刺激方式 ──
const protocols = [['PAIRED PRE->POST 8MS', (k) => ({ pre: k * 200, post: k * 200 + 8 })], ['POST ONLY', (k) => ({ post: k * 200 })], ['PRE ONLY', (k) => ({ pre: k * 200 })]];
const p3 = { x: 80, y: 690, w: 1080, h: 230 };
panel(p3.x, p3.y, p3.w, p3.h, 'L3 HEBBIAN: 60 STIMULI 5 HZ  RHO 0.45-0.58  RED PAIRED  BLUE POST ONLY  GREEN PRE ONLY');
const finals = [];
protocols.forEach(([name, ev], pi) => {
  const pl = new CalciumPlasticity(GB_NMDA, { rho0: 0.5 });
  const T = [], R = [];
  let k = 0;
  for (let t = 0; t < 12500; t += 0.1) {
    const e = k < 60 ? ev(k) : {};
    if (e.pre !== undefined && Math.abs(t - e.pre) < 0.05) pl.preSpike(t);
    if (e.post !== undefined && Math.abs(t - e.post) < 0.05) pl.postSpike();
    if (k < 60 && t > (e.post ?? e.pre) + 1) k++;
    pl.step(0.1, t);
    if (Math.round(t * 10) % 500 === 0) { T.push(t); R.push(pl.rho); }
  }
  finals.push([name, pl.rho]);
  plot(T, R, { ...p3, x0: 0, x1: 12500, y0: 0.45, y1: 0.58, color: [PALETTE[0], PALETTE[1], PALETTE[2]][pi], width: 3 });
});
cv.rect(p3.x, p3.y + p3.h * (0.08 / 0.13), p3.w, 1, GREY);
cv.save('figures/fig1-L1-L3.png');

const costs = costsForArea(cal, area);
console.log(`L1：100 ms 刺激内放电 ${spikes} 次，累计 ATP ${A1.at(-1).toExponential(2)}（20000 µm²，单次放电约 ${costs.atpPerSpike.toExponential(2)}）`);
console.log(`L2：10 个脉冲的释放囊泡 ${rel.map((r) => r.v.toFixed(2)).join(' ')}；EPSP 峰值 ${(Math.max(...V2) + 65).toFixed(2)} mV`);
console.log(`L3：60 次刺激后 ρ：${finals.map(([n, r]) => `${n} ${r.toFixed(3)}`).join('，')}`);
console.log('写入 figures/fig1-L1-L3.png');
