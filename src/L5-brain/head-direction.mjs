/** 第 5 层：方向环（果蝇中央复合体的 EPG 环吸引子 + PEN 移位通道 + PFL3 转向读出）。
 * 连接核来自 maleCNS 实测的 EPG→EPG 有效作用（data/malecns-epg-kernel.json）：
 *   对称部分 K_sym = 直接 + 经 PEG + 经 Δ7（近处支持、对面抑制）——一口能量井，凸包停在哪里都行（连续吸引子）；
 *   左右两路 PEN（K_L 偏 −45°、K_R 偏 +45°）——角速度把一侧调高、另一侧调低，核变得不对称，凸包就转（有向通道）。
 * 8 档的实测值用余弦级数插值到 N 个单元。速率单元：τ dr/dt = −r + clip(u, 0, 1)。 */
import fs from 'fs';

const KDATA = JSON.parse(fs.readFileSync(new URL('../../data/malecns-epg-kernel.json', import.meta.url), 'utf8'));
const TAU = 2 * Math.PI;
export const wrapPi = (a) => { let x = ((a % TAU) + TAU) % TAU; if (x > Math.PI) x -= TAU; return x; };

/** 8 档实测值 → 任意角度的核（余弦 + 正弦级数，最多 4 阶，恰好穿过 8 个点） */
export function kernelFromBins(values, binsDeg = KDATA.binsDeg) {
  const th = binsDeg.map((d) => (d * Math.PI) / 180), n = th.length;
  const a = [], b = [];
  for (let k = 0; k <= n / 2; k++) {
    a.push((k === 0 || k === n / 2 ? 1 : 2) / n * values.reduce((s, v, i) => s + v * Math.cos(k * th[i]), 0));
    b.push(k === 0 || k === n / 2 ? 0 : (2 / n) * values.reduce((s, v, i) => s + v * Math.sin(k * th[i]), 0));
  }
  return (d) => a.reduce((s, ak, k) => s + ak * Math.cos(k * d) + b[k] * Math.sin(k * d), 0);
}

/** 偶部（左右平均）与奇部（左右之差的一半）：偶部是对称的井；奇部是实测里左右不对称的那一点"风" */
function mirrorIndex(i) { const d = KDATA.binsDeg[i]; return KDATA.binsDeg.indexOf(d === 180 ? 180 : -d); }
export const evenPart = (values) => values.map((v, i) => 0.5 * (v + values[mirrorIndex(i)]));
export const oddPart = (values) => values.map((v, i) => 0.5 * (v - values[mirrorIndex(i)]));
export const KERNEL_DATA = KDATA;

export class HeadDirectionRing {
  /** symmetric：静态核只取偶部（默认），否则带上实测的左右不对称——黑暗中凸包会被它吹着走 */
  constructor({ N = 32, gain = 0.12, inhibition = 1, bias = 0.05, kappa = 3, tau = 1, symmetric = true } = {}) {
    this.N = N;
    this.theta = Array.from({ length: N }, (_, i) => (i / N) * TAU - Math.PI);
    const raw = KDATA.direct.map((v, i) => v + KDATA.viaDelta7[i]);
    const sym = kernelFromBins(symmetric ? evenPart(raw) : raw);
    // 左右对称版：两路 PEN 取互为镜像的平均（左路在 Δ 处 = 右路在 −Δ 处），静止时（ω = 0）合起来严格对称
    const mirrorAvg = KDATA.viaPEN_L.map((v, i) => 0.5 * (v + KDATA.viaPEN_R[mirrorIndex(i)]));
    const penL = symmetric ? mirrorAvg : KDATA.viaPEN_L, penR = symmetric ? mirrorAvg.map((_, i) => mirrorAvg[mirrorIndex(i)]) : KDATA.viaPEN_R;
    const kl = kernelFromBins(penL), kr = kernelFromBins(penR);
    const dens = N / 8;
    this.Ks = this.theta.map((ti) => this.theta.map((tj) => (gain * sym(wrapPi(ti - tj))) / dens - inhibition / N));
    this.KL = this.theta.map((ti) => this.theta.map((tj) => (gain * kl(wrapPi(ti - tj))) / dens));
    this.KR = this.theta.map((ti) => this.theta.map((tj) => (gain * kr(wrapPi(ti - tj))) / dens));
    Object.assign(this, { bias, kappa, tau });
    this.r = new Float64Array(N);
  }

  /** omega：角速度信号（无单位，>0 调高右侧 PEN）；landmark：视觉地标给出的方向（null = 黑暗）；cueGain：地标输入强度 */
  step(dt, { omega = 0, landmark = null, cueGain = 0.3, drive = 0 } = {}) {
    const N = this.N, gL = Math.max(0, 1 - this.kappa * omega), gR = Math.max(0, 1 + this.kappa * omega);
    const u = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      let s = this.bias + drive;
      for (let j = 0; j < N; j++) s += (this.Ks[i][j] + gL * this.KL[i][j] + gR * this.KR[i][j]) * this.r[j];
      if (landmark !== null) s += cueGain * Math.max(0, Math.cos(this.theta[i] - landmark));
      u[i] = s;
    }
    for (let i = 0; i < N; i++) this.r[i] += (dt / this.tau) * (Math.min(1, Math.max(0, u[i])) - this.r[i]);
  }

  /** 凸包的位置（群体向量）与强度（0–1） */
  bump() {
    let x = 0, y = 0, s = 0;
    for (let i = 0; i < this.N; i++) { x += this.r[i] * Math.cos(this.theta[i]); y += this.r[i] * Math.sin(this.theta[i]); s += this.r[i]; }
    return { pos: Math.atan2(y, x), strength: s > 1e-9 ? Math.hypot(x, y) / s : 0, active: s };
  }
}

/** PFL3 式转向读出：左右两群各拿一份平移了 ∓φ 的航向，与目标方向比较，过一个平方整流，右减左。
 * 航向等于目标时为零且稳定（偏左则向右转）；背对目标时为零但不稳定。 */
export function pfl3Turn(heading, goal, { shift = Math.PI / 4, n = 8 } = {}) {
  let L = 0, R = 0;
  for (let k = 0; k < n; k++) {
    const pref = (k / n) * TAU;
    const hL = Math.cos(heading - pref + shift), hR = Math.cos(heading - pref - shift), g = Math.cos(goal - pref);
    L += Math.max(0, hL + g) ** 2; R += Math.max(0, hR + g) ** 2;
  }
  return (L - R) / (L + R || 1);
}
