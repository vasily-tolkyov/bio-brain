/** 第 1 层：Hodgkin–Huxley 神经元（单位面积），逐步累计 Na⁺/K⁺ 跨膜电荷，换算成钠钾泵要付的 ATP。
 * 漏电拆成 Na⁺ 漏与 K⁺ 漏两部分，使合并后的反转电位等于 E_L——静息时电流为零但离子仍在漏，这就是静息成本。 */
import { ATP_PER_NA, ionsFromNanoCoulomb, UM2_TO_CM2 } from '../core/units.mjs';

/** T（°C）按 Q10 = 3 缩放门控速率（φ = 3^((T−6.3)/10)） */
export const HH_SQUID = { C: 1, gNa: 120, gK: 36, gL: 0.3, ENa: 50, EK: -77, EL: -54.387, T: 6.3 };

const vtrap = (x, y) => (Math.abs(x / y) < 1e-6 ? y * (1 - x / y / 2) : x / (Math.exp(x / y) - 1));
const rates = (V) => ({
  am: 0.1 * vtrap(-(V + 40), 10), bm: 4 * Math.exp(-(V + 65) / 18),
  ah: 0.07 * Math.exp(-(V + 65) / 20), bh: 1 / (1 + Math.exp(-(V + 35) / 10)),
  an: 0.01 * vtrap(-(V + 55), 10), bn: 0.125 * Math.exp(-(V + 65) / 80),
});

export class HHNeuron {
  constructor(p = HH_SQUID, { V0 = -65 } = {}) {
    this.p = { ...HH_SQUID, ...p };
    this.phi = 3 ** ((this.p.T - 6.3) / 10);
    const q = this.p;
    this.gLNa = q.gL * (q.EL - q.EK) / (q.ENa - q.EK);
    this.gLK = q.gL * (q.ENa - q.EL) / (q.ENa - q.EK);
    this.V = V0;
    const r = rates(V0);
    this.m = r.am / (r.am + r.bm); this.h = r.ah / (r.ah + r.bh); this.n = r.an / (r.an + r.bn);
    this.qNa = 0; // 累计内流 Na⁺ 电荷（nC/cm²）
    this.qK = 0;  // 累计外流 K⁺ 电荷（nC/cm²）
    this.t = 0;
    this.above = false;
  }

  /** 前进 dt（ms），外加电流 I（µA/cm²，正为去极化）；越过 0 mV 上行时返回 true */
  step(dt, I = 0) {
    const { C, gNa, gK, ENa, EK } = this.p;
    const V = this.V, r = rates(V);
    for (const [x, a, b] of [['m', r.am, r.bm], ['h', r.ah, r.bh], ['n', r.an, r.bn]]) {
      const inf = a / (a + b), tau = 1 / ((a + b) * this.phi);
      this[x] = inf + (this[x] - inf) * Math.exp(-dt / tau);
    }
    const iNa = (gNa * this.m ** 3 * this.h + this.gLNa) * (V - ENa);
    const iK = (gK * this.n ** 4 + this.gLK) * (V - EK);
    this.V = V + dt * (I - iNa - iK) / C;
    this.qNa += Math.max(0, -iNa) * dt;
    this.qK += Math.max(0, iK) * dt;
    this.t += dt;
    const spike = !this.above && this.V > 0;
    if (this.V > 0) this.above = true; else if (this.V < -20) this.above = false;
    return spike;
  }

  /** 到目前为止，每 cm² 膜要付的 ATP（只按 Na⁺ 计：泵每排 3 个 Na⁺ 同时带回 2 个 K⁺） */
  atpPerCm2() { return ionsFromNanoCoulomb(this.qNa) * ATP_PER_NA; }
}

/** 标定：静息成本（ATP/ms/cm²）、单个动作电位成本（ATP/cm²）、重叠系数（Na⁺ 内流 / 把膜电容充到峰值所需的最少电荷）。
 * 动作电位的 Na⁺ 只在放电本身里积分：从上行越过 −40 mV 起，到复极化回到静息电位以下为止，扣掉同时段的静息漏 */
export function calibrateHH(p = HH_SQUID, dt = 0.005, { pulse = 20, pulseMs = 1 } = {}) {
  const P = { ...HH_SQUID, ...p };
  const settle = (nrn, ms) => { for (let t = 0; t < ms; t += dt) nrn.step(dt, 0); };
  const rest = new HHNeuron(P); settle(rest, 100);
  const vRest = rest.V;
  const q0 = rest.qNa; settle(rest, 100);
  const restNaPerMs = (rest.qNa - q0) / 100;

  const nrn = new HHNeuron(P); settle(nrn, 100);
  let vPeak = -Infinity, spikes = 0, qStart = null, tStart = 0, spikeNa = NaN;
  for (let t = 0; t < 40; t += dt) {
    const prev = nrn.V;
    if (nrn.step(dt, t < pulseMs ? pulse : 0)) spikes++;
    vPeak = Math.max(vPeak, nrn.V);
    if (qStart === null && prev < -40 && nrn.V >= -40) { qStart = nrn.qNa; tStart = t; }
    if (qStart !== null && Number.isNaN(spikeNa) && vPeak > 0 && nrn.V < vRest) spikeNa = nrn.qNa - qStart - restNaPerMs * (t - tStart);
  }
  const minimalNa = P.C * (vPeak - vRest); // µF/cm² × mV = nC/cm²
  return {
    vRest, vPeak, spikes,
    restAtpPerMsCm2: ionsFromNanoCoulomb(restNaPerMs) * ATP_PER_NA,
    spikeAtpPerCm2: ionsFromNanoCoulomb(spikeNa) * ATP_PER_NA,
    spikeNaNanoC: spikeNa, minimalNaNanoC: minimalNa,
    overlap: spikeNa / minimalNa,
  };
}

/** 换算到一个给定膜面积的神经元（µm²）：每次放电与每毫秒静息的 ATP。
 * spikeEfficiency 缩放放电成本（哺乳动物的 Na⁺/K⁺ 重叠更小）；leakNs 给定时按漏电导缩放静息成本（乌贼 HH 的膜比哺乳动物漏得多） */
export function costsForArea(cal, areaUm2, { spikeEfficiency = 1, leakNs = null, p = HH_SQUID } = {}) {
  const cm2 = areaUm2 * UM2_TO_CM2;
  const restScale = leakNs === null ? 1 : leakNs / (p.gL * 1e6 * cm2);
  return { atpPerSpike: cal.spikeAtpPerCm2 * cm2 * spikeEfficiency, atpRestPerMs: cal.restAtpPerMsCm2 * cm2 * restScale };
}

/** 点神经元（LIF_DEFAULT：200 pF、10 nS，即 20000 µm²、膜电阻 20 kΩ·cm²）的哺乳动物式成本：放电重叠系数取 1.3 */
export function mammalianCosts(cal = calibrateHH(), { areaUm2 = 20000, leakNs = 10, overlap = 1.3 } = {}) {
  return costsForArea(cal, areaUm2, { spikeEfficiency: overlap / cal.overlap, leakNs });
}
