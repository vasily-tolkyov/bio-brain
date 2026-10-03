/** 第 2 层：化学突触。一个"连接" = contacts 个接触点（每个是一个释放位点）。
 * 突触前脉冲 → 每个位点以 p·R 的概率释放一个囊泡（R = 可用囊泡比例，短时程压抑会消耗它）→ 每个囊泡给突触后加一份量子电导 q。
 * 电流 I = Σ_受体 g(t)·B(V)·(E − V)（pA，正为去极化）；B(V) 是 NMDA 的镁离子阻断。
 * ATP：突触前按囊泡计（递质回收、囊泡循环、Ca²⁺ 外排）；突触后按真实流过受体的离子算（Na⁺ 由钠钾泵、Ca²⁺ 由钙泵、Cl⁻ 经 KCC2 排出）。 */
import { ATP_PER_NA, ATP_PER_CA, ATP_PER_CL, ionsFromFemtoCoulomb } from '../core/units.mjs';

export const RECEPTORS = {
  AMPA: { tauR: 0.2, tauD: 2, E: 0, q: 0.3, na: 1, ca: 0 },
  NMDA: { tauR: 2, tauD: 40, E: 0, q: 0.15, na: 0.9, ca: 0.1, mg: true },
  GABA_A: { tauR: 0.5, tauD: 10, E: -75, q: 0.5, cl: 1 },
  // 快速的 GABA_A：快放电篮状细胞（PV）落在胞体周围的突触，衰减只有几毫秒（Bartos 等 2007）——γ 节律靠它把每次齐放后的抑制压短
  GABA_A_fast: { tauR: 0.3, tauD: 4, E: -75, q: 0.5, cl: 1 },
};
export const PRE_ATP_PER_VESICLE = 2.3e4;

export const mgBlock = (V, Mg = 1) => 1 / (1 + (Mg / 3.57) * Math.exp(-0.062 * V));

/** 双指数核 g(t) = ĝ·(e^{−t/τd} − e^{−t/τr})/峰值：两个状态变量各自衰减，事件同时加到两者 */
class DualExp {
  constructor({ tauR, tauD }) {
    this.tauR = tauR; this.tauD = tauD; this.a = 0; this.b = 0;
    const tp = (tauR * tauD) / (tauD - tauR) * Math.log(tauD / tauR);
    this.norm = 1 / (Math.exp(-tp / tauD) - Math.exp(-tp / tauR));
  }
  kick(g) { this.a += g * this.norm; this.b += g * this.norm; }
  step(dt) { this.a *= Math.exp(-dt / this.tauD); this.b *= Math.exp(-dt / this.tauR); }
  get g() { return Math.max(0, this.a - this.b); }
}

export class Synapse {
  /** kind: 'exc'（AMPA + NMDA）或 'inh'（GABA_A）；scale 缩放每个囊泡的量子电导（长时程可塑性改的就是它） */
  /** category：给定时这条连接的全部 ATP 记到该类别（例如上游驱动记 'drive'），否则分记 syn_pre / syn_post */
  constructor({ kind = 'exc', contacts = 5, pRelease = 0.5, scale = 1, tauRec = 500, Mg = 1, stochastic = false, rng = null, ledger = null, category = null } = {}) {
    this.receptors = (kind === 'exc' ? ['AMPA', 'NMDA'] : ['GABA_A']).map((name) => ({ name, ...RECEPTORS[name], k: new DualExp(RECEPTORS[name]) }));
    Object.assign(this, { contacts, pRelease, scale, tauRec, Mg, stochastic, rng, ledger });
    this.catPre = category ?? 'syn_pre';
    this.catPost = category ?? 'syn_post';
    this.R = 1;
    this.vesicles = 0;
  }

  /** 突触前放电：返回释放的囊泡数 */
  presynapticSpike() {
    const p = this.pRelease * this.R;
    let k;
    if (this.stochastic && this.rng) { k = 0; for (let i = 0; i < this.contacts; i++) if (this.rng.uniform() < p) k++; }
    else k = this.contacts * p;
    this.R = Math.max(0, this.R - k / this.contacts);
    for (const r of this.receptors) r.k.kick(k * r.q * this.scale);
    this.vesicles += k;
    this.ledger?.add(this.catPre, k * PRE_ATP_PER_VESICLE);
    return k;
  }

  /** 前进 dt，返回突触后膜电位 V 下的电流（pA） */
  step(dt, V) {
    this.R += dt * (1 - this.R) / this.tauRec;
    let I = 0;
    for (const r of this.receptors) {
      r.k.step(dt);
      const g = r.k.g * (r.mg ? mgBlock(V, this.Mg) : 1);
      const i = g * (r.E - V);
      I += i;
      if (!this.ledger || g === 0) continue;
      const fC = Math.abs(i) * dt;
      if (r.cl && V > r.E) this.ledger.add(this.catPost, ionsFromFemtoCoulomb(fC) * ATP_PER_CL);
      else if (!r.cl && i > 0) {
        const ions = ionsFromFemtoCoulomb(fC);
        this.ledger.add(this.catPost, ions * r.na * ATP_PER_NA + (ions * r.ca / 2) * ATP_PER_CA);
      }
    }
    return I;
  }

  /** 此刻各受体的电导（nS，NMDA 已乘镁阻断） */
  conductance(V) { return this.receptors.reduce((s, r) => s + r.k.g * (r.mg ? mgBlock(V, this.Mg) : 1), 0); }
}
