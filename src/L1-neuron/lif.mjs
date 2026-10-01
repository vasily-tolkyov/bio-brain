/** 第 1 层（点神经元）：带适应电流的整合放电模型，ATP 成本由 HH 标定后按膜面积换算。
 * C dV/dt = −g_L(V−E_L) − w + I_syn + I_ext；τ_w dw/dt = a(V−E_L) − w；放电时 V←V_reset、w += b。
 * 单位：pF、nS、mV、ms、pA。适应（w）就是放电久了会"累"——网络层有向通道要用到它。 */
export const LIF_DEFAULT = { C: 200, gL: 10, EL: -65, Vth: -50, Vreset: -65, tRef: 2, a: 0, b: 0, tauW: 200 };

export class LIFNeuron {
  constructor(p = {}, costs = { atpPerSpike: 0, atpRestPerMs: 0 }, ledger = null) {
    this.p = { ...LIF_DEFAULT, ...p };
    this.costs = costs;
    this.ledger = ledger;
    this.V = this.p.EL;
    this.w = 0;
    this.refUntil = -Infinity;
    this.t = 0;
    this.spikes = [];
  }

  /** gSyn：突触总电导 {g: nS, E: mV} 的列表或等效电流 Isyn（pA）；返回本步是否放电 */
  step(dt, Iext = 0, Isyn = 0) {
    const { C, gL, EL, Vth, Vreset, tRef, a, b, tauW } = this.p;
    this.ledger?.add('rest', this.costs.atpRestPerMs * dt);
    this.w += dt * (a * (this.V - EL) - this.w) / tauW;
    this.t += dt;
    if (this.t < this.refUntil) return false;
    this.V += dt * (-gL * (this.V - EL) - this.w + Isyn + Iext) / C;
    if (this.V >= Vth) {
      this.V = Vreset;
      this.w += b;
      this.refUntil = this.t + tRef;
      this.spikes.push(this.t);
      this.ledger?.add('spike', this.costs.atpPerSpike);
      return true;
    }
    return false;
  }
}

/** f–I：恒定电流下 durMs 内的放电频率（Hz） */
export function rateAt(p, I, { dt = 0.05, durMs = 1000 } = {}) {
  const n = new LIFNeuron(p);
  let k = 0;
  for (let t = 0; t < durMs; t += dt) if (n.step(dt, I)) k++;
  return (k * 1000) / durMs;
}
