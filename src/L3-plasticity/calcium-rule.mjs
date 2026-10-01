/** 第 3 层：钙控制的赫布学习（Graupner & Brunel 2012 的突触效能模型）。
 * 突触里的钙 c(t) 由两路组成：突触前脉冲经 NMDA 受体带来的钙（延迟 D 后到达，幅度 Cpre），突触后放电反传到树突、经电压门控钙通道带来的钙（Cpost）。
 * 钙越过高阈值 θp 的时间推动增强，越过低阈值 θd 的时间推动减弱：
 *   τ dρ/dt = −ρ(1−ρ)(ρ*−ρ) + γp(1−ρ)·[c > θp] − γd·ρ·[c > θd]
 * 效能 ρ ∈ [0,1] 映射成突触的量子电导缩放（第 2 层的 scale）。
 * eta > 0 时加上 NMDA 的"同时检测"：突触后放电时，已结合递质的 NMDA 受体（衰减 τN）被反传电位挪开镁塞，多进 eta·Cpre·(结合比例) 的钙。 */
export const GB_DP = { tauCa: 20, Cpre: 1, Cpost: 2, thetaD: 1, thetaP: 1.3, gammaD: 200, gammaP: 321.808, tau: 150000, rhoStar: 0.5, D: 13.7, eta: 0, tauN: 40 };

/** 网络用的 NMDA 依赖版：
 * - 只有突触前放电：钙（Cpre 0.8，18 Hz 时峰值约 0.86）低于 θd → 不变；
 * - 只有突触后放电：钙（Cpost 1.0）刚越过 θd → 轻微减弱（异突触压抑）；
 * - 突触前先放、突触后随即放电：已结合递质的 NMDA 被反传电位挪开镁塞，同时检测项 η·Cpre·(结合比例) 越过 θp → 增强；
 * - 突触后先放、突触前随后：延迟到达的突触前钙叠在正在衰减的突触后钙上，落进 θd–θp → 特异性减弱（经典 STDP 的负半边，有向通道靠它）。
 * GB_DP 是为一对一配对实验拟合的：单个突触后脉冲的钙（2）就超过 θp，放进网络后任何放电神经元的全部输入都被推到 ρ ≈ 0.5，组内组间分不开 */
export const GB_NMDA = { ...GB_DP, Cpre: 0.8, Cpost: 1.0, eta: 2 };

export class CalciumPlasticity {
  constructor(p = {}, { rho0 = 0.5, accel = 1 } = {}) {
    this.p = { ...GB_DP, ...p };
    this.rho = rho0;
    this.accel = accel;
    this.c = 0;
    this.nmda = 0;
    this.pending = [];
    this.timeAboveP = 0;
    this.timeAboveD = 0;
  }

  /** 递质一释放就与 NMDA 受体结合（结合比例立刻 +1）；静息时经 NMDA 缓慢进入的那份钙延迟 D 才到 */
  preSpike(t) { this.pending.push(t + this.p.D); this.nmda += 1; }

  postSpike() {
    this.c += this.p.Cpost + this.p.eta * this.p.Cpre * this.nmda;
  }

  step(dt, t) {
    const { tauCa, thetaD, thetaP, gammaD, gammaP, tau, rhoStar, tauN, Cpre } = this.p;
    while (this.pending.length && this.pending[0] <= t) { this.pending.shift(); this.c += Cpre; }
    const r = this.rho;
    let drive = -r * (1 - r) * (rhoStar - r);
    if (this.c > thetaP) { drive += gammaP * (1 - r); this.timeAboveP += dt; }
    if (this.c > thetaD) { drive -= gammaD * r; this.timeAboveD += dt; }
    this.rho = Math.min(1, Math.max(0, r + (dt * this.accel * drive) / tau));
    this.c *= Math.exp(-dt / tauCa);
    this.nmda *= Math.exp(-dt / tauN);
  }
}

/** 标准配对实验：pairs 次"突触前在 0、突触后在 Δt"的配对，频率 freq（Hz）；返回效能变化 Δρ（从 rho0 出发，确定性） */
export function pairingChange(deltaT, { pairs = 60, freq = 1, rho0 = 0.5, dt = 0.1, p = {} } = {}) {
  const pl = new CalciumPlasticity(p, { rho0 });
  const period = 1000 / freq;
  const lead = Math.max(0, -deltaT) + 1;
  const T = pairs * period;
  let nextPre = lead, nextPost = lead + deltaT, k = 0, j = 0;
  for (let t = 0; t < T + 200; t += dt) {
    if (k < pairs && t >= nextPre) { pl.preSpike(t); k++; nextPre += period; }
    if (j < pairs && t >= nextPost) { pl.postSpike(); j++; nextPost += period; }
    pl.step(dt, t);
  }
  return pl.rho - rho0;
}
