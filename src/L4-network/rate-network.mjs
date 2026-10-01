/** 第 4 层：神经元群的速率网络（每个单元 = 一个神经元的放电率，取 0–1 倍的最高放电率）。
 *   τ dr/dt = −r + g(W r + b + I − a)，g(u) = 1/(1 + e^{−βu})
 *   τ_a da/dt = −a + k_a·r（适应 / 疲劳：放电久了自己的阈值升高）
 * 能量（动力学账）：E(r) = −½ rᵀ W_s r − (b + I − a)ᵀ r + (1/β) Σ [r ln r + (1−r) ln(1−r)]
 *   W_s 是 W 的对称部分。W 对称、a 固定时 dE/dt ≤ 0（连续时间 Hopfield），E 的极小值就是吸引子；
 *   W 的反对称部分 W_a 不能写进任何能量函数——它是让状态沿固定方向流动的"有向通道"。 */
export class RateNetwork {
  constructor(N, { beta = 20, tau = 1, tauA = 30, kA = 0, bias = -0.25, seed = null } = {}) {
    this.N = N;
    this.W = Array.from({ length: N }, () => new Float64Array(N));
    this.b = new Float64Array(N).fill(bias);
    this.r = new Float64Array(N);
    this.a = new Float64Array(N);
    Object.assign(this, { beta, tau, tauA, kA });
  }

  setWeights(W) { for (let i = 0; i < this.N; i++) this.W[i].set(W[i]); }

  input(I = null) {
    const u = new Float64Array(this.N);
    for (let i = 0; i < this.N; i++) {
      const row = this.W[i];
      let s = this.b[i] - this.a[i] + (I ? I[i] : 0);
      for (let j = 0; j < this.N; j++) s += row[j] * this.r[j];
      u[i] = s;
    }
    return u;
  }

  g(u) { return 1 / (1 + Math.exp(-this.beta * u)); }

  /** 一步（Euler）；noise：每步加在每个单元输入上的高斯扰动标准差（需要 rng） */
  step(dt, I = null, { noise = 0, rng = null } = {}) {
    const u = this.input(I);
    for (let i = 0; i < this.N; i++) {
      const ui = noise && rng ? u[i] + noise * rng.normal() : u[i];
      this.r[i] += (dt / this.tau) * (this.g(ui) - this.r[i]);
      if (this.kA) this.a[i] += (dt / this.tauA) * (this.kA * this.r[i] - this.a[i]);
    }
  }

  /** 动力学能量（只用对称部分；a 作为此刻的阈值抬升计入） */
  energy(I = null, r = this.r) {
    const N = this.N, eps = 1e-12;
    let quad = 0, lin = 0, ent = 0;
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) quad += 0.5 * (this.W[i][j] + this.W[j][i]) * r[i] * r[j];
      lin += (this.b[i] - this.a[i] + (I ? I[i] : 0)) * r[i];
      const x = Math.min(1 - eps, Math.max(eps, r[i]));
      ent += x * Math.log(x) + (1 - x) * Math.log(1 - x);
    }
    return -0.5 * quad - lin + ent / this.beta;
  }

  /** W 的对称 / 反对称部分的范数比：0 = 完全对称，1 = 完全反对称 */
  asymmetry() {
    let s = 0, a = 0;
    for (let i = 0; i < this.N; i++) for (let j = 0; j < this.N; j++) {
      const sym = 0.5 * (this.W[i][j] + this.W[j][i]), anti = 0.5 * (this.W[i][j] - this.W[j][i]);
      s += sym * sym; a += anti * anti;
    }
    return Math.sqrt(a / (s + a || 1));
  }
}

/** 与模式 ξ（0/1，稀疏度 f）的重叠：1 = 完全是这个模式，0 = 无关 */
export function overlap(r, xi, f) {
  let s = 0;
  for (let i = 0; i < r.length; i++) s += (xi[i] - f) * r[i];
  return s / (xi.length * f * (1 - f));
}
