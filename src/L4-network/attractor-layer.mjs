/** 第 4 层的粗粒化"吸引子层"：一个单元 = 一个吸引子（一群互相强连接的神经元），输出 = 各吸引子的激活强度（0–1）。
 * 与脉冲版对应的三件事：
 * - 每个吸引子有一个原型（前馈权重 w_i），刺激 u 与原型越像越容易被激活（脉冲版：组内成员被直接驱动的比例）；
 * - 竞争：只有最像的 k 个能保持激活，强度按"超出第 k+1 名多少"分级（脉冲版：共享抑制，两个吸引子同时激活会互相削弱）；
 * - 侧向联系 L：同时激活的吸引子之间赫布增强（脉冲版：组间连接 / 有向通道），激活后再沉降几拍让联系参与。
 * 学习：前馈按 instar 规则把活跃吸引子的原型拉向当前刺激（竞争性赫布）；侧向按共同激活；
 * 很少赢的吸引子阈值下降（内在可塑性 / 稳态），否则会有永远不被用到的"死"吸引子。 */
import { makeRng } from '../core/rng.mjs';

const norm = (v) => { let s = 0; for (const x of v) s += x * x; s = Math.sqrt(s) || 1; return v.map((x) => x / s); };

export class AttractorLayer {
  constructor(D, N, { k = null, seed = 1, eta = 0.05, etaLat = 0.02, latGain = 0.2, settle = 6, etaTheta = 0.01 } = {}) {
    const rng = makeRng(seed);
    Object.assign(this, { D, N, k: k ?? Math.max(1, Math.round(N / 16)), eta, etaLat, latGain, settle, etaTheta });
    this.W = Array.from({ length: N }, () => norm(Float64Array.from({ length: D }, () => rng.uniform())));
    this.L = Array.from({ length: N }, () => new Float64Array(N));
    this.theta = new Float64Array(N);
    this.wins = new Float64Array(N);
  }

  /** 竞争：前 k 名保留，强度 = (h − 第 k+1 名) / (第 1 名 − 第 k+1 名) */
  kwta(h) {
    const idx = [...h.keys()].sort((a, b) => h[b] - h[a]);
    const top = h[idx[0]], floor = idx.length > this.k ? h[idx[this.k]] : Math.min(...h) - 1e-9;
    const r = new Float64Array(h.length);
    for (let q = 0; q < this.k; q++) { const i = idx[q]; r[i] = top > floor ? (h[i] - floor) / (top - floor) : 1; }
    return r;
  }

  response(u) {
    const un = norm(Float64Array.from(u));
    const s = this.W.map((w, i) => w.reduce((a, x, j) => a + x * un[j], 0) - this.theta[i]);
    let r = this.kwta(s);
    for (let t = 0; t < this.settle; t++) {
      const h = s.map((v, i) => { let a = v; const row = this.L[i]; for (let j = 0; j < this.N; j++) a += this.latGain * row[j] * r[j]; return a; });
      r = this.kwta(h);
    }
    return { r, un };
  }

  learn(u) {
    const { r, un } = this.response(u);
    for (let i = 0; i < this.N; i++) {
      if (r[i] > 0) { const w = this.W[i]; for (let j = 0; j < this.D; j++) w[j] += this.eta * r[i] * (un[j] - w[j]); this.W[i] = norm(w); }
      this.wins[i] = 0.99 * this.wins[i] + 0.01 * (r[i] > 0 ? 1 : 0);
      this.theta[i] += this.etaTheta * (this.wins[i] - this.k / this.N);
      if (r[i] > 0) { const row = this.L[i]; for (let j = 0; j < this.N; j++) if (j !== i) row[j] += this.etaLat * (r[i] * r[j] - row[j]); }
    }
    return r;
  }

  train(samples, epochs = 2) { for (let e = 0; e < epochs; e++) for (const u of samples) this.learn(u); return this; }
  encode(u) { return this.response(u).r; }
}

/** 读出的最优解（岭回归）：给定表示时线性读出能达到的最好误差——衡量"表示本身"的能力；
 * 三因素规则（trainReadout）是在线逼近同一个解的生物实现 */
export function fitReadout(X, y, { ridge = 1e-4 } = {}) {
  const D = X[0].length + 1;
  const A = Array.from({ length: D }, () => new Float64Array(D)), b = new Float64Array(D);
  for (let n = 0; n < X.length; n++) {
    const x = X[n];
    for (let i = 0; i < D; i++) {
      const xi = i < D - 1 ? x[i] : 1;
      if (xi === 0) continue;
      b[i] += xi * y[n];
      for (let j = 0; j < D; j++) { const xj = j < D - 1 ? x[j] : 1; if (xj) A[i][j] += xi * xj; }
    }
  }
  let tr = 0; for (let i = 0; i < D; i++) tr += A[i][i];
  for (let i = 0; i < D; i++) A[i][i] += ridge * (tr / D) + 1e-12;
  // Cholesky
  const Lc = Array.from({ length: D }, () => new Float64Array(D));
  for (let i = 0; i < D; i++) for (let j = 0; j <= i; j++) {
    let s = A[i][j];
    for (let q = 0; q < j; q++) s -= Lc[i][q] * Lc[j][q];
    Lc[i][j] = i === j ? Math.sqrt(Math.max(s, 1e-15)) : s / Lc[j][j];
  }
  const z = new Float64Array(D), v = new Float64Array(D);
  for (let i = 0; i < D; i++) { let s = b[i]; for (let q = 0; q < i; q++) s -= Lc[i][q] * z[q]; z[i] = s / Lc[i][i]; }
  for (let i = D - 1; i >= 0; i--) { let s = z[i]; for (let q = i + 1; q < D; q++) s -= Lc[q][i] * v[q]; v[i] = s / Lc[i][i]; }
  return (x) => { let p = v[D - 1]; for (let j = 0; j < D - 1; j++) p += v[j] * x[j]; return p; };
}

/** 读出：预测误差（三因素）规则学的线性组合 y = v·[r, 1]；步长按同时活跃的输入数归一。返回训练好的预测函数 */
export function trainReadout(X, y, { epochs = 60, alpha = 0.5 } = {}) {
  const D = X[0].length + 1, v = new Float64Array(D);
  const aug = (x) => { const a = new Float64Array(D); a.set(x); a[D - 1] = 1; return a; };
  const Xa = X.map(aug);
  for (let e = 0; e < epochs; e++) {
    const a = alpha / (1 + e * 0.05);
    for (let n = 0; n < Xa.length; n++) {
      const x = Xa[n]; let p = 0, nn = 0;
      for (let j = 0; j < D; j++) { p += v[j] * x[j]; nn += x[j] * x[j]; }
      const d = (y[n] - p) * a / (nn || 1);
      for (let j = 0; j < D; j++) v[j] += d * x[j];
    }
  }
  return (x) => { const xa = aug(x); let p = 0; for (let j = 0; j < D; j++) p += v[j] * xa[j]; return p; };
}
