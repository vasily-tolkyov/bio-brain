/** 第 4 层的学习规则：第 3 层钙规则在群体层面的平均效果。
 * - 一起高频激活 → 两个方向都增强；一方活跃、另一方安静 → 减弱 ⇒ 协方差规则（对称）
 * - 先后激活（前在后前增强、后在前前减弱）⇒ 时间不对称规则：模式 μ 的神经元增强到模式 μ+1 的连接
 * 全局抑制（类似 APL / Δ7 的抑制性中间神经元）用统一的负权重 −J0 表示。 */
export function randomPatterns(P, N, f, rng) {
  return Array.from({ length: P }, () => {
    const idx = rng.shuffle([...Array(N).keys()]).slice(0, Math.round(f * N));
    const x = new Float64Array(N);
    for (const i of idx) x[i] = 1;
    return x;
  });
}

/** 突触强度有上限的版本：同属一个模式的两个神经元之间加固定增量 w（饱和在 w），全局抑制 −J0/N。
 * 一个集合要有足够多成员（k > θ/w）才能自己维持——小集合靠单个突触撑不起来 */
export function boundedHebbianWeights(patterns, { w = 0.05, J0 = 1 } = {}) {
  const N = patterns[0].length;
  const W = Array.from({ length: N }, () => new Float64Array(N));
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
    if (i === j) continue;
    W[i][j] = (patterns.some((p) => p[i] && p[j]) ? w : 0) - J0 / N;
  }
  return W;
}

/** 对称协方差规则 + 时间不对称项：W_ij = [Σ_μ (ξ_i^μ − f)(ξ_j^μ − f) + λ Σ_μ (ξ_i^{μ+1} − f)(ξ_j^μ − f)] / (N f (1−f)) − J0/N */
export function hebbianWeights(patterns, f, { lambda = 0, J0 = 0, cyclic = false } = {}) {
  const N = patterns[0].length, P = patterns.length, norm = 1 / (N * f * (1 - f));
  const W = Array.from({ length: N }, () => new Float64Array(N));
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
    if (i === j) continue;
    let s = 0;
    for (let m = 0; m < P; m++) {
      s += (patterns[m][i] - f) * (patterns[m][j] - f);
      const next = m + 1 < P ? m + 1 : cyclic ? 0 : -1;
      if (lambda && next >= 0) s += lambda * (patterns[next][i] - f) * (patterns[m][j] - f);
    }
    W[i][j] = s * norm - J0 / N;
  }
  return W;
}
