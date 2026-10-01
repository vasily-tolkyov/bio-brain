/** 第 5 层：皮层的规则学习（R2/R3 的生物对应）。
 * 预测误差驱动的三因素规则：Δw = α·δ·x，其中 δ = 实际结果 − 预测（多巴胺式的"比预期好/差"信号），x = 突触前活动（资格）。
 * 这就是 Rescorla–Wagner：一个线索只有在它能解释"别的线索解释不了"的那部分结果时才被学进去——
 *   这给出了"效应臂 vs 对照臂"的差分（伴随出现但单独出现时没有结果的线索被剔除）与阻断（已被预测的结果不再教新线索）。
 * 合取规则（"状态 s 且动作 a 才有结果"）线性学不了；先经稀疏随机扩张（蕈状体 / 齿状回式），再用同一规则就能学。 */
import { kWTA } from './hippocampus.mjs';
import { makeRng } from '../core/rng.mjs';

export class PredictionErrorLearner {
  constructor(nIn, nOut = 1, { alpha = 0.1 } = {}) {
    this.W = Array.from({ length: nOut }, () => new Float64Array(nIn));
    this.alpha = alpha;
  }
  predict(x) { return this.W.map((row) => row.reduce((s, w, j) => s + w * x[j], 0)); }
  /** 步长按同时活跃的输入数归一（α·Σx² < 2 才稳定，稀疏码里同时活跃的特征多时尤其要紧） */
  learn(x, target) {
    const y = this.predict(x), t = Array.isArray(target) || ArrayBuffer.isView(target) ? target : [target];
    const delta = y.map((v, k) => t[k] - v);
    const norm = Math.max(1, x.reduce((s, v) => s + v * v, 0));
    this.W.forEach((row, k) => { for (let j = 0; j < row.length; j++) row[j] += (this.alpha * delta[k] * x[j]) / norm; });
    return delta;
  }
}

/** 稀疏随机扩张：每个扩张单元随机取 fan 个输入，取 k 个赢者（全局抑制）。线性不可分的合取在扩张空间里变得可分 */
export class SparseExpansion {
  constructor(nIn, nExp, { fan = 3, k = 0.05, seed = 1 } = {}) {
    const rng = makeRng(seed);
    this.k = Math.max(1, Math.round(k * nExp));
    this.W = Array.from({ length: nExp }, () => { const row = new Float64Array(nIn); for (const j of rng.shuffle([...Array(nIn).keys()]).slice(0, fan)) row[j] = 1; return row; });
  }
  encode(x) { return kWTA(this.W.map((row) => row.reduce((s, w, j) => s + w * x[j], 0) + 1e-9 * row.reduce((s, w, j) => s + w * j, 0)), this.k); }
}
