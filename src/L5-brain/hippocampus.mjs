/** 第 5 层：海马（快速情节记忆）。用 k 个赢者（k-WTA）代表强反馈抑制（篮状细胞）把活动压成稀疏码。
 *   EC（内嗅皮层输入）→ DG：随机稀疏投射 + 极稀疏的赢者 —— 模式分离：相似的输入被拉开
 *   DG → CA3：苔藓纤维（少而强，"引爆"CA3）；编码时 CA3 内部一次性赫布学习（同时活跃的互连）——吸引子
 *   EC → CA3：编码时一并学，提取时用残缺线索从这里进入，再由 CA3 的回返连接补全
 *   CA3 → CA1：学成异联想；EC → CA1 直接给出"实际输入"的 CA1 码；两者不一致 = 新奇
 *   连续编码的情节在 CA3 里留下不对称连接（前一项 → 后一项）；休息时噪声 + 疲劳让它按顺序重放（有向通道） */
import { makeRng } from '../core/rng.mjs';

export function kWTA(u, k) {
  const idx = [...u.keys()].sort((a, b) => u[b] - u[a]).slice(0, k).filter((i) => u[i] > 0);
  const x = new Float64Array(u.length);
  for (const i of idx) x[i] = 1;
  return x;
}
export const overlapFrac = (a, b) => { let s = 0, n = 0; for (let i = 0; i < a.length; i++) { s += a[i] * b[i]; n += a[i]; } return n ? s / n : 0; };
const mv = (W, x) => W.map((row) => row.reduce((s, w, j) => s + w * x[j], 0));
const zeros = (n, m) => Array.from({ length: n }, () => new Float64Array(m));
const randomFan = (rows, cols, fan, rng, w = 1) => { const W = zeros(rows, cols); for (let i = 0; i < rows; i++) for (const j of rng.shuffle([...Array(cols).keys()]).slice(0, fan)) W[i][j] = w; return W; };

export class Hippocampus {
  constructor({ nEC = 100, nDG = 500, nCA3 = 200, nCA1 = 100, kDG = 10, kCA3 = 12, kCA1 = 12, fanEC = 10, fanMF = 3, seed = 1 } = {}) {
    const rng = makeRng(seed);
    Object.assign(this, { nEC, nDG, nCA3, nCA1, kDG, kCA3, kCA1, rng });
    this.ecDG = randomFan(nDG, nEC, fanEC, rng);
    this.dgCA3 = zeros(nCA3, nDG);
    for (let j = 0; j < nDG; j++) for (const i of rng.shuffle([...Array(nCA3).keys()]).slice(0, fanMF)) this.dgCA3[i][j] = 1;
    this.ecCA1 = randomFan(nCA1, nEC, 15, rng);
    this.ecCA3 = zeros(nCA3, nEC);
    this.ca3Rec = zeros(nCA3, nCA3);
    this.ca3Seq = zeros(nCA3, nCA3);
    this.ca3CA1 = zeros(nCA1, nCA3);
    this.ca1EC = zeros(nEC, nCA1);
    this.episodes = [];
  }

  dg(ec) { return kWTA(mv(this.ecDG, ec), this.kDG); }
  ca1FromEC(ec) { return kWTA(mv(this.ecCA1, ec), this.kCA1); }

  /** 编码一次（一次性学习）：返回各区的码；prevCA3 给出时，学前一项 → 这一项的不对称连接 */
  encode(ec, prevCA3 = null) {
    const dg = this.dg(ec);
    const ca3 = kWTA(mv(this.dgCA3, dg).map((v, i) => v + 1e-6 * this.rng.uniform()), this.kCA3);
    const ca1 = this.ca1FromEC(ec);
    for (let i = 0; i < this.nCA3; i++) {
      if (!ca3[i]) continue;
      for (let j = 0; j < this.nCA3; j++) if (ca3[j] && i !== j) this.ca3Rec[i][j] = 1;
      for (let j = 0; j < this.nEC; j++) if (ec[j]) this.ecCA3[i][j] = 1;
      if (prevCA3) for (let j = 0; j < this.nCA3; j++) if (prevCA3[j] && !ca3[j]) this.ca3Seq[i][j] = 1;
    }
    for (let i = 0; i < this.nCA1; i++) if (ca1[i]) for (let j = 0; j < this.nCA3; j++) if (ca3[j]) this.ca3CA1[i][j] = 1;
    for (let i = 0; i < this.nEC; i++) if (ec[i]) for (let j = 0; j < this.nCA1; j++) if (ca1[j]) this.ca1EC[i][j] = 1;
    const codes = { ec, dg, ca3, ca1 };
    this.episodes.push(codes);
    return codes;
  }

  encodeSequence(list) { let prev = null; return list.map((ec) => { const c = this.encode(ec, prev); prev = c.ca3; return c; }); }

  /** 从残缺线索提取：EC→CA3 给出初始码，CA3 回返连接迭代补全，再经 CA1 还原 EC */
  recall(cue, { iters = 6 } = {}) {
    let ca3 = kWTA(mv(this.ecCA3, cue), this.kCA3);
    for (let k = 0; k < iters; k++) ca3 = kWTA(mv(this.ca3Rec, ca3).map((v, i) => v + 0.5 * mv(this.ecCA3, cue)[i] / (cue.reduce((s, x) => s + x, 0) || 1)), this.kCA3);
    const ca1 = kWTA(mv(this.ca3CA1, ca3), this.kCA1);
    const ecOut = kWTA(mv(this.ca1EC, ca1), Math.round(cue.length * 0.2));
    return { ca3, ca1, ecOut };
  }

  /** 新奇：用当前输入作线索，CA3 提取出的 CA1 预测与 EC 直接给出的 CA1 码有多不一致（0 熟悉，1 全新） */
  novelty(ec) { return 1 - overlapFrac(this.recall(ec).ca1, this.ca1FromEC(ec)); }

  /** 休息时重放：从第一项的 CA3 码出发，回返（对称）+ 序列（不对称）+ 疲劳，记录依次最像哪一项 */
  replay({ start, steps = 40, seqGain = 0.8, fatigue = 0.4, tauF = 3 } = {}) {
    let x = start.slice();
    const a = new Float64Array(this.nCA3), visits = [];
    for (let t = 0; t < steps; t++) {
      const rec = mv(this.ca3Rec, x), seq = mv(this.ca3Seq, x);
      const u = rec.map((v, i) => v + seqGain * seq[i] - fatigue * a[i] * this.kCA3 + 1e-6 * this.rng.uniform());
      for (let i = 0; i < this.nCA3; i++) a[i] += (x[i] - a[i]) / tauF;
      x = kWTA(u, this.kCA3);
      const ov = this.episodes.map((e) => overlapFrac(x, e.ca3));
      const b = ov.indexOf(Math.max(...ov));
      if (ov[b] > 0.6 && visits.at(-1) !== b) visits.push(b);
    }
    return visits;
  }
}
