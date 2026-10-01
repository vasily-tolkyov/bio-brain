/** 第 5 层：前额叶工作记忆（门控的吸引子）。每个槽位 = M 个"条目"群体，群体自激（持续活动 = 吸引子）、
 * 群体之间互相抑制（同一槽位一次只记一个条目）。门（多巴胺 / 基底节式的门控信号）决定外部输入能否进入：
 * 门关时输入被挡在外面，旧条目靠自己的吸引子维持；门开时新输入能把状态推到另一个吸引子。 */
export class PrefrontalWM {
  constructor({ slots = 3, items = 6, wSelf = 0.9, wInh = 0.8, theta = 0.45, beta = 20, tau = 1 } = {}) {
    Object.assign(this, { slots, items, wSelf, wInh, theta, beta, tau });
    this.r = Array.from({ length: slots }, () => new Float64Array(items));
  }

  /** gates[s] ∈ [0,1]；inputs[s] = 条目编号或 null */
  step(dt, { gates = [], inputs = [], inputGain = 2 } = {}) {
    for (let s = 0; s < this.slots; s++) {
      const r = this.r[s], tot = r.reduce((a, v) => a + v, 0), g = gates[s] ?? 0;
      const next = new Float64Array(this.items);
      for (let m = 0; m < this.items; m++) {
        const ext = inputs[s] === m ? inputGain * g : 0;
        const u = this.wSelf * r[m] - this.wInh * (tot - r[m]) + ext - this.theta;
        next[m] = r[m] + (dt / this.tau) * (1 / (1 + Math.exp(-this.beta * u)) - r[m]);
      }
      this.r[s] = next;
    }
  }

  held(s) { const r = this.r[s]; const m = r.indexOf(Math.max(...r)); return r[m] > 0.5 ? m : null; }
  activity() { return this.r.flatMap((r) => [...r]); }
}
