/** 代谢账本：按类别累计 ATP 分子数。
 * 类别约定：rest（维持静息）、spike（动作电位）、syn_pre（递质释放与回收）、syn_post（突触后离子流）、drive（外部/上游驱动）。 */
export class EnergyLedger {
  constructor() { this.atp = new Map(); }

  add(category, atp) {
    if (!(atp > 0)) return;
    this.atp.set(category, (this.atp.get(category) ?? 0) + atp);
  }

  get(category) { return this.atp.get(category) ?? 0; }

  total() { let s = 0; for (const v of this.atp.values()) s += v; return s; }

  /** 各类别占比（0–1） */
  shares() {
    const t = this.total() || 1;
    return Object.fromEntries([...this.atp].map(([k, v]) => [k, v / t]));
  }

  merge(other) { for (const [k, v] of other.atp) this.add(k, v); return this; }

  reset() { this.atp.clear(); }

  /** 每秒 ATP（durationMs 为累计时长） */
  perSecond(durationMs) { return this.total() / (durationMs / 1000); }
}
