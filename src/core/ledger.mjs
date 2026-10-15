/** 代谢账本：按类别累计 ATP 分子数。
 * 类别约定：rest（维持静息）、spike（动作电位）、syn_pre（递质释放与回收）、syn_post（突触后离子流）、drive（外部/上游驱动）。 */
const CATEGORIES = ['drive', 'rest', 'spike', 'syn_pre', 'syn_post'];
const INDEX = Object.fromEntries(CATEGORIES.map((category, i) => [category, i]));
export class EnergyLedger {
  constructor() { this._atp = new Map(); this._batch = false; this._sums = new Float64Array(5); this._present = new Uint8Array(5); }

  // 长循环暂存在固定类别的数值槽；每次加法顺序保持不变。读取公开 Map 时先同步。
  get atp() { this.flushBatch(); return this._atp; }
  set atp(value) { this._atp = value; this.syncBatch(); }

  beginBatch() {
    if (this._batch) return false;
    this._batch = true; this.syncBatch();
    return true;
  }
  flushBatch() { if (this._batch) for (let i = 0; i < 5; i++) if (this._present[i]) this._atp.set(CATEGORIES[i], this._sums[i]); }
  syncBatch() { if (this._batch) for (let i = 0; i < 5; i++) { this._present[i] = this._atp.has(CATEGORIES[i]) ? 1 : 0; this._sums[i] = this._atp.get(CATEGORIES[i]) ?? 0; } }
  endBatch() { this.flushBatch(); this._batch = false; }

  /** beginBatch 后使用的固定类别槽，省去每次脉冲账目的字符串查找。 */
  addAt(index, atp) {
    if (!(atp > 0)) return;
    if (!this._batch) { const category = CATEGORIES[index]; this._atp.set(category, (this._atp.get(category) ?? 0) + atp); return; }
    if (!this._present[index]) { this._present[index] = 1; this._atp.set(CATEGORIES[index], this._sums[index]); }
    this._sums[index] += atp;
  }

  add(category, atp) {
    if (!(atp > 0)) return;
    if (!this._batch) { this._atp.set(category, (this._atp.get(category) ?? 0) + atp); return; }
    const index = Object.hasOwn(INDEX, category) ? INDEX[category] : undefined;
    if (index !== undefined) this.addAt(index, atp);
    else this._atp.set(category, (this._atp.get(category) ?? 0) + atp);
  }

  get(category) { return this.atp.get(category) ?? 0; }

  total() { let s = 0; for (const v of this.atp.values()) s += v; return s; }

  /** 各类别占比（0–1） */
  shares() {
    const t = this.total() || 1;
    return Object.fromEntries([...this.atp].map(([k, v]) => [k, v / t]));
  }

  merge(other) { for (const [k, v] of other.atp) this.add(k, v); return this; }

  reset() { this._atp.clear(); this._sums.fill(0); this._present.fill(0); }

  /** 每秒 ATP（durationMs 为累计时长） */
  perSecond(durationMs) { return this.total() / (durationMs / 1000); }
}
