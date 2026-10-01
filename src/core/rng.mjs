/** 可复现随机数：mulberry32 + 正态 + 泊松 */
export function makeRng(seed = 1) {
  let s = seed >>> 0;
  const uniform = () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  let spare = null;
  const normal = () => {
    if (spare !== null) { const v = spare; spare = null; return v; }
    let u = 0, v = 0;
    while (u === 0) u = uniform();
    v = uniform();
    const r = Math.sqrt(-2 * Math.log(u));
    spare = r * Math.sin(2 * Math.PI * v);
    return r * Math.cos(2 * Math.PI * v);
  };
  /** 一个时间步内发生事件的概率（泊松过程，rateHz × dtMs） */
  const bernoulliRate = (rateHz, dtMs) => uniform() < 1 - Math.exp(-rateHz * dtMs / 1000);
  const pick = (n) => Math.floor(uniform() * n);
  const shuffle = (arr) => { for (let i = arr.length - 1; i > 0; i--) { const j = pick(i + 1); [arr[i], arr[j]] = [arr[j], arr[i]]; } return arr; };
  return { uniform, normal, bernoulliRate, pick, shuffle };
}
