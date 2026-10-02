/** 多层脉冲网络：每一层都是第 4 层（脉冲版）那样的完整网络——
 *   100 个兴奋性 + 25 个抑制性 LIF 神经元；层内兴奋→兴奋突触按钙规则学习（+ 异突触竞争），抑制→兴奋按抑制性可塑性学习；
 *   每个兴奋性神经元有背景泊松输入。层内的吸引子与有向通道都是学出来的。
 * 层与层之间（都是真实突触）：
 *   前馈 ↑：第 k 层兴奋 → 第 k+1 层兴奋，按组对应的拓扑投射（同组概率 ffSame，异组 ffOther），强度固定；
 *          另有第 k 层兴奋 → 第 k+1 层抑制的前馈抑制；
 *   反馈 ↓：第 k+1 层兴奋 → 第 k 层兴奋，随机连接，起点很弱，按钙规则学习（+ 异突触竞争）。
 * 上层的异突触预算可以单独设（upperBudget）：上层的吸引子由下层几个吸引子合成，更大，组内连接会占掉更多预算；
 *   真实皮层沿层级往上，锥体神经元的树突棘也越多（Elston 等人的测量）。
 * 连接按层依次生成：K 层网络的前 k 层与 k 层网络的连接完全相同，所以可以逐层训练、逐层加层。 */
import { SpikingNetwork } from './spiking-network.mjs';
import { mammalianCosts } from '../L1-neuron/hh.mjs';

export const LAYER_DEFAULTS = { ffSame: 0.5, ffOther: 0.02, ffScale: 6, ffiP: 0.3, ffiScale: 1.5, fbP: 0.3, fbMax: 6, fbRho0: 0.05, fbBudget: 0.15 };

/** cfg：单层参数（与 demo/fig2-network.mjs 的 CFG 相同：NE、NI、K 组 × M 个神经元、pConn、contacts、sMax、rho0、budget、sie、istdpEta、rule、bg、seed）+ 层间参数 */
export function buildLayered(nLayers, cfg) {
  const c = { ...LAYER_DEFAULTS, ...cfg };
  const pops = [];
  for (let k = 1; k <= nLayers; k++) pops.push({ name: `E${k}`, n: c.NE, p: { b: 20, tauW: 300 } }, { name: `I${k}`, n: c.NI, p: { C: 100, tRef: 1 } });
  const net = new SpikingNetwork({ costs: mammalianCosts(), seed: c.seed, plasticity: c.rule, pops });
  const block = (a) => Math.min(Math.floor(a / c.M), c.K);
  const layers = [];
  for (let k = 1; k <= nLayers; k++) {
    const L = { k, E: net.idx(`E${k}`), I: net.idx(`I${k}`) };
    L.EE = net.connect({ from: `E${k}`, to: `E${k}`, p: c.pConn, contacts: c.contacts, pRelease: 0.3, tauRec: 100, plastic: { sMin: 0.1, sMax: c.sMax, rho0: c.rho0, accel: 5, budget: k > 1 ? c.upperBudget ?? c.budget : c.budget } });
    L.EI = net.connect({ from: `E${k}`, to: `I${k}`, p: 0.5, contacts: 10, pRelease: 0.3, scale: 1.5 });
    L.IE = net.connect({ from: `I${k}`, to: `E${k}`, p: 0.5, contacts: 10, pRelease: 0.5, tauRec: 300, kind: 'inh', scale: c.sie, istdp: { eta: c.istdpEta, target: 5 } });
    if (k > 1) {
      const lo = layers[k - 2], pairs = [];
      for (let a = 0; a < c.NE; a++) for (let b = 0; b < c.NE; b++) if (net.rng.uniform() < (block(a) === block(b) ? c.ffSame : c.ffOther)) pairs.push([lo.E[a], L.E[b]]);
      L.FF = net.connect({ pairs, contacts: 10, pRelease: 0.3, tauRec: 100, scale: c.ffScale });
      L.FFI = net.connect({ from: `E${k - 1}`, to: `I${k}`, p: c.ffiP, contacts: 10, pRelease: 0.3, scale: c.ffiScale });
      L.FB = net.connect({ from: `E${k}`, to: `E${k - 1}`, p: c.fbP, contacts: 10, pRelease: 0.3, tauRec: 100, plastic: { sMin: 0, sMax: c.fbMax, rho0: c.fbRho0, accel: 5, budget: c.fbBudget } });
    }
    layers.push(L);
  }
  const layerOf = new Int32Array(net.N).fill(-1);
  layers.forEach((L, k) => L.E.forEach((i) => (layerOf[i] = k)));
  return { net, layers, cfg: c, layerOf };
}

/** 只让给定的组学习，其余冻结 */
export function learnOnly(stack, groups) { for (const g of stack.net.groups) g.frozen = true; for (const g of groups) if (g) g.frozen = false; }

const setRho = (g, rho) => { g.rho.set(rho); const { sMin, sMax } = g.plastic; for (let k = 0; k < rho.length; k++) g.scale[k] = sMin + (sMax - sMin) * g.rho[k]; };
/** 可学习的状态：每层的兴奋→兴奋效能、抑制强度、到下一层的反馈效能 */
export const saveState = (stack) => stack.layers.map((L) => ({ EE: [...L.EE.rho], IE: [...L.IE.scale], FB: L.FB ? [...L.FB.rho] : null }));
export function loadState(stack, state) {
  state.forEach((s, k) => { const L = stack.layers[k]; if (!L) return; setRho(L.EE, s.EE); L.IE.scale.set(s.IE); if (s.FB && L.FB) setRho(L.FB, s.FB); });
}

/** 外部输入：第一层兴奋性神经元按 drives（层内编号 → 每条输入的 Hz），其余兴奋性神经元背景，抑制性神经元没有外部输入；
 * extra：直接给任意神经元（全局编号 → Hz），例如不经第一层、直接驱动上层（"想象"测试） */
export function inputRate(stack, drives = new Map(), extra = null) {
  const arr = new Float64Array(stack.net.N);
  stack.layers.forEach((L, k) => L.E.forEach((i, a) => (arr[i] = k === 0 ? drives.get(a) ?? stack.cfg.bg : stack.cfg.bg)));
  if (extra) for (const [i, hz] of extra) arr[i] = hz;
  return (i) => arr[i];
}

/** [t0, t1) 内各神经元的放电时刻（放电记录按时间追加，从尾部往前找） */
export function spikesIn(net, ids, t0, t1) {
  return ids.map((i) => { const s = net.spikes[i], out = []; for (let q = s.length - 1; q >= 0 && s[q] >= t0; q--) if (s[q] < t1) out.push(s[q]); return out.reverse(); });
}
