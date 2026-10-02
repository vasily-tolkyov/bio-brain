/** 多层脉冲网络：每一层都是第 4 层（脉冲版）那样的完整网络——
 *   100 个兴奋性 + 25 个抑制性 LIF 神经元；层内兴奋→兴奋突触按钙规则学习（+ 异突触竞争），抑制→兴奋按抑制性可塑性学习；
 *   每个兴奋性神经元有背景泊松输入。层内的吸引子与有向通道都是学出来的。
 * 层与层之间（都是真实突触）：
 *   前馈 ↑：第 k 层兴奋 → 第 k+1 层兴奋，按组对应的拓扑投射（同组概率 ffSame，异组 ffOther），强度固定；
 *          另有第 k 层兴奋 → 第 k+1 层抑制的前馈抑制；
 *   反馈 ↓：第 k+1 层兴奋 → 第 k 层兴奋，随机连接，起点很弱，按钙规则学习（+ 异突触竞争）。
 * apical：兴奋性神经元带顶树突分区（见 spiking-network.mjs 的 APICAL_DEFAULT），反馈全部落在顶树突上，
 *   前馈与层内连接落在胞体——反馈只做阈下调制，与胞体输入同时到来时触发树突平台、把放电变成爆发；
 * 上层的异突触预算可以单独设（upperBudget）：上层的吸引子由下层几个吸引子合成，更大，组内连接会占掉更多预算；
 *   真实皮层沿层级往上，锥体神经元的树突棘也越多（Elston 等人的测量）。
 * 连接按层依次生成：K 层网络的前 k 层与 k 层网络的连接完全相同，所以可以逐层训练、逐层加层。 */
import { SpikingNetwork } from './spiking-network.mjs';
import { mammalianCosts } from '../L1-neuron/hh.mjs';

export const LAYER_DEFAULTS = { ffSame: 0.5, ffOther: 0.02, ffScale: 6, ffiP: 0.3, ffiScale: 1.5, fbP: 0.3, fbMax: 6, fbRho0: 0.05, fbBudget: 0.15 };
/** 深层神经元群（deep）：膜时间常数 40 ms、放电适应弱、自身的抑制弱——当前状态的拷贝能保持几十毫秒、多个吸引子可以同时活跃（当前的 + 预测的）；
 * 拷贝：本层浅层 → 深层，按组对应（copySame），强度固定，以 NMDA 为主（慢、电压依赖：浅层一次爆发让深层持续去极化几十毫秒）；
 * 本地预测：深层 → 本层浅层的顶树突，按组对应（localSame），强度固定 */
export const DEEP_DEFAULTS = { deepP: { C: 400, b: 5, tauW: 300 }, copySame: 0.5, copyScale: 20, copyKind: 'nmda', localSame: 0.3, localScale: 3, dIE: 3, deepBudget: 0.3 };

/** cfg：单层参数（与 demo/fig2-network.mjs 的 CFG 相同：NE、NI、K 组 × M 个神经元、pConn、contacts、sMax、rho0、budget、sie、istdpEta、rule、bg、seed）+ 层间参数；
 * deep：每层再加一群深层神经元（见上），层间反馈改由深层发出：落到下一层浅层的顶树突（FBA）与下一层深层（FBD），都按钙规则学习 */
export function buildLayered(nLayers, cfg) {
  const c = { ...LAYER_DEFAULTS, ...DEEP_DEFAULTS, ...cfg };
  const pops = [];
  for (let k = 1; k <= nLayers; k++) {
    pops.push({ name: `E${k}`, n: c.NE, p: { b: 20, tauW: 300 }, apical: c.apical || c.deep || undefined }, { name: `I${k}`, n: c.NI, p: { C: 100, tRef: 1 } });
    if (c.deep) pops.push({ name: `D${k}`, n: c.NE, p: c.deepP, adp: c.deepAdp }, { name: `DI${k}`, n: c.NI, p: { C: 100, tRef: 1 } });
  }
  const net = new SpikingNetwork({ costs: mammalianCosts(), seed: c.seed, plasticity: c.rule, pops });
  const block = (a) => Math.min(Math.floor(a / c.M), c.K);
  const topo = (pre, post, pSame, pOther = 0) => { const pairs = []; for (let a = 0; a < pre.length; a++) for (let b = 0; b < post.length; b++) if (net.rng.uniform() < (block(a) === block(b) ? pSame : pOther)) pairs.push([pre[a], post[b]]); return pairs; };
  const fbPlastic = { sMin: 0, sMax: c.fbMax, rho0: c.fbRho0, accel: 5, budget: c.fbBudget };
  const layers = [];
  for (let k = 1; k <= nLayers; k++) {
    const L = { k, E: net.idx(`E${k}`), I: net.idx(`I${k}`) };
    L.EE = net.connect({ from: `E${k}`, to: `E${k}`, p: c.pConn, contacts: c.contacts, pRelease: 0.3, tauRec: 100, plastic: { sMin: 0.1, sMax: c.sMax, rho0: c.rho0, accel: 5, budget: k > 1 ? c.upperBudget ?? c.budget : c.budget } });
    L.EI = net.connect({ from: `E${k}`, to: `I${k}`, p: 0.5, contacts: 10, pRelease: 0.3, scale: 1.5 });
    L.IE = net.connect({ from: `I${k}`, to: `E${k}`, p: 0.5, contacts: 10, pRelease: 0.5, tauRec: 300, kind: 'inh', scale: c.sie, istdp: { eta: c.istdpEta, target: 5 } });
    if (c.deep) {
      L.D = net.idx(`D${k}`); L.DI = net.idx(`DI${k}`);
      L.DEE = net.connect({ from: `D${k}`, to: `D${k}`, p: c.pConn, contacts: c.contacts, pRelease: 0.3, tauRec: 100, plastic: { sMin: 0.1, sMax: c.sMax, rho0: c.rho0, accel: 5, budget: c.deepBudget } });
      L.DEI = net.connect({ from: `D${k}`, to: `DI${k}`, p: 0.5, contacts: 10, pRelease: 0.3, scale: 1.5 });
      L.DIE = net.connect({ from: `DI${k}`, to: `D${k}`, p: 0.5, contacts: 10, pRelease: 0.5, tauRec: 300, kind: 'inh', scale: c.dIE, istdp: { eta: c.istdpEta, target: 5 } });
      L.COPY = net.connect({ pairs: topo(L.E, L.D, c.copySame), contacts: 10, pRelease: 0.3, tauRec: 100, scale: c.copyScale, kind: c.copyKind });
      L.LOCAL = net.connect({ pairs: topo(L.D, L.E, c.localSame), contacts: 10, pRelease: 0.3, tauRec: 100, scale: c.localScale, target: 'apical' });
    }
    if (k > 1) {
      const lo = layers[k - 2];
      L.FF = net.connect({ pairs: topo(lo.E, L.E, c.ffSame, c.ffOther), contacts: 10, pRelease: 0.3, tauRec: 100, scale: c.ffScale });
      L.FFI = net.connect({ from: `E${k - 1}`, to: `I${k}`, p: c.ffiP, contacts: 10, pRelease: 0.3, scale: c.ffiScale });
      if (c.deep) {
        L.FBA = net.connect({ from: `D${k}`, to: `E${k - 1}`, p: c.fbP, contacts: 10, pRelease: 0.3, tauRec: 100, target: 'apical', plastic: fbPlastic });
        L.FBD = net.connect({ from: `D${k}`, to: `D${k - 1}`, p: c.fbP, contacts: 10, pRelease: 0.3, tauRec: 100, plastic: fbPlastic });
      } else L.FB = net.connect({ from: `E${k}`, to: `E${k - 1}`, p: c.fbP, contacts: 10, pRelease: 0.3, tauRec: 100, target: c.apical ? 'apical' : 'soma', plastic: fbPlastic });
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
/** 可学习的状态：每层的兴奋→兴奋效能（浅层 EE、深层 DEE）、抑制强度（IE、DIE）、反馈效能（FB；深层结构下 FBA、FBD） */
const RHO_KEYS = ['EE', 'FB', 'DEE', 'FBA', 'FBD'], SCALE_KEYS = ['IE', 'DIE'];
export const saveState = (stack) => stack.layers.map((L) => Object.fromEntries([...RHO_KEYS.filter((key) => L[key]).map((key) => [key, [...L[key].rho]]), ...SCALE_KEYS.filter((key) => L[key]).map((key) => [key, [...L[key].scale]])]));
export function loadState(stack, state) {
  state.forEach((s, k) => {
    const L = stack.layers[k];
    if (!L) return;
    for (const key of RHO_KEYS) if (s[key] && L[key]) setRho(L[key], s[key]);
    for (const key of SCALE_KEYS) if (s[key] && L[key]) L[key].scale.set(s[key]);
  });
}

/** 外部输入：第一层浅层兴奋性神经元按 drives（层内编号 → 每条输入的 Hz），其余兴奋性神经元（含深层）背景，抑制性神经元没有外部输入；
 * extra：直接给任意神经元（全局编号 → Hz），例如不经第一层、直接驱动上层（"想象"测试） */
export function inputRate(stack, drives = new Map(), extra = null) {
  const arr = new Float64Array(stack.net.N);
  stack.layers.forEach((L, k) => { L.E.forEach((i, a) => (arr[i] = k === 0 ? drives.get(a) ?? stack.cfg.bg : stack.cfg.bg)); if (L.D) for (const i of L.D) arr[i] = stack.cfg.bg; });
  if (extra) for (const [i, hz] of extra) arr[i] = hz;
  // θ 节律：深层的抑制性神经元收到按 f Hz 起伏的外部输入，每个周期把深层整体抑制一次（θ 的起搏来自内侧隔区等处的节律性抑制）
  const th = stack.cfg.theta;
  if (!th) return (i) => arr[i];
  const isDI = new Uint8Array(stack.net.N);
  for (const L of stack.layers) if (L.DI) for (const i of L.DI) isDI[i] = 1;
  return (i, t) => (isDI[i] ? (th.rate * (1 + Math.cos((2 * Math.PI * th.f * t) / 1000))) / 2 : arr[i]);
}

/** [t0, t1) 内各神经元的放电时刻（放电记录按时间追加，从尾部往前找） */
export function spikesIn(net, ids, t0, t1) {
  return ids.map((i) => { const s = net.spikes[i], out = []; for (let q = s.length - 1; q >= 0 && s[q] >= t0; q--) if (s[q] < t1) out.push(s[q]); return out.reverse(); });
}
