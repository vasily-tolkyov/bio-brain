/** 多层脉冲网络：每一层都是第 4 层（脉冲版）那样的完整网络——
 *   兴奋性 + 抑制性 LIF 神经元；层内兴奋→兴奋突触按钙规则学习（+ 异突触竞争），抑制→兴奋按抑制性可塑性学习；
 *   每个兴奋性神经元有背景泊松输入。层内的吸引子与有向通道都是学出来的。
 * 层与层之间（都是真实突触）：
 *   前馈 ↑：第 k 层兴奋 → 第 k+1 层兴奋，按组对应的拓扑投射（同组概率 ffSame，异组 ffOther），强度固定；
 *          另有第 k 层兴奋 → 第 k+1 层抑制的前馈抑制；
 *   反馈 ↓：第 k+1 层兴奋 → 第 k 层兴奋，随机连接，起点很弱，按钙规则学习（+ 异突触竞争）。
 * apical：兴奋性神经元带顶树突分区（见 spiking-network.mjs 的 APICAL_DEFAULT），反馈全部落在顶树突上，
 *   前馈与层内连接落在胞体——反馈只做阈下调制，与胞体输入同时到来时触发树突平台、把放电变成爆发；
 * 上层的异突触预算可以单独设（upperBudget）：上层的吸引子由下层几个吸引子合成，更大，组内连接会占掉更多预算；
 *   真实皮层沿层级往上，锥体神经元的树突棘也越多（Elston 等人的测量）。
 * 平衡态（见 demo/balanced-layer.mjs）：抑制群的连接概率、强度、受体（快速 GABA_A）、抑制之间的互相抑制、稳态目标频率可设
 *   （pEI、sEI、pIE、sie、ieKind、pII、sII、eTarget、ieMax、iP），抑制性神经元也收背景（bgI），上层兴奋性神经元的背景可单独设（bgUpper）。
 * pc（预测编码）：每层另有误差群 N 与预测中继群 S（SOM 式抑制性中间神经元），上层不再直接反馈给下层表征：
 *   证据：本层表征 E → 本层误差 N（同组，固定，兴奋）；预测：本层中继 S → 本层误差 N（抑制，按抑制性可塑性学习）；
 *   预测下行：上层表征 E → 下层中继 S（同组，固定）；误差上行：下层误差 N → 上层表征 E（同组，固定）；前馈照旧。
 *   误差细胞有自发基线：证据多于预测时高于基线（意外），预测多于证据时低于基线（该来的没来）。
 *   抑制性可塑性（Vogels 等 2011）让中继 → 误差的抑制与可预期的兴奋同步起伏、把它抵消掉，剩下的就是预测误差——
 *   这正是"把规则和现实对齐"：预测对了误差回到基线、学习停止；预测错了，误差的方向指出错在哪。
 * 连接按层依次生成：K 层网络的前 k 层与 k 层网络的连接完全相同，所以可以逐层训练、逐层加层。 */
import { SpikingNetwork } from './spiking-network.mjs';
import { mammalianCosts } from '../L1-neuron/hh.mjs';

export const LAYER_DEFAULTS = { ffSame: 0.5, ffOther: 0.02, ffScale: 6, ffiP: 0.3, ffiScale: 1.5, fbP: 0.3, fbMax: 6, fbRho0: 0.05, fbBudget: 0.15 };
/** 预测编码各群的规模（组按与表征群同样的比例划分：K 组 + 不成组）、细胞参数、背景、连接。
 * 背景与预测下行强度按平衡态的表征层（demo/balanced-layer.mjs）标定：误差细胞自发约 4.5 Hz（抑制性可塑性再把它拉向目标 3 Hz），
 * 中继自发约 8 Hz、上层对应组活跃时 25–35 Hz；预测下行若与证据一样强，上层自发的 2.6 Hz 就把中继推到 20 Hz 以上、误差全被压住。
 * 中继 → 误差的突触压抑较弱（释放概率 0.3、恢复 100 ms）：SOM 细胞的输出突触不像 PV 那样强烈压抑，持续的预测才不会自己衰减掉。
 * tdKind：预测下行的受体。'nmda'（只有 NMDA，自上而下的作用主要靠 NMDA，Self 等 2012）让中继比上层表征慢：
 *   上层从 X 换到 Y 后，X 的中继还要再放 20–30 ms，正好盖住 Y 的误差出现的时刻；'exc'（AMPA + NMDA）时中继 10 ms 内就跟着熄灭。
 * predTauPre：中继 → 误差的抑制性可塑性在增强一侧用的突触前痕迹（ms）。默认的 20 ms 配对窗口只看"同时"，
 *   学到的是"X 在场时 Y 不在"（150 ms）压倒"X 刚过 Y 来了"（约 15 ms），序列越练"X 预测 Y"越弱；200 ms 的痕迹把后者记进来。
 *   兴奋性突触上的资格痕迹有直接证据（几秒，He 等 2015），抑制性突触上还没有直接测到，这里是假设 */
export const PC_DEFAULTS = {
  NN: 200, NS: 100, nP: { b: 5, tauW: 200 }, sP: { C: 100, tRef: 2, b: 5, tauW: 200 }, bgN: 14, bgS: 9,
  pEvid: 0.2, sEvid: 3, pPred: 0.5, sPred: 0.5, predU: 0.3, predTau: 100, predEta: 0.02, predTarget: 3, predMax: 30, predTauPre: 200, pTD: 0.3, sTD: 2, tdKind: 'nmda', pUp: 0.2, sUp: 3,
};

/** cfg：单层参数（与 demo/fig2-network.mjs 的 CFG 相同：NE、NI、K 组 × M 个神经元、pConn、contacts、sMax、rho0、budget、sie、istdpEta、rule、bg、seed）+ 层间参数 */
export function buildLayered(nLayers, cfg) {
  const c = { ...LAYER_DEFAULTS, ...cfg };
  const pc = c.pc ? (c.pcP = { ...PC_DEFAULTS, ...(c.pcOver ?? {}) }) : null;
  const pops = [];
  for (let k = 1; k <= nLayers; k++) {
    pops.push({ name: `E${k}`, n: c.NE, p: { b: 20, tauW: 300 }, apical: c.apical || undefined }, { name: `I${k}`, n: c.NI, p: c.iP ?? { C: 100, tRef: 1 } });
    if (pc) pops.push({ name: `N${k}`, n: pc.NN, p: pc.nP }, { name: `S${k}`, n: pc.NS, p: pc.sP });
  }
  const net = new SpikingNetwork({ costs: mammalianCosts(), seed: c.seed, plasticity: c.rule, pops });
  // 组：每个群按自己的规模等比例划分成 K 组（组大小 = 群大小 × M / NE），余下的不成组（编号 K）
  const blockIn = (n) => (a) => Math.min(Math.floor(a / ((n * c.M) / c.NE)), c.K);
  const topoN = (pre, post, pSame, pOther = 0) => { const bp = blockIn(pre.length), bq = blockIn(post.length), pairs = []; for (let a = 0; a < pre.length; a++) for (let b = 0; b < post.length; b++) if (net.rng.uniform() < (bp(a) === bq(b) ? pSame : pOther)) pairs.push([pre[a], post[b]]); return pairs; };
  const topo = (pre, post, pSame, pOther = 0) => topoN(pre, post, pSame, pOther);
  const fbPlastic = { sMin: 0, sMax: c.fbMax, rho0: c.fbRho0, accel: 5, budget: c.fbBudget };
  const layers = [];
  for (let k = 1; k <= nLayers; k++) {
    const L = { k, E: net.idx(`E${k}`), I: net.idx(`I${k}`) };
    L.EE = net.connect({ from: `E${k}`, to: `E${k}`, p: c.pConn, contacts: c.contacts, pRelease: 0.3, tauRec: 100, plastic: { sMin: 0.1, sMax: c.sMax, rho0: c.rho0, accel: 5, budget: k > 1 ? c.upperBudget ?? c.budget : c.budget } });
    L.EI = net.connect({ from: `E${k}`, to: `I${k}`, p: c.pEI ?? 0.5, contacts: 10, pRelease: 0.3, scale: c.sEI ?? 1.5 });
    L.IE = net.connect({ from: `I${k}`, to: `E${k}`, p: c.pIE ?? 0.5, contacts: 10, pRelease: 0.5, tauRec: 300, kind: c.ieKind ?? 'inh', scale: c.sie, istdp: { eta: c.istdpEta, target: c.eTarget ?? 5, wMax: c.ieMax ?? 20 } });
    if (c.pII) L.II = net.connect({ from: `I${k}`, to: `I${k}`, p: c.pII, contacts: 10, pRelease: 0.5, tauRec: 300, kind: c.ieKind ?? 'inh', scale: c.sII ?? 5 });
    if (pc) {
      L.N = net.idx(`N${k}`); L.S = net.idx(`S${k}`);
      L.EVID = net.connect({ pairs: topoN(L.E, L.N, pc.pEvid), contacts: 10, pRelease: 0.3, tauRec: 100, scale: pc.sEvid });
      L.PRED = net.connect({ from: `S${k}`, to: `N${k}`, p: pc.pPred, contacts: 10, pRelease: pc.predU, tauRec: pc.predTau, kind: 'inh', scale: pc.sPred, istdp: { eta: pc.predEta, target: pc.predTarget, wMax: pc.predMax, ...(pc.predTauPre ? { tauPre: pc.predTauPre } : {}) } });
    }
    if (k > 1) {
      const lo = layers[k - 2];
      L.FF = net.connect({ pairs: topo(lo.E, L.E, c.ffSame, c.ffOther), contacts: 10, pRelease: 0.3, tauRec: 100, scale: c.ffScale });
      L.FFI = net.connect({ from: `E${k - 1}`, to: `I${k}`, p: c.ffiP, contacts: 10, pRelease: 0.3, scale: c.ffiScale });
      if (pc) {
        L.UP = net.connect({ pairs: topoN(lo.N, L.E, pc.pUp), contacts: 10, pRelease: 0.3, tauRec: 100, scale: pc.sUp });
        L.TD = net.connect({ pairs: topoN(L.E, lo.S, pc.pTD), contacts: 10, pRelease: 0.3, tauRec: 100, kind: pc.tdKind, scale: pc.sTD });
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
/** 可学习的状态：每层的兴奋→兴奋效能（EE）、抑制强度（IE）、反馈效能（FB）、预测抑制强度（PRED） */
const RHO_KEYS = ['EE', 'FB'], SCALE_KEYS = ['IE', 'PRED'];
export const saveState = (stack) => stack.layers.map((L) => Object.fromEntries([...RHO_KEYS.filter((key) => L[key]).map((key) => [key, [...L[key].rho]]), ...SCALE_KEYS.filter((key) => L[key]).map((key) => [key, [...L[key].scale]])]));
export function loadState(stack, state) {
  state.forEach((s, k) => {
    const L = stack.layers[k];
    if (!L) return;
    for (const key of RHO_KEYS) if (s[key] && L[key]) setRho(L[key], s[key]);
    for (const key of SCALE_KEYS) if (s[key] && L[key]) L[key].scale.set(s[key]);
  });
}

/** 外部输入：第一层兴奋性神经元按 drives（层内编号 → 每条输入的 Hz），其余兴奋性神经元背景（上层可用 bgUpper）；
 * 抑制性神经元只在平衡态配置下收背景（bgI）；误差群、中继群收各自的背景（预测编码配置）；
 * extra：直接给任意神经元（全局编号 → Hz），例如不经第一层、直接驱动上层（"想象"测试） */
export function inputRate(stack, drives = new Map(), extra = null) {
  const arr = new Float64Array(stack.net.N), pc = stack.cfg.pcP;
  stack.layers.forEach((L, k) => {
    L.E.forEach((i, a) => (arr[i] = k === 0 ? drives.get(a) ?? stack.cfg.bg : stack.cfg.bgUpper ?? stack.cfg.bg));
    if (stack.cfg.bgI) for (const i of L.I) arr[i] = stack.cfg.bgI;
    if (pc) { for (const i of L.N) arr[i] = pc.bgN; for (const i of L.S) arr[i] = pc.bgS; }
  });
  if (extra) for (const [i, hz] of extra) arr[i] = hz;
  return (i) => arr[i];
}

/** 某个群（全局编号表 ids）里第 g 组的成员；g = K 为不成组的部分 */
export const groupIds = (stack, ids, g) => ids.filter((_, a) => Math.min(Math.floor(a / ((ids.length * stack.cfg.M) / stack.cfg.NE)), stack.cfg.K) === g);

/** [t0, t1) 内各神经元的放电时刻（放电记录按时间追加，从尾部往前找） */
export function spikesIn(net, ids, t0, t1) {
  return ids.map((i) => { const s = net.spikes[i], out = []; for (let q = s.length - 1; q >= 0 && s[q] >= t0; q--) if (s[q] < t1) out.push(s[q]); return out.reverse(); });
}
