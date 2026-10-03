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
// copyP：拷贝突触的释放概率（越高短时程压抑越强：持续的强输入被压下去，短暂的弱输入几乎不受影响）；
// fbAccel / fbBudgetMode：层间反馈的学习速度与异突触预算的形式（'sub' 等量下调、'mult' 按比例缩小——减法式会让最后学的物体把别的反馈清零）
export const DEEP_DEFAULTS = { deepP: { C: 400, b: 5, tauW: 300 }, copySame: 0.5, copyScale: 20, copyP: 0.3, copyKind: 'nmda', localSame: 0.3, localScale: 3, dIE: 3, deepBudget: 0.3, fbAccel: 5, fbBudgetMode: 'sub' };

/** 抑制性神经元分三类（inhTypes = true 时，替代每层浅层那一群 I；细胞性质与连接概率取自 Pfeffer 等 2013、Jiang 等 2015 的皮层测量）：
 * PV（快放电篮状细胞）：膜快、不适应；接收兴奋（压抑型）与下层的前馈抑制，以快速 GABA_A 打在锥体细胞胞体周围，彼此相互抑制——
 *   γ 节律、整体增益、按抑制性可塑性维持兴奋/抑制平衡（L.IE 指向 PV→锥体细胞，L.I 指向 PV，旧代码照用）；
 * SOM（Martinotti 细胞）：会适应、有一点自发放电；接收的兴奋是易化型的（持续高频才起来），打在锥体细胞的顶树突上，也抑制 PV 与 VIP；
 * VIP：接收自上而下的输入（上一层深层）和少量本层兴奋，几乎只抑制 SOM——去抑制：上层来的时候把本层顶树突的门打开。 */
export const INH_DEFAULTS = {
  nPV: 15, nSOM: 10, nVIP: 5,
  pv: { C: 60, gL: 10, tRef: 1 }, som: { C: 150, gL: 8, tRef: 2, b: 15, tauW: 200 }, vip: { C: 100, gL: 8, tRef: 2, b: 5, tauW: 200 },
  // 标定（未训练的单层，驱动一个组的 16 个成员 200 ms）：被驱动的组 16 Hz（单一抑制时 23 Hz）；SOM 静息 6 Hz、持续驱动时易化到 15 Hz；
  // 顶树突：有 SOM 时被驱动组的顶树突比静息高 1.0 mV，去掉 SOM→顶树突 2.7 mV，去掉 VIP→SOM（SOM 升到 20–25 Hz）0 mV
  somBg: 12, vipBg: 9,
  // vipMod：学习态（net.mod = 1）时 VIP 额外的背景输入——乙酰胆碱经烟碱受体直接兴奋 VIP（Fu 等 2014；Letzkus 等 2011）
  vipMod: 0,
  sEPV: 2, sPVE: 5, sPVPV: 5, sESOM: 4, sSOME: 15, sSOMPV: 3, sSOMVIP: 3, sVIPSOM: 8, sEVIP: 1, sFBV: 3, sFFI: 1.5,
  pEPV: 0.5, pPVE: 0.6, pPVPV: 0.5, pESOM: 0.3, pSOME: 0.5, pSOMPV: 0.3, pSOMVIP: 0.3, pVIPSOM: 0.6, pEVIP: 0.1, pFBV: 0.3,
};

/** cfg：单层参数（与 demo/fig2-network.mjs 的 CFG 相同：NE、NI、K 组 × M 个神经元、pConn、contacts、sMax、rho0、budget、sie、istdpEta、rule、bg、seed）+ 层间参数；
 * deep：每层再加一群深层神经元（见上），层间反馈改由深层发出：落到下一层浅层的顶树突（FBA）与下一层深层（FBD），都按钙规则学习；
 * inhTypes：浅层的抑制分 PV / SOM / VIP 三类（见 INH_DEFAULTS，cfg.inh 覆盖其中的参数），上一层（深层）另投射到本层的 VIP */
export function buildLayered(nLayers, cfg) {
  const c = { ...LAYER_DEFAULTS, ...DEEP_DEFAULTS, ...cfg };
  const ih = (c.inhP = { ...INH_DEFAULTS, ...(c.inh ?? {}) });
  const hasApical = !!(c.apical || c.deep);
  const pops = [];
  for (let k = 1; k <= nLayers; k++) {
    pops.push({ name: `E${k}`, n: c.NE, p: { b: 20, tauW: 300 }, apical: hasApical ? c.apicalP ?? true : undefined });
    if (c.inhTypes) pops.push({ name: `PV${k}`, n: ih.nPV, p: ih.pv }, { name: `SOM${k}`, n: ih.nSOM, p: ih.som }, { name: `VIP${k}`, n: ih.nVIP, p: ih.vip });
    else pops.push({ name: `I${k}`, n: c.NI, p: c.iP ?? { C: 100, tRef: 1 } });
    if (c.deep) pops.push({ name: `D${k}`, n: c.NE, p: c.deepP, adp: c.deepAdp }, { name: `DI${k}`, n: c.NI, p: { C: 100, tRef: 1 } });
  }
  const net = new SpikingNetwork({ costs: mammalianCosts(), seed: c.seed, plasticity: c.rule, pops });
  const block = (a) => Math.min(Math.floor(a / c.M), c.K);
  const topo = (pre, post, pSame, pOther = 0) => { const pairs = []; for (let a = 0; a < pre.length; a++) for (let b = 0; b < post.length; b++) if (net.rng.uniform() < (block(a) === block(b) ? pSame : pOther)) pairs.push([pre[a], post[b]]); return pairs; };
  // fbRule = 'plateau'：反馈（落在顶树突）按平台 × 资格迹学习（见 spiking-network.mjs），fbPlateau 覆盖 tauE / etaP / thetaPl
  const fbPlastic = { sMin: 0, sMax: c.fbMax, rho0: c.fbRho0, accel: c.fbAccel, budget: c.fbBudget, budgetMode: c.fbBudgetMode, ...(c.fbRule ? { rule: c.fbRule, ...(c.fbPlateau ?? {}) } : {}) };
  const layers = [];
  for (let k = 1; k <= nLayers; k++) {
    const L = { k, E: net.idx(`E${k}`) };
    L.EE = net.connect({ from: `E${k}`, to: `E${k}`, p: c.pConn, contacts: c.contacts, pRelease: 0.3, tauRec: 100, plastic: { sMin: 0.1, sMax: c.sMax, rho0: c.rho0, accel: 5, budget: k > 1 ? c.upperBudget ?? c.budget : c.budget } });
    if (c.inhTypes) {
      const [E, PV, SOM, VIP] = ['E', 'PV', 'SOM', 'VIP'].map((n) => `${n}${k}`), inh = { contacts: 10, pRelease: 0.5, tauRec: 300 };
      L.PV = net.idx(PV); L.SOM = net.idx(SOM); L.VIP = net.idx(VIP); L.I = L.PV;
      L.EI = net.connect({ from: E, to: PV, p: ih.pEPV, contacts: 10, pRelease: 0.5, tauRec: 100, scale: ih.sEPV });
      L.IE = net.connect({ from: PV, to: E, p: ih.pPVE, ...inh, kind: 'inhFast', scale: ih.sPVE, istdp: { eta: c.istdpEta, target: c.eTarget ?? 5, wMax: c.ieMax ?? 60 } });
      L.PVPV = net.connect({ from: PV, to: PV, p: ih.pPVPV, ...inh, kind: 'inhFast', scale: ih.sPVPV });
      L.ESOM = net.connect({ from: E, to: SOM, p: ih.pESOM, contacts: 10, tauRec: 50, U: 0.1, tauF: 500, scale: ih.sESOM });
      L.SOME = net.connect({ from: SOM, to: E, p: ih.pSOME, ...inh, kind: 'inh', target: hasApical ? 'apical' : 'soma', scale: ih.sSOME });
      L.SOMPV = net.connect({ from: SOM, to: PV, p: ih.pSOMPV, ...inh, kind: 'inh', scale: ih.sSOMPV });
      L.SOMVIP = net.connect({ from: SOM, to: VIP, p: ih.pSOMVIP, ...inh, kind: 'inh', scale: ih.sSOMVIP });
      L.VIPSOM = net.connect({ from: VIP, to: SOM, p: ih.pVIPSOM, ...inh, kind: 'inh', scale: ih.sVIPSOM });
      L.EVIP = net.connect({ from: E, to: VIP, p: ih.pEVIP, contacts: 10, pRelease: 0.3, scale: ih.sEVIP });
    } else {
      // 单一抑制群；平衡态的配置（见 demo/balanced-layer.mjs）会改连接概率、强度、受体（快速 GABA_A）、抑制之间的互相抑制、稳态目标频率
      L.I = net.idx(`I${k}`);
      L.EI = net.connect({ from: `E${k}`, to: `I${k}`, p: c.pEI ?? 0.5, contacts: 10, pRelease: 0.3, scale: c.sEI ?? 1.5 });
      L.IE = net.connect({ from: `I${k}`, to: `E${k}`, p: c.pIE ?? 0.5, contacts: 10, pRelease: 0.5, tauRec: 300, kind: c.ieKind ?? 'inh', scale: c.sie, istdp: { eta: c.istdpEta, target: c.eTarget ?? 5, wMax: c.ieMax ?? 20 } });
      if (c.pII) L.II = net.connect({ from: `I${k}`, to: `I${k}`, p: c.pII, contacts: 10, pRelease: 0.5, tauRec: 300, kind: c.ieKind ?? 'inh', scale: c.sII ?? 5 });
    }
    if (c.deep) {
      L.D = net.idx(`D${k}`); L.DI = net.idx(`DI${k}`);
      L.DEE = net.connect({ from: `D${k}`, to: `D${k}`, p: c.pConn, contacts: c.contacts, pRelease: 0.3, tauRec: 100, plastic: { sMin: 0.1, sMax: c.sMax, rho0: c.rho0, accel: 5, budget: c.deepBudget } });
      L.DEI = net.connect({ from: `D${k}`, to: `DI${k}`, p: 0.5, contacts: 10, pRelease: 0.3, scale: 1.5 });
      L.DIE = net.connect({ from: `DI${k}`, to: `D${k}`, p: 0.5, contacts: 10, pRelease: 0.5, tauRec: 300, kind: 'inh', scale: c.dIE, istdp: { eta: c.istdpEta, target: c.dieTarget ?? 5 } });
      L.COPY = net.connect({ pairs: topo(L.E, L.D, c.copySame), contacts: 10, pRelease: c.copyP, tauRec: 100, scale: c.copyScale, kind: c.copyKind });
      L.LOCAL = net.connect({ pairs: topo(L.D, L.E, c.localSame), contacts: 10, pRelease: 0.3, tauRec: 100, scale: c.localScale, target: 'apical' });
    }
    if (k > 1) {
      const lo = layers[k - 2];
      L.FF = net.connect({ pairs: topo(lo.E, L.E, c.ffSame, c.ffOther), contacts: 10, pRelease: 0.3, tauRec: 100, scale: c.ffScale });
      L.FFI = net.connect({ from: `E${k - 1}`, to: c.inhTypes ? `PV${k}` : `I${k}`, p: c.ffiP, contacts: 10, pRelease: 0.3, scale: c.inhTypes ? ih.sFFI : c.ffiScale });
      if (c.deep) {
        L.FBA = net.connect({ from: `D${k}`, to: `E${k - 1}`, p: c.fbP, contacts: 10, pRelease: 0.3, tauRec: 100, target: 'apical', plastic: fbPlastic });
        L.FBD = net.connect({ from: `D${k}`, to: `D${k - 1}`, p: c.fbP, contacts: 10, pRelease: 0.3, tauRec: 100, plastic: fbPlastic });
      } else L.FB = net.connect({ from: `E${k}`, to: `E${k - 1}`, p: c.fbP, contacts: 10, pRelease: 0.3, tauRec: 100, target: c.apical ? 'apical' : 'soma', plastic: fbPlastic, modSup: c.fbModSup ?? 0 });
      // 自上而下打到下一层的 VIP（去抑制：上层一活跃，下一层 SOM 被压、顶树突的门打开）
      if (c.inhTypes) L.FBV = net.connect({ from: c.deep ? `D${k}` : `E${k}`, to: `VIP${k - 1}`, p: ih.pFBV, contacts: 10, pRelease: 0.3, tauRec: 100, scale: ih.sFBV });
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

/** 外部输入：第一层浅层兴奋性神经元按 drives（层内编号 → 每条输入的 Hz），其余兴奋性神经元（含深层）背景；
 * 抑制性神经元里只有 SOM、VIP 有背景（inhTypes 时），PV 与单一抑制群没有外部输入；
 * extra：直接给任意神经元（全局编号 → Hz），例如不经第一层、直接驱动上层（"想象"测试） */
export function inputRate(stack, drives = new Map(), extra = null) {
  const arr = new Float64Array(stack.net.N);
  stack.layers.forEach((L, k) => {
    // bgUpper：上层兴奋性神经元的背景（平衡态里上层还收下层自发活动经前馈送来的输入，背景要相应低一些）
    L.E.forEach((i, a) => (arr[i] = k === 0 ? drives.get(a) ?? stack.cfg.bg : stack.cfg.bgUpper ?? stack.cfg.bg));
    if (L.D) for (const i of L.D) arr[i] = stack.cfg.bg;
    // SOM、VIP 有一点自发放电（体内 SOM 细胞几赫兹的持续活动）；PV 只靠本地兴奋驱动
    if (L.SOM) for (const i of L.SOM) arr[i] = stack.cfg.inhP.somBg;
    if (L.VIP) for (const i of L.VIP) arr[i] = stack.cfg.inhP.vipBg + stack.net.mod * stack.cfg.inhP.vipMod;
    // 平衡态：抑制性神经元也收背景输入（真实皮层里中间神经元同样接收丘脑与远程兴奋）
    if (stack.cfg.bgI) for (const i of L.I) arr[i] = stack.cfg.bgI;
  });
  if (extra) for (const [i, hz] of extra) arr[i] = hz;
  // θ 节律：抑制性神经元收到按 f Hz 起伏的外部输入，每个周期把所在的群整体抑制一次（θ 的起搏来自内侧隔区等处的节律性抑制）；
  // target：'deep'（默认，深层抑制 DI）、'surf'（浅层抑制 I）、'both'
  const th = stack.cfg.theta;
  if (!th) return (i) => arr[i];
  const target = th.target ?? 'deep', isTh = new Uint8Array(stack.net.N);
  for (const L of stack.layers) {
    if (target !== 'surf' && L.DI) for (const i of L.DI) isTh[i] = 1;
    if (target !== 'deep') for (const i of L.I) isTh[i] = 1;
  }
  return (i, t) => (isTh[i] ? (th.rate * (1 + Math.cos((2 * Math.PI * th.f * t) / 1000))) / 2 : arr[i]);
}

/** [t0, t1) 内各神经元的放电时刻（放电记录按时间追加，从尾部往前找） */
export function spikesIn(net, ids, t0, t1) {
  return ids.map((i) => { const s = net.spikes[i], out = []; for (let q = s.length - 1; q >= 0 && s[q] >= t0; q--) if (s[q] < t1) out.push(s[q]); return out.reverse(); });
}
