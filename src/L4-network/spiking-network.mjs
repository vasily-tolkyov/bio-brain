/** 第 4 层（脉冲版）：由真实点神经元与突触组成的网络，方程与第 1–3 层逐项相同，只是按突触后神经元聚合以便跑上百个神经元：
 * - 神经元：第 1 层的 LIF + 适应（C dV/dt = −g_L(V−E_L) − w + I，τ_w dw/dt = a(V−E_L) − w，放电后 V←V_reset、w += b、不应期）
 * - 突触：第 2 层的双指数 AMPA / NMDA（镁阻断 B(V)）/ GABA_A；同一突触后神经元的同类受体电导是线性叠加的，所以按神经元存一份状态
 *   ——与逐个突触各存一份完全等价（verify 里逐拍对拍）；短时程压抑按突触前神经元（同一组连接参数相同）
 * - 可塑性：第 3 层的钙规则。突触 j→i 的钙 = Cpre·x_j(延迟 D) + Cpost·y_i，x、y 是按神经元的指数迹（钙增量线性叠加、衰减常数相同），
 *   所以每个突触的钙不必单独积分，但每个突触有自己的效能 ρ 与强度
 * - 外部刺激：每个神经元收 nIn 条泊松输入（AMPA + NMDA），它们的放电与突触成本记到 'drive'
 * - ATP：静息、放电、突触前（每囊泡）、突触后（真实受体电流里的 Na⁺ / Ca²⁺ / Cl⁻）、外部驱动，与第 1、2 层同一套换算 */
import { RECEPTORS, PRE_ATP_PER_VESICLE, mgBlock } from '../L2-synapse/synapse.mjs';
import { GB_DP } from '../L3-plasticity/calcium-rule.mjs';
import { LIF_DEFAULT } from '../L1-neuron/lif.mjs';
import { ATP_PER_NA, ATP_PER_CA, ATP_PER_CL, ionsFromFemtoCoulomb } from '../core/units.mjs';
import { EnergyLedger } from '../core/ledger.mjs';
import { makeRng } from '../core/rng.mjs';

const norm = ({ tauR, tauD }) => { const tp = (tauR * tauD) / (tauD - tauR) * Math.log(tauD / tauR); return 1 / (Math.exp(-tp / tauD) - Math.exp(-tp / tauR)); };
const REC = ['AMPA', 'NMDA', 'GABA_A', 'GABA_A_fast'].map((n) => ({ n, ...RECEPTORS[n], k: norm(RECEPTORS[n]) }));
const NREC = REC.length;

/** 顶树突分区（锥体细胞的远端树突，Larkum 1999；简化形式取自 Naud & Sprekeler 2018, PNAS）：
 * Cd dVd/dt = −gLd(Vd−E_L) + gd·[f(Vd)−f(E_L)]⁺ − wd + I_syn,d + I_bAP；τwd dwd/dt = ad(Vd−E_L) − wd；
 * f(V) = 1/(1+e^{−(V−Ed)/Dd}) 是钙峰非线性：树突电位越过约 −35 mV 就自我放大成平台，wd 让平台在几十毫秒后终止；
 * 胞体放电后 bapMs 内向树突注入 cd（反传动作电位，约抬高 15 mV）；胞体只经由平台收到 gs·[f(Vd)−f(E_L)]⁺，
 * 所以树突没有输入时神经元与点神经元完全相同。单位：pF、nS、mV、ms、pA。 */
export const APICAL_DEFAULT = { Cd: 170, gLd: 24, Ed: -38, Dd: 6, gd: 1200, gs: 700, ad: 20, tauWd: 30, cd: 2600, bapMs: 1 };
const fCa = (V, a) => 1 / (1 + Math.exp(-(V - a.Ed) / a.Dd));

function poisson(lambda, rng) {
  if (lambda <= 0) return 0;
  const L = Math.exp(-lambda);
  let k = 0, p = 1;
  do { k++; p *= rng.uniform(); } while (p > L);
  return k - 1;
}

export class SpikingNetwork {
  /** pops: [{ name, n, p: LIF 参数, apical: 顶树突参数（true = APICAL_DEFAULT；不给 = 点神经元） }]；costs: 第 1 层标定的 { atpPerSpike, atpRestPerMs } */
  constructor({ pops, costs, seed = 1, dt = 0.1, plasticity = {} }) {
    this.dt = dt;
    this.rng = makeRng(seed);
    this.costs = costs;
    this.ledger = new EnergyLedger();
    this.gb = { ...GB_DP, ...plasticity };
    this.pops = {};
    let off = 0;
    for (const p of pops) { this.pops[p.name] = { start: off, n: p.n, p: { ...LIF_DEFAULT, ...(p.p ?? {}) } }; off += p.n; }
    const N = (this.N = off);
    this.par = Array.from({ length: N }, (_, i) => Object.values(this.pops).find((q) => i >= q.start && i < q.start + q.n).p);
    this.V = Float64Array.from(this.par, (q) => q.EL);
    this.w = new Float64Array(N);
    this.ref = new Float64Array(N).fill(-Infinity);
    // 受体状态：[受体][a/b][神经元]；外部输入单独一套（成本记到 drive）
    this.ra = REC.map(() => new Float64Array(N)); this.rb = REC.map(() => new Float64Array(N));
    this.ea = [new Float64Array(N), new Float64Array(N)]; this.eb = [new Float64Array(N), new Float64Array(N)];
    this.Rin = new Float64Array(N).fill(1);
    this.groups = [];
    this.out = Array.from({ length: N }, () => []); // 每个突触前神经元：[组号, 突触号]
    this.x = new Float64Array(N); this.y = new Float64Array(N);
    this.pending = []; this.pendHead = 0;
    this.plastic = []; // 可塑的组
    this.plateauGroups = []; // 按顶树突平台 × 资格迹学习的组（也在 plastic 里，参加异突触预算）
    // 神经调质（模拟乙酰胆碱）：1 = 学习态、0 = 回忆态；门控平台规则的可塑性，并按 modI 去极化顶树突
    this.mod = 1;
    this.t = 0;
    this.spikes = Array.from({ length: N }, () => []);
    this.idx = (name) => { const q = this.pops[name]; return Array.from({ length: q.n }, (_, k) => q.start + k); };
    // 后去极化（Lisman & Idiart 1995 的工作记忆模型）：放电后一股先升后降的去极化电流 amp·(e^{−t/decay} − e^{−t/rise})，
    // 让一个神经元在下一个 θ 周期的同一时段再次放电——被保持的几项在每个周期里按原来的先后重演
    if (pops.some((p) => p.adp)) {
      this.adpOf = new Array(N).fill(null);
      for (const p of pops) if (p.adp) { const q = this.pops[p.name]; for (let k = 0; k < q.n; k++) this.adpOf[q.start + k] = p.adp; }
      this.adpR = new Float64Array(N); this.adpD = new Float64Array(N);
    }
    if (pops.some((p) => p.apical)) {
      this.ap = new Array(N).fill(null);
      for (const p of pops) if (p.apical) {
        const a = { ...APICAL_DEFAULT, ...(p.apical === true ? {} : p.apical) }, q = this.pops[p.name];
        a.f0 = fCa(q.p.EL, a);
        for (let k = 0; k < q.n; k++) this.ap[q.start + k] = a;
      }
      this.Vd = Float64Array.from(this.par, (q) => q.EL); this.wd = new Float64Array(N); this.bapUntil = new Float64Array(N).fill(-Infinity);
      this.plat = new Float64Array(N); // 此刻的平台强度 [f(Vd) − f(E_L)]⁺，平台规则的指导信号
      this.dra = REC.map(() => new Float64Array(N)); this.drb = REC.map(() => new Float64Array(N));
    }
  }

  /** 随机连接一组突触。plastic: { sMin, sMax, rho0, accel, budget } 给定时这组突触按钙规则学习；
   *   budget 给定时加异突触竞争：每 10 ms，一个神经元的输入突触平均效能超过 budget 就把全部输入等量下调（总量有限，强了一部分、别的就得弱）；
   *   budgetMode: 'mult' 时改为按比例缩小（弱的输入不会先被压到零）；
   * kind：'exc'（AMPA + NMDA）、'nmda'（只有 NMDA：慢、电压依赖，可维持持续活动）、'ampa'（只有 AMPA：快、近似线性）、'inh'（GABA_A）、'inhFast'（快速 GABA_A，篮状细胞式）；
   * 返回的组上设 frozen = true 时这组突触暂停学习（其余组不受影响，逐拍结果与不设时相同）；
   * istdp: { eta, target, tau, wMax } 给定时（抑制性突触）按 Vogels 等 2011 的抑制性可塑性学习：
   *   突触前抑制性神经元放电时 w += η(y_post − α)，突触后兴奋性神经元放电时 w += η·x_pre，α = 2·目标频率·τ——
   *   突触后放电高于目标频率时抑制变强，低于时变弱，局部地维持兴奋/抑制平衡；
   * tauF 给定时是易化型突触（Tsodyks–Markram）：释放概率 u 每次放电先涨 U(1−u)、再按 u 释放，之间按 tauF 回落到 0——
   *   第一次放电只释放 U，持续高频时越放越多（锥体细胞 → SOM 细胞的突触就是这样，Silberberg & Markram 2007）；
   * plastic.rule = 'plateau'（只用于落在顶树突的组）：行为时间尺度的可塑性（Bittner 等 2017；新皮层 Gambino 等 2014）——
   *   突触前每放一次电留下资格迹 e += 1，按 tauE（毫秒）衰减；突触后顶树突处于平台（强度超过 thetaPl）时，
   *   dρ/dt = etaP · mod · (平台 − thetaPl) · e · (1 − ρ)。指导信号来自突触后细胞自己的平台，不要求顶树突输入驱动胞体；
   *   几百毫秒前放过电的突触前也能被写进去，所以按行为节奏（150 ms 一项）经历的先后可以学到；
   *   cov = true 时取协方差形式（BTSP 本身是双向的，Milstein 等 2021）：用资格迹减去它自己按 tauBar 的滑动平均 d = e − ē，
   *   d > 0 时按 (1 − ρ) 增强、d < 0 时按 ρ 削弱——自发活动带来的随机相遇平均下来互相抵消，只有与刺激锁定的活动被写进去；
   * plastic.rule = 'dendPred'（只用于落在顶树突的组）：树突预测规则（Urbanczik & Senn 2014），资格迹改成行为时间尺度——
   *   顶树突上的突触学着预测本细胞的胞体放电：突触后每放一次电，带资格迹的突触 ρ += etaD · mod · e · (1 − ρ)；
   *   每一时刻按树突电位所预示的放电率 φ(Vd) 持续削弱 ρ −= etaD · mod · e · φ(Vd) · dt · ρ；
   *   φ(Vd) = s0 + phiMax / (1 + e^{−(Vd − vh)/k})（Hz）：静息时约等于自发放电率（自发活动下不漂移），树突进入平台时对应爆发的放电率
   *   （平台引起的爆发不会自我强化）。预测对了学习就停；反馈抬高了树突、胞体却没放电，就削弱——自我限制；
   * modSup 给定时这组突触的传递在学习态按 (1 − mod · modSup) 减弱：乙酰胆碱在突触前抑制递归与反馈连接的释放（编码 / 提取切换，
   *   Hasselmo & Bower 1992；Hasselmo 2006）——学习时网络按自下而上的输入学，已学到的反馈不会自己引出平台、把自己越写越强 */
  connect({ from, to, p = 1, contacts = 5, pRelease = 0.3, tauRec = 100, scale = 1, kind = 'exc', plastic = null, istdp = null, pairs = null, autapses = false, target = 'soma', U = null, tauF = null, modSup = 0 }) {
    const pre = [], post = [];
    if (pairs) for (const [a, b] of pairs) { pre.push(a); post.push(b); }
    else for (const a of this.idx(from)) for (const b of this.idx(to)) if ((autapses || a !== b) && this.rng.uniform() < p) { pre.push(a); post.push(b); }
    const n = pre.length;
    if (target === 'apical' && post.some((b) => !this.ap?.[b])) throw new Error('落在顶树突的突触，突触后神经元必须有顶树突分区');
    const g = { kind, target, contacts, pRelease, tauRec, pre: Int32Array.from(pre), post: Int32Array.from(post), R: new Float64Array(this.N).fill(1), plastic,
      rho: plastic ? new Float64Array(n).fill(plastic.rho0 ?? 0.5) : null, scale: new Float64Array(n).fill(scale) };
    if (tauF) { g.tauF = tauF; g.U = U ?? 0.1; g.u = new Float64Array(this.N); }
    if (modSup) g.modSup = modSup;
    if (plastic?.rule === 'plateau') {
      if (target !== 'apical') throw new Error('平台规则只用于落在顶树突的突触');
      g.plateau = true; g.elig = new Float64Array(this.N); // 资格迹按突触前神经元存（同一突触前的各个突触相同）
      g.plastic = { tauE: 1500, etaP: 0.002, thetaPl: 0.3, cov: false, tauBar: 5000, ...plastic };
      if (g.plastic.cov) g.eBar = new Float64Array(this.N);
      this.plateauGroups.push(g);
    }
    if (plastic?.rule === 'dendPred') {
      if (target !== 'apical') throw new Error('树突预测规则只用于落在顶树突的突触');
      g.plateau = true; g.dendPred = true; g.elig = new Float64Array(this.N); // 不走钙规则（plateau 标记让钙规则的循环跳过它）
      g.plastic = { tauE: 300, etaD: 0.01, s0: 2, phiMax: 150, vh: -40, k: 3, ...plastic };
      this.dendPredGroups = [...(this.dendPredGroups ?? []), g];
    }
    if (plastic) {
      for (let k = 0; k < n; k++) g.scale[k] = plastic.sMin + (plastic.sMax - plastic.sMin) * g.rho[k];
      g.z = new Float64Array(n); // 每个突触自己的"同时检测"钙
      if (this.gb.vdep) g.cp = new Float64Array(n); // 电压依赖时每个突触自己的突触前钙（到达时乘突触后的镁解阻断）
      g.incP = Array.from({ length: this.N }, () => []);
      for (let k = 0; k < n; k++) g.incP[post[k]].push(k);
      this.nmda ??= new Float64Array(this.N);
    }
    if (istdp) {
      g.istdp = { eta: 0.02, target: 5, tau: 20, wMax: 20, ...istdp };
      g.inc = Array.from({ length: this.N }, () => []);
      for (let k = 0; k < n; k++) g.inc[post[k]].push(k);
      this.istdpGroups = [...(this.istdpGroups ?? []), g];
      this.xs ??= new Float64Array(this.N); this.ys ??= new Float64Array(this.N);
    }
    const gi = this.groups.push(g) - 1;
    for (let k = 0; k < n; k++) this.out[pre[k]].push(gi, k);
    if (plastic) this.plastic.push(g);
    return g;
  }

  /** 运行 ms。rate(i, t)：神经元 i 此刻每条外部输入的放电率（Hz）；extra(t)：直接注入的电流（pA，数组或 null） */
  run(ms, { rate = null, nIn = 60, inContacts = 10, inPRelease = 0.3, inTauRec = 100, learn = true, extra = null, onStep = null } = {}) {
    const { dt, N, rng, ledger, costs, gb } = this;
    const end = this.t + ms - 1e-9;
    const dAr = REC.map((r) => Math.exp(-dt / r.tauR)), dAd = REC.map((r) => Math.exp(-dt / r.tauD));
    const dCa = Math.exp(-dt / gb.tauCa);
    const fired = new Uint8Array(N);
    while (this.t < end) {
      const t = this.t;
      // 突触前资源恢复（与第 2 层 Synapse.step 同序：先恢复、再衰减、再算电流）；易化型突触的释放概率回落
      for (const g of this.groups) {
        for (let i = 0; i < N; i++) g.R[i] += (dt * (1 - g.R[i])) / g.tauRec;
        if (g.tauF) { const f = Math.exp(-dt / g.tauF); for (let i = 0; i < N; i++) g.u[i] *= f; }
        if (g.elig) { const f = Math.exp(-dt / g.plastic.tauE); for (let i = 0; i < N; i++) g.elig[i] *= f; }
        if (g.eBar) { const f = dt / g.plastic.tauBar; for (let i = 0; i < N; i++) g.eBar[i] += f * (g.elig[i] - g.eBar[i]); }
      }
      for (let i = 0; i < N; i++) this.Rin[i] += (dt * (1 - this.Rin[i])) / inTauRec;
      // 外部泊松输入
      if (rate) for (let i = 0; i < N; i++) {
        const r = rate(i, t);
        if (!(r > 0)) continue;
        const k = poisson((nIn * r * dt) / 1000, rng);
        if (!k) continue;
        const ves = k * inContacts * inPRelease * this.Rin[i];
        this.Rin[i] = Math.max(0, this.Rin[i] - (k * inPRelease * this.Rin[i]) / nIn);
        for (let q = 0; q < 2; q++) { const v = ves * REC[q].q * REC[q].k; this.ea[q][i] += v; this.eb[q][i] += v; }
        ledger.add('drive', ves * PRE_ATP_PER_VESICLE + k * costs.atpPerSpike);
      }
      const ext = extra ? extra(t) : null;
      for (let i = 0; i < N; i++) {
        const V = this.V[i], B = mgBlock(V);
        let I = ext ? ext[i] ?? 0 : 0;
        for (let q = 0; q < NREC; q++) {
          this.ra[q][i] *= dAd[q]; this.rb[q][i] *= dAr[q];
          const gq = Math.max(0, this.ra[q][i] - this.rb[q][i]) * (REC[q].mg ? B : 1);
          if (gq === 0) continue;
          const iq = gq * (REC[q].E - V);
          I += iq;
          const fC = Math.abs(iq) * dt;
          if (REC[q].cl) { if (V > REC[q].E) ledger.add('syn_post', ionsFromFemtoCoulomb(fC) * ATP_PER_CL); }
          else if (iq > 0) { const ions = ionsFromFemtoCoulomb(fC); ledger.add('syn_post', ions * REC[q].na * ATP_PER_NA + (ions * REC[q].ca / 2) * ATP_PER_CA); }
        }
        for (let q = 0; q < 2; q++) {
          this.ea[q][i] *= dAd[q]; this.eb[q][i] *= dAr[q];
          const gq = Math.max(0, this.ea[q][i] - this.eb[q][i]) * (REC[q].mg ? B : 1);
          if (gq === 0) continue;
          const iq = gq * (REC[q].E - V);
          I += iq;
          if (iq > 0) { const ions = ionsFromFemtoCoulomb(iq * dt); ledger.add('drive', ions * REC[q].na * ATP_PER_NA + (ions * REC[q].ca / 2) * ATP_PER_CA); }
        }
        // 顶树突：自己的受体电流（NMDA 的镁阻断按树突电位）、钙峰非线性、反传动作电位；只经由平台电流影响胞体
        const a = this.ap?.[i];
        if (a) {
          const Vd = this.Vd[i], Bd = mgBlock(Vd);
          let Id = t < this.bapUntil[i] ? a.cd : 0;
          if (a.modI) Id += this.mod * a.modI; // 胆碱能使顶树突去极化、更易产生平台（学习态）
          for (let q = 0; q < NREC; q++) {
            this.dra[q][i] *= dAd[q]; this.drb[q][i] *= dAr[q];
            const gq = Math.max(0, this.dra[q][i] - this.drb[q][i]) * (REC[q].mg ? Bd : 1);
            if (gq === 0) continue;
            const iq = gq * (REC[q].E - Vd);
            Id += iq;
            const fC = Math.abs(iq) * dt;
            if (REC[q].cl) { if (Vd > REC[q].E) ledger.add('syn_post', ionsFromFemtoCoulomb(fC) * ATP_PER_CL); }
            else if (iq > 0) { const ions = ionsFromFemtoCoulomb(fC); ledger.add('syn_post', ions * REC[q].na * ATP_PER_NA + (ions * REC[q].ca / 2) * ATP_PER_CA); }
          }
          const plateau = Math.max(0, fCa(Vd, a) - a.f0), EL = this.par[i].EL;
          this.plat[i] = plateau;
          this.wd[i] += (dt * (a.ad * (Vd - EL) - this.wd[i])) / a.tauWd;
          this.Vd[i] = Vd + (dt * (-a.gLd * (Vd - EL) + a.gd * plateau - this.wd[i] + Id)) / a.Cd;
          I += a.gs * plateau;
        }
        const adp = this.adpOf?.[i];
        if (adp) { this.adpR[i] *= Math.exp(-dt / adp.rise); this.adpD[i] *= Math.exp(-dt / adp.decay); I += this.adpD[i] - this.adpR[i]; }
        // 第 1 层 LIF（与 LIFNeuron.step 同式同序）
        const p = this.par[i];
        ledger.add('rest', costs.atpRestPerMs * dt);
        this.w[i] += (dt * (p.a * (V - p.EL) - this.w[i])) / p.tauW;
        fired[i] = 0;
        if (t + dt < this.ref[i]) continue;
        this.V[i] = V + (dt * (-p.gL * (V - p.EL) - this.w[i] + I)) / p.C;
        if (this.V[i] >= p.Vth) {
          this.V[i] = p.Vreset; this.w[i] += p.b; this.ref[i] = t + dt + p.tRef;
          fired[i] = 1; this.spikes[i].push(t + dt);
          ledger.add('spike', costs.atpPerSpike);
          if (a) this.bapUntil[i] = t + dt + a.bapMs;
          if (adp) { this.adpR[i] += adp.amp; this.adpD[i] += adp.amp; }
        }
      }
      // 突触前放电 → 释放囊泡、给突触后加电导；钙迹
      for (let j = 0; j < N; j++) {
        if (!fired[j]) continue;
        const o = this.out[j];
        const done = new Set();
        for (let m = 0; m < o.length; m += 2) {
          const g = this.groups[o[m]], k = o[m + 1];
          if (!done.has(o[m])) { done.add(o[m]); if (g.tauF) g.u[j] += g.U * (1 - g.u[j]); if (g.elig) g.elig[j] += 1; }
          const ves = g.contacts * (g.tauF ? g.u[j] : g.pRelease) * g.R[j] * (g.modSup ? 1 - this.mod * g.modSup : 1);
          const i = g.post[k], s = ves * g.scale[k];
          const A = g.target === 'apical' ? this.dra : this.ra, Bq = g.target === 'apical' ? this.drb : this.rb;
          if (g.kind === 'exc') for (let q = 0; q < 2; q++) { const v = s * REC[q].q * REC[q].k; A[q][i] += v; Bq[q][i] += v; }
          else if (g.kind === 'nmda') { const v = s * REC[1].q * REC[1].k; A[1][i] += v; Bq[1][i] += v; }
          else if (g.kind === 'ampa') { const v = s * REC[0].q * REC[0].k; A[0][i] += v; Bq[0][i] += v; }
          else if (g.kind === 'inhFast') { const v = s * REC[3].q * REC[3].k; A[3][i] += v; Bq[3][i] += v; }
          else { const v = s * REC[2].q * REC[2].k; A[2][i] += v; Bq[2][i] += v; }
          ledger.add('syn_pre', ves * PRE_ATP_PER_VESICLE);
        }
        for (const gi of done) { const g = this.groups[gi], p = g.tauF ? g.u[j] : g.pRelease; g.R[j] = Math.max(0, g.R[j] - p * g.R[j]); }
        this.pending.push(t + gb.D, j);
        this.y[j] += 1;
        if (learn && this.istdpGroups) for (const g of this.istdpGroups) {
          if (g.frozen) continue;
          const { eta, target, tau, wMax } = g.istdp, alpha = (2 * target * tau) / 1000;
          for (let m = 0; m < o.length; m += 2) { if (this.groups[o[m]] !== g) continue; const k = o[m + 1]; g.scale[k] = Math.min(wMax, Math.max(0, g.scale[k] + eta * (this.ys[g.post[k]] - alpha))); }
          for (const k of g.inc[j]) g.scale[k] = Math.min(wMax, Math.max(0, g.scale[k] + eta * this.xs[g.pre[k]]));
        }
      }
      if (this.xs) { const dI = Math.exp(-dt / (this.istdpGroups[0].istdp.tau)); for (let i = 0; i < N; i++) { if (fired[i]) { this.xs[i] += 1; this.ys[i] += 1; } this.xs[i] *= dI; this.ys[i] *= dI; } }
      // NMDA 同时检测：突触前放电立刻结合；突触后放电时，每个输入突触按其突触前的结合比例多进一份钙
      if (this.nmda && gb.eta) {
        for (let j = 0; j < N; j++) if (fired[j]) this.nmda[j] += 1;
        for (const g of this.plastic) {
          if (g.frozen || g.plateau) continue;
          for (let i = 0; i < N; i++) {
            if (!fired[i]) continue;
            for (const k of g.incP[i]) g.z[k] += gb.eta * gb.Cpre * this.nmda[g.pre[k]];
          }
        }
      }
      // 钙规则（与第 3 层 CalciumPlasticity 同序：到期的突触前钙先到，再按此刻的钙更新效能，最后钙衰减）
      while (this.pendHead < this.pending.length && this.pending[this.pendHead] <= t) {
        const j = this.pending[this.pendHead + 1];
        this.x[j] += 1; this.pendHead += 2;
        // 电压依赖：突触前经 NMDA 带来的钙按突触后此刻的镁解阻断比例进（顶树突上的突触看顶树突电位）
        if (gb.vdep) { const o = this.out[j], Bref = mgBlock(gb.Vref); for (let m = 0; m < o.length; m += 2) { const g = this.groups[o[m]]; if (!g.cp || g.frozen) continue; const k = o[m + 1], i = g.post[k]; g.cp[k] += mgBlock(g.target === 'apical' ? this.Vd[i] : this.V[i]) / Bref; } }
      }
      if (this.pendHead > 4096) { this.pending = this.pending.slice(this.pendHead); this.pendHead = 0; }
      if (learn) for (const g of this.plastic) {
        if (g.frozen || g.plateau) continue;
        const { sMin, sMax, accel = 1 } = g.plastic, rate_ = (dt * accel) / gb.tau;
        for (let k = 0; k < g.pre.length; k++) {
          const c = gb.Cpre * (g.cp ? g.cp[k] : this.x[g.pre[k]]) + gb.Cpost * this.y[g.post[k]] + g.z[k];
          const r = g.rho[k];
          let drive = -r * (1 - r) * (gb.rhoStar - r);
          if (c > gb.thetaP) drive += gb.gammaP * (1 - r);
          if (c > gb.thetaD) drive -= gb.gammaD * r;
          const nr = Math.min(1, Math.max(0, r + rate_ * drive));
          if (nr !== r) { g.rho[k] = nr; g.scale[k] = sMin + (sMax - sMin) * nr; }
        }
      }
      // 树突预测规则：胞体放电（增强）减去树突电位所预示的放电（削弱），乘突触前的资格迹
      if (learn && this.mod > 0 && this.dendPredGroups) for (const g of this.dendPredGroups) {
        if (g.frozen) continue;
        const { sMin, sMax, etaD, s0, phiMax, vh, k: kk } = g.plastic, c = etaD * this.mod;
        for (let i = 0; i < N; i++) {
          const inc = g.incP[i];
          if (!inc.length) continue;
          const phi = s0 + phiMax / (1 + Math.exp(-(this.Vd[i] - vh) / kk)), dep = (phi * dt) / 1000, spk = fired[i];
          for (const q of inc) {
            const e = g.elig[g.pre[q]];
            if (e < 1e-4) continue;
            const r = g.rho[q], nr = Math.min(1, Math.max(0, r + c * e * ((spk ? 1 - r : 0) - dep * r)));
            if (nr !== r) { g.rho[q] = nr; g.scale[q] = sMin + (sMax - sMin) * nr; }
          }
        }
      }
      // 平台规则：只看此刻处于平台的突触后神经元，把它们带资格迹的输入突触往上写
      if (learn && this.mod > 0) for (const g of this.plateauGroups) {
        if (g.frozen || g.dendPred) continue;
        const { sMin, sMax, etaP, thetaPl } = g.plastic;
        for (let i = 0; i < N; i++) {
          const P = this.plat[i] - thetaPl;
          if (!(P > 0)) continue;
          const gain = dt * etaP * this.mod * P;
          for (const k of g.incP[i]) {
            const j = g.pre[k], d = g.eBar ? g.elig[j] - g.eBar[j] : g.elig[j];
            if (Math.abs(d) < 1e-6) continue;
            const r = g.rho[k], nr = Math.min(1, Math.max(0, r + gain * d * (d > 0 ? 1 - r : r)));
            g.rho[k] = nr; g.scale[k] = sMin + (sMax - sMin) * nr;
          }
        }
      }
      for (let i = 0; i < N; i++) { this.x[i] *= dCa; this.y[i] *= dCa; }
      for (const g of this.plastic) { if (g.frozen || g.plateau) continue; for (let k = 0; k < g.z.length; k++) g.z[k] *= dCa; if (g.cp) for (let k = 0; k < g.cp.length; k++) g.cp[k] *= dCa; }
      if (learn && Math.round(this.t / dt) % Math.max(1, Math.round(10 / dt)) === 0) for (const g of this.plastic) {
        const budget = g.plastic.budget;
        if (budget === undefined || g.frozen) continue;
        const { sMin, sMax } = g.plastic;
        for (let i = 0; i < N; i++) {
          const inc = g.incP[i];
          if (!inc.length) continue;
          let m = 0; for (const k of inc) m += g.rho[k]; m /= inc.length;
          if (m <= budget) continue;
          if (g.plastic.budgetMode === 'mult') { const f = budget / m; for (const k of inc) { g.rho[k] *= f; g.scale[k] = sMin + (sMax - sMin) * g.rho[k]; } continue; }
          const d = m - budget;
          for (const k of inc) { g.rho[k] = Math.max(0, g.rho[k] - d); g.scale[k] = sMin + (sMax - sMin) * g.rho[k]; }
        }
      }
      if (this.nmda) { const dN = Math.exp(-dt / gb.tauN); for (let i = 0; i < N; i++) this.nmda[i] *= dN; }
      this.t += dt;
      onStep?.(this.t, fired);
    }
  }

  /** [t0, t1) 内每个神经元的放电数 */
  counts(t0, t1, ids = null) { return (ids ?? [...Array(this.N).keys()]).map((i) => { let k = 0; for (const s of this.spikes[i]) if (s >= t0 && s < t1) k++; return k; }); }
}
