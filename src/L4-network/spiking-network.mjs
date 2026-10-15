/** 第 4 层（脉冲版）：由真实点神经元与突触组成的网络，方程与第 1–3 层逐项相同，只是按突触后神经元聚合以便跑上百个神经元：
 * - 神经元：第 1 层的 LIF + 适应（C dV/dt = −g_L(V−E_L) − w + I，τ_w dw/dt = a(V−E_L) − w，放电后 V←V_reset、w += b、不应期）
 * - 突触：第 2 层的双指数 AMPA / NMDA（镁阻断 B(V)）/ GABA_A（另有衰减更快的 GABA_A，篮状细胞式）；同一突触后神经元的同类受体电导是线性叠加的，
 *   所以按神经元存一份状态——与逐个突触各存一份完全等价（verify 里逐拍对拍）；短时程压抑按突触前神经元（同一组连接参数相同）
 * - 可塑性：第 3 层的钙规则。突触 j→i 的钙 = Cpre·x_j(延迟 D) + Cpost·y_i，x、y 是按神经元的指数迹（钙增量线性叠加、衰减常数相同），
 *   所以每个突触的钙不必单独积分，但每个突触有自己的效能 ρ 与强度；抑制性突触可按 Vogels 等 2011 的抑制性可塑性学习
 * - 外部刺激：每个神经元收 nIn 条泊松输入（AMPA + NMDA），它们的放电与突触成本记到 'drive'
 * - ATP：静息、放电、突触前（每囊泡）、突触后（真实受体电流里的 Na⁺ / Ca²⁺ / Cl⁻）、外部驱动，与第 1、2 层同一套换算 */
import { RECEPTORS, PRE_ATP_PER_VESICLE, mgBlock } from '../L2-synapse/synapse.mjs';
import { GB_DP } from '../L3-plasticity/calcium-rule.mjs';
import { LIF_DEFAULT } from '../L1-neuron/lif.mjs';
import { ATP_PER_NA, ATP_PER_CA, ATP_PER_CL, ionsFromFemtoCoulomb } from '../core/units.mjs';
import { EnergyLedger } from '../core/ledger.mjs';
import crypto from 'node:crypto';

const norm = ({ tauR, tauD }) => { const tp = (tauR * tauD) / (tauD - tauR) * Math.log(tauD / tauR); return 1 / (Math.exp(-tp / tauD) - Math.exp(-tp / tauR)); };
const REC = ['AMPA', 'NMDA', 'GABA_A', 'GABA_A_fast'].map((n) => ({ n, ...RECEPTORS[n], k: norm(RECEPTORS[n]) }));
const NREC = REC.length;

const STATE_VECTORS = ['V','w','ref','Rin','x','y','y2','nmda','xs','ys','xsl','Vd','wd','bapUntil'];
const STATE_RECEPTORS = ['ra','rb','ea','eb','dra','drb'];
const GROUP_VECTORS = ['R','rho','scale','z','nm'];
const packVector = value => ({length:value.length,data:Buffer.from(value.buffer,value.byteOffset,value.byteLength).toString('base64')});
function unpackVector(value, length = value.length) {
  if(value.length!==length)throw new Error('状态数组长度不匹配');
  const bytes=Buffer.from(value.data,'base64');
  if(bytes.length!==length*8)throw new Error('状态数组字节数不匹配');
  const out=new Float64Array(length);Buffer.from(out.buffer).set(bytes);return out;
}
function packPlain(value) {
  if(typeof value==='number'&&(!Number.isFinite(value)||Object.is(value,-0)))return {$number:Object.is(value,-0)?'-0':String(value)};
  if(Array.isArray(value))return value.map(packPlain);
  if(value&&typeof value==='object')return Object.fromEntries(Object.entries(value).map(([key,item])=>[key,packPlain(item)]));
  return value;
}
function unpackPlain(value) {
  if(value&&typeof value==='object'&&Object.hasOwn(value,'$number'))return value.$number==='-0'?-0:Number(value.$number);
  if(Array.isArray(value))return value.map(unpackPlain);
  if(value&&typeof value==='object')return Object.fromEntries(Object.entries(value).map(([key,item])=>[key,unpackPlain(item)]));
  return value;
}
function parameterTable(parameters) {
  const table=[],refs=[],seen=new Map();
  for(const p of parameters){if(p===null){refs.push(null);continue;}if(!seen.has(p)){seen.set(p,table.length);table.push(packPlain(p));}refs.push(seen.get(p));}
  return {table,refs};
}
function restoreParameters(saved) {const table=saved.table.map(unpackPlain);return saved.refs.map(index=>index===null?null:table[index]);}
// 与 core/rng.mjs 完全相同的 mulberry32 / Box–Muller 运算；增添状态接口，不增添随机数调用。
function statefulRng(initial=1,saved=null) {
  let s=saved?.s??(initial>>>0),spare=saved?unpackPlain(saved.spare):null;
  const uniform=()=>{s=(s+0x6d2b79f5)>>>0;let t=s;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return ((t^(t>>>14))>>>0)/4294967296;};
  const normal=()=>{if(spare!==null){const value=spare;spare=null;return value;}let u=0,v=0;while(u===0)u=uniform();v=uniform();const r=Math.sqrt(-2*Math.log(u));spare=r*Math.sin(2*Math.PI*v);return r*Math.cos(2*Math.PI*v);};
  const pick=n=>Math.floor(uniform()*n);
  const shuffle=arr=>{for(let i=arr.length-1;i>0;i--){const j=pick(i+1);[arr[i],arr[j]]=[arr[j],arr[i]];}return arr;};
  return {uniform,normal,pick,shuffle,bernoulliRate:(rateHz,dtMs)=>uniform()<1-Math.exp(-rateHz*dtMs/1000),state:()=>s,exportState:()=>({algorithm:'mulberry32',s,spare:packPlain(spare)})};
}
function topologyHash(net) {
  const hash=crypto.createHash('sha256');
  hash.update(JSON.stringify({N:net.N,dt:net.dt,pops:Object.entries(net.pops).map(([name,p])=>[name,p.start,p.n]),apical:net.ap?.map(a=>!!a)??null}));
  for(const g of net.groups){hash.update(JSON.stringify({kind:g.kind,target:g.target,plastic:!!g.plastic,istdp:!!g.istdp,slow:g.slow?[g.slow.tauR,g.slow.tauD]:null}));for(const a of [g.pre,g.post])hash.update(Buffer.from(a.buffer,a.byteOffset,a.byteLength));}
  return hash.digest('hex');
}
function copyStateVector(owner,name,saved,length) {
  const data=unpackVector(saved,length);
  if(owner[name]?.length===length)owner[name].set(data);else owner[name]=data;
}

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
    this.rng = statefulRng(seed);
    this._running = false;
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
    this.t = 0;
    this.spikes = Array.from({ length: N }, () => []);
    this.idx = (name) => { const q = this.pops[name]; return Array.from({ length: q.n }, (_, k) => q.start + k); };
    if (pops.some((p) => p.apical)) {
      this.ap = new Array(N).fill(null);
      for (const p of pops) if (p.apical) {
        const a = { ...APICAL_DEFAULT, ...(p.apical === true ? {} : p.apical) }, q = this.pops[p.name];
        a.f0 = fCa(q.p.EL, a);
        for (let k = 0; k < q.n; k++) this.ap[q.start + k] = a;
      }
      this.Vd = Float64Array.from(this.par, (q) => q.EL); this.wd = new Float64Array(N); this.bapUntil = new Float64Array(N).fill(-Infinity);
      this.dra = REC.map(() => new Float64Array(N)); this.drb = REC.map(() => new Float64Array(N));
    }
  }

  /** 随机连接一组突触。plastic: { sMin, sMax, rho0, accel, budget, tauN } 给定时这组突触按钙规则学习；
   *   budget 给定时加异突触竞争：每 10 ms，一个神经元的输入突触平均效能超过 budget 就把全部输入等量下调（总量有限，强了一部分、别的就得弱）；
   *   tauN 给定时这组突触"先前后后"的配对窗口（突触前放电后 NMDA 受体保持结合的时间）改为 tauN ms，同时检测的增益按 τN/tauN 缩小，
   *   持续的共同放电补进的钙不变，只是相隔更久的"前一个先、后一个后"也能配上；
   * kind：'exc'（AMPA + NMDA）、'nmda'（只有 NMDA：慢、电压依赖，可维持持续活动）、'inh'（GABA_A）、'inhFast'（快速 GABA_A，篮状细胞式）；
   * 'slow'：独立双指数电导，tauR / tauD 以 ms 计，反转电位 0 mV，无镁阻断，按 AMPA 的 q 与离子比例记账；
   * 返回的组上设 frozen = true 时这组突触暂停学习（其余组不受影响，逐拍结果与不设时相同）；
   * istdp: { eta, target, tau, wMax } 给定时（抑制性突触）按 Vogels 等 2011 的抑制性可塑性学习：
   *   突触前抑制性神经元放电时 w += η(y_post − α)，突触后兴奋性神经元放电时 w += η·x_pre，α = 2·目标频率·τ——
   *   突触后放电高于目标频率时抑制变强，低于时变弱，局部地维持兴奋/抑制平衡；抑制最终与兴奋同步起伏（细致平衡），可预期的兴奋被抵消。
   *   tauPre 给定时增强一侧改用更长的突触前痕迹（资格痕迹，几百毫秒），α = 目标频率·(τ + τ_pre) 使不动点仍是目标频率 */
  connect({ from, to, p = 1, contacts = 5, pRelease = 0.3, tauRec = 100, scale = 1, kind = 'exc', plastic = null, istdp = null, pairs = null, autapses = false, target = 'soma', tauR = 10, tauD = 30 }) {
    const pre = [], post = [];
    if (pairs) for (const [a, b] of pairs) { pre.push(a); post.push(b); }
    else for (const a of this.idx(from)) for (const b of this.idx(to)) if ((autapses || a !== b) && this.rng.uniform() < p) { pre.push(a); post.push(b); }
    const n = pre.length;
    if (target === 'apical' && post.some((b) => !this.ap?.[b])) throw new Error('落在顶树突的突触，突触后神经元必须有顶树突分区');
    const g = { kind, target, contacts, pRelease, tauRec, pre: Int32Array.from(pre), post: Int32Array.from(post), R: new Float64Array(this.N).fill(1), plastic,
      rho: plastic ? new Float64Array(n).fill(plastic.rho0 ?? 0.5) : null, scale: new Float64Array(n).fill(scale) };
    if (plastic) {
      for (let k = 0; k < n; k++) g.scale[k] = plastic.sMin + (plastic.sMax - plastic.sMin) * g.rho[k];
      g.z = new Float64Array(n); // 每个突触自己的"同时检测"钙
      g.incP = Array.from({ length: this.N }, () => []);
      for (let k = 0; k < n; k++) g.incP[post[k]].push(k);
      this.nmda ??= new Float64Array(this.N);
      if (plastic.tauN) g.nm = new Float64Array(this.N);
    }
    if (istdp) {
      g.istdp = { eta: 0.02, target: 5, tau: 20, wMax: 20, ...istdp };
      g.inc = Array.from({ length: this.N }, () => []);
      for (let k = 0; k < n; k++) g.inc[post[k]].push(k);
      this.istdpGroups = [...(this.istdpGroups ?? []), g];
      this.xs ??= new Float64Array(this.N); this.ys ??= new Float64Array(this.N);
      if (g.istdp.tauPre) { this.xsl ??= new Float64Array(this.N); this.tauPreLong = g.istdp.tauPre; }
    }
    /** 创建连接时，为慢受体分配独立双指数状态和稀疏突触后索引。 */
    if(kind==='slow'){
      if(!(Number.isFinite(tauR)&&Number.isFinite(tauD)&&tauR>0&&tauD>tauR))throw new Error('slow 需要有限的 0 < tauR < tauD（ms）');
      g.slow={tauR,tauD,k:norm({tauR,tauD}),a:new Float64Array(this.N),b:new Float64Array(this.N)};
      this.slowGroups??=[];this.slowGroups.push(g);
      this.slowSoma??=Array.from({length:this.N},()=>[]);
      this.slowApical??=Array.from({length:this.N},()=>[]);
      const incoming=target==='apical'?this.slowApical:this.slowSoma;
      for(const i of [...new Set(post)].sort((a,b)=>a-b))incoming[i].push(g);
    }
    const gi = this.groups.push(g) - 1;
    for (let k = 0; k < n; k++) this.out[pre[k]].push(gi, k);
    if (plastic) this.plastic.push(g);
    return g;
  }

  /** 运行 ms。rate(i, t)：神经元 i 此刻每条外部输入的放电率（Hz）；extra(t)：直接注入的电流（pA，数组或 null）。
   * onStepWritesResources=false 声明回调不写任何 R，可保留按实际突触前编号恢复资源的路径。默认保持原行为。 */
  run(ms, { rate = null, nIn = 60, inContacts = 10, inPRelease = 0.3, inTauRec = 100, learn = true, extra = null, onStep = null, onStepWritesResources = true } = {}) {
    if (this._running) throw new Error('同一个网络不支持嵌套 run');
    const { dt, N, rng, ledger, costs, gb } = this;
    const end = this.t + ms - 1e-9;
    const dAr = REC.map((r) => Math.exp(-dt / r.tauR)), dAd = REC.map((r) => Math.exp(-dt / r.tauD));
    const dCa = Math.exp(-dt / gb.tauCa);
    const dSlow = new Map((this.slowGroups ?? []).map(g => [g, [Math.exp(-dt/g.slow.tauR), Math.exp(-dt/g.slow.tauD)]]));
    const fired = new Uint8Array(N);
    const seen = new Uint8Array(this.groups.length), done = [];
    const preCa = new Float64Array(N), postCa = new Float64Array(N);
    // 未参与释放的资源恒为 1；保留外部预置的非 1 项。有逐步回调时允许回调改任意资源。
    const resourceIds = this.groups.map(g => onStep && onStepWritesResources ? null : Int32Array.from(
      [...new Set([...g.pre, ...Array.from(g.R, (r,i) => r !== 1 ? i : -1).filter(i => i >= 0)])].sort((a,b) => a-b)));
    const ownsBatch = ledger.beginBatch();
    this._running = true;
    try {
    while (this.t < end) {
      const t = this.t;
      // 突触前资源恢复（与第 2 层 Synapse.step 同序：先恢复、再衰减、再算电流）
      for (let gi = 0; gi < this.groups.length; gi++) {
        const g = this.groups[gi], ids = resourceIds[gi];
        if (ids) for (const i of ids) g.R[i] += (dt * (1 - g.R[i])) / g.tauRec;
        else for (let i = 0; i < N; i++) g.R[i] += (dt * (1 - g.R[i])) / g.tauRec;
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
        ledger.addAt(0, ves * PRE_ATP_PER_VESICLE + k * costs.atpPerSpike);
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
          if (REC[q].cl) { if (V > REC[q].E) ledger.addAt(4, ionsFromFemtoCoulomb(fC) * ATP_PER_CL); }
          else if (iq > 0) { const ions = ionsFromFemtoCoulomb(fC); ledger.addAt(4, ions * REC[q].na * ATP_PER_NA + (ions * REC[q].ca / 2) * ATP_PER_CA); }
        }
        if(this.slowSoma)for(const g of this.slowSoma[i]){
          g.slow.a[i]*=dSlow.get(g)[1];g.slow.b[i]*=dSlow.get(g)[0];
          const gq=Math.max(0,g.slow.a[i]-g.slow.b[i]);
          if(gq===0)continue;
          const iq=gq*(0-V);I+=iq;
          if(iq>0){const ions=ionsFromFemtoCoulomb(iq*dt);ledger.addAt(4,ions*REC[0].na*ATP_PER_NA+(ions*REC[0].ca/2)*ATP_PER_CA);}
        }
        for (let q = 0; q < 2; q++) {
          this.ea[q][i] *= dAd[q]; this.eb[q][i] *= dAr[q];
          const gq = Math.max(0, this.ea[q][i] - this.eb[q][i]) * (REC[q].mg ? B : 1);
          if (gq === 0) continue;
          const iq = gq * (REC[q].E - V);
          I += iq;
          if (iq > 0) { const ions = ionsFromFemtoCoulomb(iq * dt); ledger.addAt(0, ions * REC[q].na * ATP_PER_NA + (ions * REC[q].ca / 2) * ATP_PER_CA); }
        }
        // 顶树突：自己的受体电流（NMDA 的镁阻断按树突电位）、钙峰非线性、反传动作电位；只经由平台电流影响胞体
        const a = this.ap?.[i];
        if (a) {
          const Vd = this.Vd[i], Bd = mgBlock(Vd);
          let Id = t < this.bapUntil[i] ? a.cd : 0;
          for (let q = 0; q < NREC; q++) {
            this.dra[q][i] *= dAd[q]; this.drb[q][i] *= dAr[q];
            const gq = Math.max(0, this.dra[q][i] - this.drb[q][i]) * (REC[q].mg ? Bd : 1);
            if (gq === 0) continue;
            const iq = gq * (REC[q].E - Vd);
            Id += iq;
            const fC = Math.abs(iq) * dt;
            if (REC[q].cl) { if (Vd > REC[q].E) ledger.addAt(4, ionsFromFemtoCoulomb(fC) * ATP_PER_CL); }
            else if (iq > 0) { const ions = ionsFromFemtoCoulomb(fC); ledger.addAt(4, ions * REC[q].na * ATP_PER_NA + (ions * REC[q].ca / 2) * ATP_PER_CA); }
          }
          if(this.slowApical)for(const g of this.slowApical[i]){
            g.slow.a[i]*=dSlow.get(g)[1];g.slow.b[i]*=dSlow.get(g)[0];
            const gq=Math.max(0,g.slow.a[i]-g.slow.b[i]);
            if(gq===0)continue;
            const iq=gq*(0-Vd);Id+=iq;
            if(iq>0){const ions=ionsFromFemtoCoulomb(iq*dt);ledger.addAt(4,ions*REC[0].na*ATP_PER_NA+(ions*REC[0].ca/2)*ATP_PER_CA);}
          }
          const plateau = Math.max(0, fCa(Vd, a) - a.f0), EL = this.par[i].EL;
          this.wd[i] += (dt * (a.ad * (Vd - EL) - this.wd[i])) / a.tauWd;
          this.Vd[i] = Vd + (dt * (-a.gLd * (Vd - EL) + a.gd * plateau - this.wd[i] + Id)) / a.Cd;
          I += a.gs * plateau;
        }
        // 第 1 层 LIF（与 LIFNeuron.step 同式同序）
        const p = this.par[i];
        ledger.addAt(1, costs.atpRestPerMs * dt);
        this.w[i] += (dt * (p.a * (V - p.EL) - this.w[i])) / p.tauW;
        fired[i] = 0;
        if (t + dt < this.ref[i]) continue;
        this.V[i] = V + (dt * (-p.gL * (V - p.EL) - this.w[i] + I)) / p.C;
        if (this.V[i] >= p.Vth) {
          this.V[i] = p.Vreset; this.w[i] += p.b; this.ref[i] = t + dt + p.tRef;
          fired[i] = 1; this.spikes[i].push(t + dt);
          ledger.addAt(2, costs.atpPerSpike);
          if (a) this.bapUntil[i] = t + dt + a.bapMs;
        }
      }
      // 突触前放电 → 释放囊泡、给突触后加电导；钙迹
      for (let j = 0; j < N; j++) {
        if (!fired[j]) continue;
        const o = this.out[j];
        done.length = 0;
        for (let m = 0; m < o.length; m += 2) {
          const g = this.groups[o[m]], k = o[m + 1];
          const ves = g.contacts * g.pRelease * g.R[j];
          if (!seen[o[m]]) { seen[o[m]] = 1; done.push(o[m]); }
          const i = g.post[k], s = ves * g.scale[k];
          const A = g.target === 'apical' ? this.dra : this.ra, Bq = g.target === 'apical' ? this.drb : this.rb;
          if (g.kind === 'slow') { const v = s * REC[0].q * g.slow.k; g.slow.a[i] += v; g.slow.b[i] += v; }
          else if (g.kind === 'exc') for (let q = 0; q < 2; q++) { const v = s * REC[q].q * REC[q].k; A[q][i] += v; Bq[q][i] += v; }
          else if (g.kind === 'nmda') { const v = s * REC[1].q * REC[1].k; A[1][i] += v; Bq[1][i] += v; }
          else if (g.kind === 'inhFast') { const v = s * REC[3].q * REC[3].k; A[3][i] += v; Bq[3][i] += v; }
          else { const v = s * REC[2].q * REC[2].k; A[2][i] += v; Bq[2][i] += v; }
          ledger.addAt(3, ves * PRE_ATP_PER_VESICLE);
        }
        for (const gi of done) { const g = this.groups[gi]; g.R[j] = Math.max(0, g.R[j] - g.pRelease * g.R[j]); seen[gi] = 0; }
        this.pending.push(t + gb.D, j);
        this.y[j] += 1;
        if (learn && this.istdpGroups) for (const g of this.istdpGroups) {
          if (g.frozen) continue;
          const { eta, target, tau, wMax, tauPre } = g.istdp, alpha = (target * (tau + (tauPre ?? tau))) / 1000, xr = tauPre ? this.xsl : this.xs;
          for (let m = 0; m < o.length; m += 2) { if (this.groups[o[m]] !== g) continue; const k = o[m + 1]; g.scale[k] = Math.min(wMax, Math.max(0, g.scale[k] + eta * (this.ys[g.post[k]] - alpha))); }
          for (const k of g.inc[j]) g.scale[k] = Math.min(wMax, Math.max(0, g.scale[k] + eta * xr[g.pre[k]]));
        }
      }
      if (this.xs) { const dI = Math.exp(-dt / (this.istdpGroups[0].istdp.tau)); for (let i = 0; i < N; i++) { if (fired[i]) { this.xs[i] += 1; this.ys[i] += 1; } this.xs[i] *= dI; this.ys[i] *= dI; } }
      if (this.xsl) { const dL = Math.exp(-dt / this.tauPreLong); for (let i = 0; i < N; i++) { if (fired[i]) this.xsl[i] += 1; this.xsl[i] *= dL; } }
      // NMDA 同时检测：突触前放电立刻结合；突触后放电时，每个输入突触按其突触前的结合比例多进一份钙
      if (this.nmda && gb.eta) {
        for (let j = 0; j < N; j++) if (fired[j]) this.nmda[j] += 1;
        for (const g of this.plastic) if (g.nm) for (let j = 0; j < N; j++) if (fired[j]) g.nm[j] += 1;
        const y2 = gb.post2Tau ? (this.y2 ??= new Float64Array(N)) : null;
        for (const g of this.plastic) {
          if (g.frozen) continue;
          const tr = g.nm ?? this.nmda, eta = g.nm ? (gb.eta * gb.tauN) / g.plastic.tauN : gb.eta;
          for (let i = 0; i < N; i++) {
            if (!fired[i]) continue;
            const f = y2 ? y2[i] : 1;
            if (f === 0) continue;
            for (const k of g.incP[i]) g.z[k] += eta * gb.Cpre * tr[g.pre[k]] * f;
          }
        }
        // 与 CalciumPlasticity 同序：用本次脉冲之前的迹，再先衰减、后加 1。
        if (y2) { const d2 = Math.exp(-dt / gb.post2Tau); for (let i = 0; i < N; i++) { y2[i] *= d2; if (fired[i]) y2[i] += 1; } }
      }
      // 钙规则（与第 3 层 CalciumPlasticity 同序：到期的突触前钙先到，再按此刻的钙更新效能，最后钙衰减）
      while (this.pendHead < this.pending.length && this.pending[this.pendHead] <= t) { this.x[this.pending[this.pendHead + 1]] += 1; this.pendHead += 2; }
      if (this.pendHead > 4096) { this.pending = this.pending.slice(this.pendHead); this.pendHead = 0; }
      if (learn && this.plastic.length) for (let i = 0; i < N; i++) { preCa[i] = gb.Cpre * this.x[i]; postCa[i] = gb.Cpost * this.y[i]; }
      if (learn) for (const g of this.plastic) {
        if (g.frozen) continue;
        const { sMin, sMax, accel = 1 } = g.plastic, rate_ = (dt * accel) / gb.tau, sRange = sMax - sMin;
        const { pre, post, z, rho, scale } = g;
        for (let k = 0; k < pre.length; k++) {
          const c = preCa[pre[k]] + postCa[post[k]] + z[k];
          const r = rho[k];
          let drive = -r * (1 - r) * (gb.rhoStar - r);
          if (c > gb.thetaP) drive += gb.gammaP * (1 - r);
          if (c > gb.thetaD) drive -= gb.gammaD * r;
          const nr = Math.min(1, Math.max(0, r + rate_ * drive));
          if (nr !== r) { rho[k] = nr; scale[k] = sMin + sRange * nr; }
        }
      }
      for (let i = 0; i < N; i++) { this.x[i] *= dCa; this.y[i] *= dCa; }
      for (const g of this.plastic) { if (g.frozen) continue; const z = g.z; for (let k = 0; k < z.length; k++) z[k] *= dCa; }
      if (learn && Math.round(this.t / dt) % Math.max(1, Math.round(10 / dt)) === 0) for (const g of this.plastic) {
        const budget = g.plastic.budget;
        if (budget === undefined || g.frozen) continue;
        const { sMin, sMax } = g.plastic;
        for (let i = 0; i < N; i++) {
          const inc = g.incP[i];
          if (!inc.length) continue;
          let m = 0; for (const k of inc) m += g.rho[k]; m /= inc.length;
          if (m <= budget) continue;
          const d = m - budget;
          for (const k of inc) { g.rho[k] = Math.max(0, g.rho[k] - d); g.scale[k] = sMin + (sMax - sMin) * g.rho[k]; }
        }
      }
      if (this.nmda) { const dN = Math.exp(-dt / gb.tauN); for (let i = 0; i < N; i++) this.nmda[i] *= dN; }
      for (const g of this.plastic) if (g.nm) { const dG = Math.exp(-dt / g.plastic.tauN); for (let i = 0; i < N; i++) g.nm[i] *= dG; }
      this.t += dt;
      if (onStep) { ledger.flushBatch(); onStep(this.t, fired); ledger.syncBatch(); }
    }
    } finally { this._running = false; if (ownsBatch) ledger.endBatch(); }
  }

  /** JSON 可往返的完整动态状态；Float64 直接编码字节，保留 -Infinity、NaN 与负零。 */
  exportState() {
    const rng=this.rng.exportState?.()??(typeof this.rng.state?.()==='number'?{algorithm:'mulberry32',s:this.rng.state(),spare:null}:null);
    if(!rng||rng.algorithm!=='mulberry32')throw new Error('随机数生成器没有可续算的状态接口');
    const parameters=parameterTable(this.par);
    const popParameters=Object.fromEntries(Object.entries(this.pops).map(([name,p])=>[name,parameters.refs[this.par.indexOf(p.p)]??null]));
    const vectors=Object.fromEntries(STATE_VECTORS.filter(name=>this[name]).map(name=>[name,packVector(this[name])]));
    const receptors=Object.fromEntries(STATE_RECEPTORS.filter(name=>this[name]).map(name=>[name,this[name].map(packVector)]));
    return {schema:1,encoding:'float64-bytes-base64',topology:topologyHash(this),N:this.N,dt:this.dt,t:this.t,
      vectors,receptors,parameters,popParameters,popFallback:packPlain(Object.fromEntries(Object.entries(this.pops).map(([name,p])=>[name,p.p]))),
      apical:this.ap?parameterTable(this.ap):null,gb:packPlain(this.gb),costs:packPlain(this.costs),tauPreLong:packPlain(this.tauPreLong??null),
      groups:this.groups.map(g=>({vectors:Object.fromEntries(GROUP_VECTORS.filter(name=>g[name]).map(name=>[name,packVector(g[name])])),
        params:packPlain({contacts:g.contacts,pRelease:g.pRelease,tauRec:g.tauRec,plastic:g.plastic,istdp:g.istdp??null}),
        frozen:Object.hasOwn(g,'frozen')?g.frozen:null,slow:g.slow?{tauR:g.slow.tauR,tauD:g.slow.tauD,k:g.slow.k,a:packVector(g.slow.a),b:packVector(g.slow.b)}:null})),
      pending:packVector(Float64Array.from(this.pending)),pendHead:this.pendHead,spikes:this.spikes.map(row=>packVector(Float64Array.from(row))),
      rng,ledger:packPlain([...this.ledger.atp])};
  }

  importState(state) {
    if(this._running)throw new Error('请在 run 调用之间导入状态');
    if(state.schema!==1||state.N!==this.N||state.dt!==this.dt||state.groups.length!==this.groups.length||state.topology!==topologyHash(this))throw new Error('状态与网络拓扑不匹配');
    // 先验证全部字节与长度，避免格式错误的状态部分写入网络。
    for(const name of ['V','w','ref','Rin','x','y'])if(!state.vectors?.[name])throw new Error('缺少必需的动态数组');
    if(state.parameters?.refs.length!==this.N||state.parameters.refs.some(i=>!Number.isInteger(i)||i<0||i>=state.parameters.table.length))throw new Error('神经元参数引用无效');
    if(!Number.isFinite(state.t)||state.t<0)throw new Error('模拟时刻无效');
    for(const name of ['ra','rb','ea','eb'])if(!state.receptors?.[name])throw new Error('缺少受体状态');
    if(this.ap)for(const name of ['Vd','wd','bapUntil'])if(!state.vectors?.[name])throw new Error('缺少顶树突状态');
    if(this.ap&&(!state.receptors.dra||!state.receptors.drb||state.apical?.refs.length!==this.N))throw new Error('缺少顶树突受体或参数');
    for(const [name,value] of Object.entries(state.vectors)){if(!STATE_VECTORS.includes(name))throw new Error('未知状态数组');unpackVector(value,this.N);}
    for(const [name,rows] of Object.entries(state.receptors)){
      if(rows.length!==(this[name]?.length??rows.length))throw new Error('受体状态不匹配');
      for(const value of rows)unpackVector(value,this.N);
    }
    state.groups.forEach((saved,i)=>{const g=this.groups[i];if(!saved.vectors.R||!saved.vectors.scale||(g.plastic&&(!saved.vectors.rho||!saved.vectors.z))||!!saved.slow!==!!g.slow)throw new Error('缺少连接组状态');for(const [name,value] of Object.entries(saved.vectors))unpackVector(value,['R','nm'].includes(name)?this.N:g.pre.length);if(saved.slow){unpackVector(saved.slow.a,this.N);unpackVector(saved.slow.b,this.N);}});
    const pending=unpackVector(state.pending);
    if(state.spikes.length!==this.N||state.pendHead<0||state.pendHead>pending.length||state.pendHead%2||pending.length%2)throw new Error('脉冲队列状态不匹配');
    const spikes=state.spikes.map(row=>unpackVector(row));
    if(state.rng?.algorithm!=='mulberry32'||!Number.isInteger(state.rng.s)||state.rng.s<0||state.rng.s>0xffffffff)throw new Error('随机数状态无效');
    for(const name of STATE_VECTORS){if(state.vectors[name])copyStateVector(this,name,state.vectors[name],this.N);else delete this[name];}
    for(const name of STATE_RECEPTORS){if(state.receptors[name]){this[name]??=state.receptors[name].map(()=>new Float64Array(this.N));state.receptors[name].forEach((value,k)=>copyStateVector(this[name],k,value,this.N));}else delete this[name];}
    this.par=restoreParameters(state.parameters);
    const fallback=unpackPlain(state.popFallback);
    for(const [name,p] of Object.entries(this.pops)){const index=state.popParameters[name];p.p=index===null?fallback[name]:this.par[state.parameters.refs.indexOf(index)];}
    if(state.apical)this.ap=restoreParameters(state.apical);
    this.gb=unpackPlain(state.gb);this.costs=unpackPlain(state.costs);
    if(state.tauPreLong===null)delete this.tauPreLong;else this.tauPreLong=unpackPlain(state.tauPreLong);
    state.groups.forEach((saved,i)=>{
      const g=this.groups[i];
      for(const name of GROUP_VECTORS){if(saved.vectors[name])copyStateVector(g,name,saved.vectors[name],['R','nm'].includes(name)?this.N:g.pre.length);else if(name==='rho')g.rho=null;else delete g[name];}
      Object.assign(g,unpackPlain(saved.params));
      if(g.istdp===null)delete g.istdp;
      if(saved.frozen===null)delete g.frozen;else g.frozen=saved.frozen;
      if(saved.slow){Object.assign(g.slow,{tauR:saved.slow.tauR,tauD:saved.slow.tauD,k:saved.slow.k});copyStateVector(g.slow,'a',saved.slow.a,this.N);copyStateVector(g.slow,'b',saved.slow.b,this.N);}
    });
    this.plastic=this.groups.filter(g=>g.plastic);
    const inhibition=this.groups.filter(g=>g.istdp);
    if(inhibition.length)this.istdpGroups=inhibition;else delete this.istdpGroups;
    this.pending=[...pending];this.pendHead=state.pendHead;
    spikes.forEach((data,i)=>{this.spikes[i].length=0;for(const t of data)this.spikes[i].push(t);});
    this.rng=statefulRng(1,state.rng);this.ledger.atp=new Map(unpackPlain(state.ledger));this.t=state.t;
    return this;
  }

  /** [t0, t1) 内每个神经元的放电数 */
  counts(t0, t1, ids = null) { return (ids ?? [...Array(this.N).keys()]).map((i) => { let k = 0; for (const s of this.spikes[i]) if (s >= t0 && s < t1) k++; return k; }); }
}
