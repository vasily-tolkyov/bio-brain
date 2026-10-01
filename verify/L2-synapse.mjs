/** 第 2 层验证：突触数决定强度、NMDA 的电压依赖、每次传递的 ATP、短时程压抑、整次放电里突触占的能耗份额 */
import { Synapse, mgBlock, PRE_ATP_PER_VESICLE } from '../src/L2-synapse/synapse.mjs';
import { LIFNeuron } from '../src/L1-neuron/lif.mjs';
import { EnergyLedger } from '../src/core/ledger.mjs';
import { calibrateHH, costsForArea } from '../src/L1-neuron/hh.mjs';
import { checker, fmt } from '../src/core/check.mjs';

const pearson = (x, y) => {
  const n = x.length, mx = x.reduce((s, v) => s + v, 0) / n, my = y.reduce((s, v) => s + v, 0) / n;
  let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < n; i++) { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) ** 2; syy += (y[i] - my) ** 2; }
  return sxy / Math.sqrt(sxx * syy);
};

/** 无放电的突触后膜（阈值抬高）上，单个突触前脉冲引起的 EPSP 峰值与这次传递的 ATP */
function epsp(contacts, { kind = 'exc', V0 = -65 } = {}) {
  const ledger = new EnergyLedger();
  const syn = new Synapse({ kind, contacts, pRelease: 0.5, ledger });
  const post = new LIFNeuron({ Vth: 1e9, EL: V0 });
  const dt = 0.05;
  syn.presynapticSpike();
  let peak = 0;
  for (let t = 0; t < 400; t += dt) { const I = syn.step(dt, post.V); post.step(dt, 0, I); peak = Math.max(peak, Math.abs(post.V - V0)); }
  return { peak, ledger, vesicles: syn.vesicles };
}

export function verifyL2() {
  const c = checker('第 2 层 突触连接');
  const ns = [1, 2, 4, 6, 8, 10, 15, 20];
  const amps = ns.map((n) => epsp(n).peak);
  c.check('同一类连接里，EPSP 与接触点数线性相关（果蝇 ORN→PN 实测 r = 0.993–0.999）', pearson(ns, amps) > 0.99,
    `r = ${fmt(pearson(ns, amps), 4)}；1 个接触 ${fmt(amps[0], 3)} mV → 20 个 ${fmt(amps.at(-1))} mV`);
  const big = [40, 60].map((n) => epsp(n).peak);
  c.check('接触点很多时变得亚线性（驱动力 E − V 随去极化减小）', (big[1] - big[0]) / 20 < (amps[2] - amps[0]) / 3,
    `每加一个接触：小连接 +${fmt((amps[2] - amps[0]) / 3, 3)} mV，大连接 +${fmt((big[1] - big[0]) / 20, 3)} mV`);

  const ratio = (mgBlock(-20) * 20) / (mgBlock(-70) * 70);
  c.check('NMDA 的镁阻断：同样电导下 −20 mV 的电流是 −70 mV 的 3 倍以上（去极化才放行——赫布学习的"同时检测"靠它）', ratio > 3,
    `B(−70)=${fmt(mgBlock(-70), 3)}，B(−20)=${fmt(mgBlock(-20), 3)}，电流比 ${fmt(ratio)}`);

  const post = ns.map((n) => epsp(n).ledger.get('syn_post'));
  const one = epsp(10);
  const perVesicle = one.ledger.get('syn_post') / one.vesicles;
  c.check('每次传递的突触后 ATP 与电导（接触点数）成正比：越强的突触单次越贵', pearson(ns, post) > 0.99, `r = ${fmt(pearson(ns, post), 4)}`);
  c.check('每个囊泡的突触后成本在 10⁴–10⁶ ATP（皮层估算约 1.4×10⁵）', perVesicle > 1e4 && perVesicle < 1e6,
    `突触后 ${fmt(perVesicle)} ATP/囊泡，突触前 ${fmt(PRE_ATP_PER_VESICLE)} ATP/囊泡`);

  const syn = new Synapse({ contacts: 10, pRelease: 0.5, tauRec: 500 });
  const v1 = syn.presynapticSpike();
  for (let t = 0; t < 50; t += 0.05) syn.step(0.05, -65);
  const v2 = syn.presynapticSpike();
  for (let t = 0; t < 1500; t += 0.05) syn.step(0.05, -65);
  const v3 = syn.presynapticSpike();
  c.check('短时程压抑：20 Hz 的第二个脉冲释放更少，1.5 s 后恢复', v2 / v1 < 0.8 && v3 / v1 > 0.9, `成对脉冲比 ${fmt(v2 / v1)}，恢复后 ${fmt(v3 / v1)}`);

  const inh = epsp(10, { kind: 'inh', V0: -55 });
  c.check('GABA_A 在静息之上产生抑制性电位，并要付 Cl⁻ 外排的 ATP', inh.peak > 0.1 && inh.ledger.get('syn_post') > 0, `IPSP ${fmt(inh.peak, 3)} mV，${fmt(inh.ledger.get('syn_post'))} ATP`);

  // 一次放电的全部代价：自己的动作电位 + 它的输出突触（突触前 + 突触后）
  const cal = calibrateHH();
  const mammal = costsForArea(cal, 20000, { spikeEfficiency: 1.3 / cal.overlap });
  const outContacts = 1000, ves = outContacts * 0.5;
  const synCost = ves * (PRE_ATP_PER_VESICLE + perVesicle);
  const share = synCost / (synCost + mammal.atpPerSpike);
  c.note(`一个 20000 µm² 的高效（哺乳动物式，重叠 1.3）神经元放一次电：动作电位 ${fmt(mammal.atpPerSpike)} ATP；1000 个输出接触释放 ${ves} 个囊泡 ${fmt(synCost)} ATP`);
  c.check('突触传递占一次放电总成本的大头（皮层估算：突触后约 50%）', share > 0.4, `突触占 ${fmt(share * 100, 0)}%`);
  return { ok: c.done(), perVesicle };
}

if (import.meta.url === `file:///${process.argv[1].replace(/\\/g, '/')}`) process.exit(verifyL2().ok ? 0 : 1);
