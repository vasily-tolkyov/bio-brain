/** 第 3 层验证：钙规则给出 STDP 时间窗；NMDA 同时检测加强"前在后前"的增强；
 * 一群神经元被一起激活后互相增强，之后再要让它们一起放电，所需的上游驱动与整个系统的 ATP 都变少（两本账分开记）。 */
import { pairingChange, GB_NMDA, GB_RATE } from '../src/L3-plasticity/calcium-rule.mjs';
import { SpikingCircuit } from '../src/L3-plasticity/circuit.mjs';
import { mammalianCosts } from '../src/L1-neuron/hh.mjs';
import { checker, fmt } from '../src/core/check.mjs';

export function verifyL3() {
  const c = checker('第 3 层 赫布学习');

  const ds = [-100, -60, -40, -20, -10, -5, 5, 10, 20, 40, 60, 100];
  const dw = ds.map((d) => pairingChange(d));
  const at = (d) => dw[ds.indexOf(d)];
  const peak = Math.max(...dw.map(Math.abs));
  c.note(`60 次配对、1 Hz 的效能变化 Δρ：${ds.map((d, i) => `${d > 0 ? '+' : ''}${d}ms ${fmt(dw[i], 3)}`).join('，')}`);
  c.check('突触前先放（Δt = +5…+20 ms）→ 增强', at(5) > 0 && at(10) > 0 && at(20) > 0, `+10 ms：${fmt(at(10), 3)}`);
  c.check('突触后先放（Δt = −5…−40 ms）→ 减弱', at(-5) < 0 && at(-20) < 0 && at(-40) < 0, `−20 ms：${fmt(at(-20), 3)}`);
  c.check('相隔太远（±100 ms）几乎不变', Math.abs(at(100)) < 0.2 * peak && Math.abs(at(-100)) < 0.2 * peak, `+100：${fmt(at(100), 3)}，−100：${fmt(at(-100), 3)}`);

  const withNmda = (d) => pairingChange(d, { p: { eta: 1 } });
  c.check('NMDA 同时检测（反传电位挪开镁塞时多进的钙）只加强"前在后前"一侧', withNmda(30) > 1.3 * pairingChange(30) && Math.abs(withNmda(-20) - at(-20)) < 0.05 * Math.abs(at(-20)),
    `+30 ms：${fmt(pairingChange(30), 3)} → ${fmt(withNmda(30), 3)}；−20 ms 不变`);

  for (const [freq, deltaT, reference] of [[1, 10, -0.074], [5, 10, -0.113], [10, 10, -0.122], [20, 10, 0.040], [40, 10, 0.112], [20, -10, -0.119]]) {
    const actual = pairingChange(deltaT, { p: GB_RATE, pairs: 60, freq });
    c.check(`GB_RATE 60 次配对：${freq} Hz、${deltaT > 0 ? '+' : ''}${deltaT} ms 与参考值误差不超过 0.005`, Math.abs(actual - reference) <= 0.005,
      `Δρ ${fmt(actual, 6)}，参考 ${fmt(reference, 3)}`);
  }
  const original = pairingChange(10, { p: GB_NMDA, pairs: 60, freq: 1 });
  c.check('GB_NMDA 原规则不变：1 Hz、+10 ms 仍增强约 +0.053', Math.abs(original - 0.053) <= 0.001, `Δρ ${fmt(original, 6)}`);

  // 一群 10 个神经元两两互连，一起被爆发式激活 → 互相增强
  const costs = mammalianCosts();
  const N = 10;
  const mk = (rho0, seed = 1) => {
    const synapses = [];
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) if (i !== j) synapses.push({ pre: i, post: j, contacts: 10, plastic: { sMin: 0.1, sMax: 4, rho0, accel: 5 } });
    return new SpikingCircuit({ costs, seed, neurons: Array.from({ length: N }, () => ({})), synapses, drives: [{ targets: [...Array(N).keys()], n: 30, contacts: 10 }] });
  };
  const learner = mk(0.05);
  for (let k = 0; k < 15; k++) learner.run(1000, { driveRate: (d, t) => ((t % 500) < 100 ? 60 : 2) });
  const rho = learner.syn.reduce((s, x) => s + x.pl.rho, 0) / learner.syn.length;
  c.check('一起被激活 15 s 后，群内突触从弱（ρ=0.05）增强到 ρ > 0.5', rho > 0.5, `平均 ρ ${fmt(rho, 3)}，突触强度 ×${fmt((0.1 + 3.9 * rho) / (0.1 + 3.9 * 0.05), 1)}`);

  const measure = (r0, rate) => {
    const m = mk(r0, 3);
    m.run(300, { driveRate: () => rate, learn: false });
    m.ledger.reset();
    const v0 = m.syn.reduce((s, x) => s + x.obj.vesicles, 0), ts = m.t;
    m.run(1500, { driveRate: () => rate, learn: false });
    const ves = m.syn.reduce((s, x) => s + x.obj.vesicles, 0) - v0;
    const rates = m.rates(ts, m.t);
    return { mean: rates.reduce((s, v) => s + v, 0) / N, atpPerS: m.ledger.total() / 1.5, drive: m.ledger.get('drive') / 1.5, own: (m.ledger.total() - m.ledger.get('drive')) / 1.5, postPerVesicle: m.ledger.get('syn_post') / (ves || 1) };
  };
  const minRate = (r0, target = 10) => { let lo = 0, hi = 80; for (let k = 0; k < 9; k++) { const mid = (lo + hi) / 2; if (measure(r0, mid).mean >= target) hi = mid; else lo = mid; } return hi; };
  const rB = minRate(0.05), rA = minRate(rho);
  const B = measure(0.05, rB), A = measure(rho, rA);
  c.check('学习后，让这群神经元平均放到 10 Hz 所需的上游驱动明显变少（动力学账：所需外力下降）', rA < 0.85 * rB, `上游每条输入 ${fmt(rB, 1)} Hz → ${fmt(rA, 1)} Hz`);
  c.check('整个系统（含上游）每秒的 ATP 随之下降（代谢账）', A.atpPerS < 0.85 * B.atpPerS, `${fmt(B.atpPerS)} → ${fmt(A.atpPerS)} ATP/s（−${fmt((1 - A.atpPerS / B.atpPerS) * 100, 0)}%）`);
  c.check('但群内每次传递更贵了：增强的突触单个囊泡的突触后成本更高', A.postPerVesicle > 1.5 * B.postPerVesicle, `${fmt(B.postPerVesicle)} → ${fmt(A.postPerVesicle)} ATP/囊泡`);
  c.note(`群体自己的开销（静息 + 放电 + 群内突触）：${fmt(B.own)} → ${fmt(A.own)} ATP/s；上游驱动：${fmt(B.drive)} → ${fmt(A.drive)} ATP/s`);
  return { ok: c.done() };
}

if (import.meta.url === `file:///${process.argv[1].replace(/\\/g, '/')}`) process.exit(verifyL3().ok ? 0 : 1);
