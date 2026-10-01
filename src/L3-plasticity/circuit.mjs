/** 小型脉冲回路：把第 1–3 层拼起来——LIF 神经元 + 第 2 层突触 + 钙规则可塑性 + 泊松上游驱动，全部记进同一本 ATP 账。
 * 上游驱动 = 每个目标神经元收 n 条来自独立泊松神经元的突触；这些上游神经元的放电与突触成本都记到 'drive'。 */
import { LIFNeuron } from '../L1-neuron/lif.mjs';
import { Synapse } from '../L2-synapse/synapse.mjs';
import { CalciumPlasticity } from './calcium-rule.mjs';
import { EnergyLedger } from '../core/ledger.mjs';
import { makeRng } from '../core/rng.mjs';

export class SpikingCircuit {
  /** neurons: [{ p }]；synapses: [{ pre, post, kind, contacts, pRelease, scale, plastic: { sMin, sMax, rho0, accel, p } }]；
   * drives: [{ targets: [索引], n, contacts, pRelease }]（速率在 run 时给） */
  constructor({ neurons, synapses = [], drives = [], costs, seed = 1, dt = 0.1 }) {
    this.dt = dt;
    this.rng = makeRng(seed);
    this.ledger = new EnergyLedger();
    this.costs = costs;
    this.neurons = neurons.map((n) => new LIFNeuron(n.p ?? {}, costs, this.ledger));
    this.syn = synapses.map((s) => {
      const obj = new Synapse({ kind: s.kind ?? 'exc', contacts: s.contacts ?? 5, pRelease: s.pRelease ?? 0.5, scale: s.scale ?? 1, tauRec: s.tauRec ?? 500, ledger: this.ledger });
      const pl = s.plastic ? new CalciumPlasticity(s.plastic.p ?? {}, { rho0: s.plastic.rho0 ?? 0.5, accel: s.plastic.accel ?? 1 }) : null;
      if (pl) obj.scale = s.plastic.sMin + (s.plastic.sMax - s.plastic.sMin) * pl.rho;
      return { ...s, obj, pl };
    });
    this.drives = drives.map((d) => ({ ...d, syn: d.targets.map(() => Array.from({ length: d.n }, () => new Synapse({ kind: 'exc', contacts: d.contacts ?? 5, pRelease: d.pRelease ?? 0.3, tauRec: d.tauRec ?? 100, ledger: this.ledger, category: 'drive' }))) }));
    this.t = 0;
    this.spikeLog = this.neurons.map(() => []);
  }

  /** 运行 ms 毫秒。driveRate(d, t)：第 d 组驱动此刻每条输入的放电率（Hz）；common(d, t)：此刻是否所有输入同时放电（共同输入） */
  run(ms, { driveRate = () => 0, common = () => false, learn = true, extra = null } = {}) {
    const { dt, rng } = this;
    const N = this.neurons.length;
    const end = this.t + ms;
    const I = new Float64Array(N);
    while (this.t < end) {
      I.fill(0);
      this.drives.forEach((d, di) => {
        const rate = driveRate(di, this.t), together = common(di, this.t);
        d.targets.forEach((ti, k) => {
          for (const s of d.syn[k]) {
            if (together || rng.bernoulliRate(rate, dt)) { s.presynapticSpike(); this.ledger.add('drive', this.costs.atpPerSpike); }
            I[ti] += s.step(dt, this.neurons[ti].V);
          }
        });
      });
      for (const s of this.syn) I[s.post] += s.obj.step(dt, this.neurons[s.post].V);
      if (extra) { const e = extra(this.t); if (e) for (let i = 0; i < N; i++) I[i] += e[i] ?? 0; }
      const fired = this.neurons.map((n, i) => n.step(dt, 0, I[i]));
      fired.forEach((f, i) => { if (f) this.spikeLog[i].push(this.t); });
      for (const s of this.syn) {
        if (fired[s.pre]) { s.obj.presynapticSpike(); s.pl?.preSpike(this.t); }
        if (s.pl) {
          if (fired[s.post]) s.pl.postSpike();
          if (learn) { s.pl.step(dt, this.t); s.obj.scale = s.plastic.sMin + (s.plastic.sMax - s.plastic.sMin) * s.pl.rho; }
        }
      }
      this.t += dt;
    }
  }

  /** [t0, t1) 内各神经元的放电率（Hz） */
  rates(t0, t1) { return this.spikeLog.map((ts) => (ts.filter((t) => t >= t0 && t < t1).length * 1000) / (t1 - t0)); }
}
