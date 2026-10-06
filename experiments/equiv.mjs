import { SpikingNetwork } from '../src/L4-network/spiking-network.mjs';
import { SpikingCircuit } from '../src/L3-plasticity/circuit.mjs';
import { GB_DP, GB_NMDA } from '../src/L3-plasticity/calcium-rule.mjs';
import { mammalianCosts } from '../src/L1-neuron/hh.mjs';

const costs = mammalianCosts();
const pairs = [[0, 1], [1, 2], [2, 0], [0, 2]];
const extra = (t) => [t % 200 < 60 ? 400 : 0, t % 300 < 40 ? 350 : 0, (t + 50) % 250 < 50 ? 380 : 0];
for (const [name, gb] of [['DP', GB_DP], ['NMDA', GB_NMDA]]) {
  const plastic = { sMin: 0.1, sMax: 4, rho0: 0.3, accel: 5 };
  const obj = new SpikingCircuit({ costs, seed: 1, neurons: [{}, {}, {}], synapses: pairs.map(([a, b]) => ({ pre: a, post: b, contacts: 10, pRelease: 0.3, tauRec: 100, plastic: { ...plastic, p: gb } })) });
  obj.run(3000, { extra });
  const fast = new SpikingNetwork({ costs, pops: [{ name: 'E', n: 3 }], plasticity: gb });
  fast.connect({ pairs, contacts: 10, pRelease: 0.3, tauRec: 100, plastic });
  fast.run(3000, { extra });
  const so = obj.neurons.map((n) => n.spikes), sf = fast.spikes;
  let maxDt = 0, nmis = 0;
  for (let i = 0; i < 3; i++) { if (so[i].length !== sf[i].length) nmis++; for (let k = 0; k < Math.min(so[i].length, sf[i].length); k++) maxDt = Math.max(maxDt, Math.abs(so[i][k] - sf[i][k])); }
  console.log(name, 'spikes', so.map((s) => s.length), sf.map((s) => s.length), 'maxΔt', maxDt, 'rho', obj.syn.map((s) => s.pl.rho.toFixed(6)).join(','), '|', [...fast.groups[0].rho].map((r) => r.toFixed(6)).join(','), 'ATP', obj.ledger.total().toExponential(5), fast.ledger.total().toExponential(5));
}
