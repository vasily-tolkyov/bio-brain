/** 逐层验证：任何一层有未通过的检查即返回非零 */
import { verifyL1 } from './L1-neuron.mjs';
import { verifyL2 } from './L2-synapse.mjs';
import { verifyL3 } from './L3-hebbian.mjs';
import { verifyL4 } from './L4-network.mjs';
import { verifyL5 } from './L5-brain.mjs';

const t0 = Date.now();
const results = [];
for (const [name, fn] of [['L1', verifyL1], ['L2', verifyL2], ['L3', verifyL3], ['L4', verifyL4], ['L5', verifyL5]]) {
  const t = Date.now();
  const { ok } = fn();
  results.push({ name, ok, s: (Date.now() - t) / 1000 });
}
console.log(`\n${results.map((r) => `${r.ok ? '✓' : '✗'} ${r.name}（${r.s.toFixed(0)} s）`).join('  ')}，共 ${((Date.now() - t0) / 1000).toFixed(0)} s`);
process.exit(results.every((r) => r.ok) ? 0 : 1);
