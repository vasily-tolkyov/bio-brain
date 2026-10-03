/** 逐层验证：任何一层有未通过的检查即返回非零。FAST=1 跳过脉冲网络的长实验（L4b 约 4 分钟、L4d 约 2 分钟、L4e 约 10 分钟、L4f 约 15 分钟） */
import { verifyL1 } from './L1-neuron.mjs';
import { verifyL2 } from './L2-synapse.mjs';
import { verifyL3 } from './L3-hebbian.mjs';
import { verifyL4 } from './L4-network.mjs';
import { verifyL4Emergence } from './L4-emergence.mjs';
import { verifyL4Uat } from './L4-uat.mjs';
import { verifyL4TwoLayer } from './L4-two-layer.mjs';
import { verifyL4Hierarchy } from './L4-hierarchy.mjs';
import { verifyL4Predictive } from './L4-predictive.mjs';
import { verifyL5 } from './L5-brain.mjs';

const t0 = Date.now();
const results = [];
const layers = [['L1', verifyL1], ['L2', verifyL2], ['L3', verifyL3], ['L4', verifyL4], ['L4b', verifyL4Emergence], ['L4c', verifyL4Uat], ['L4d', verifyL4TwoLayer], ['L4e', verifyL4Hierarchy], ['L4f', verifyL4Predictive], ['L5', verifyL5]]
  .filter(([n]) => !(process.env.FAST && ['L4b', 'L4d', 'L4e', 'L4f'].includes(n)));
for (const [name, fn] of layers) {
  const t = Date.now();
  const { ok } = fn();
  results.push({ name, ok, s: (Date.now() - t) / 1000 });
}
console.log(`\n${results.map((r) => `${r.ok ? '✓' : '✗'} ${r.name}（${r.s.toFixed(0)} s）`).join('  ')}，共 ${((Date.now() - t0) / 1000).toFixed(0)} s`);
process.exit(results.every((r) => r.ok) ? 0 : 1);
