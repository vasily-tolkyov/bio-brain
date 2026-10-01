/** 速率状态的代谢账：每个单元 = 一个神经元，放电率 = r × rMax。
 * 每秒 ATP = N × 静息 + Σ r_i·rMax × (单次放电 + 它的输出突触一次传递的成本)。
 * 单次放电与突触成本来自第 1、2 层的标定（mammalianCosts 与每囊泡成本）。 */
import { PRE_ATP_PER_VESICLE } from '../L2-synapse/synapse.mjs';

export function spikeTotalCost(costs, { outContacts = 1000, pRelease = 0.5, postPerVesicle = 1.65e5 } = {}) {
  return costs.atpPerSpike + outContacts * pRelease * (PRE_ATP_PER_VESICLE + postPerVesicle);
}

export function rateStateAtpPerSecond(r, costs, { rMax = 100, perSpike = spikeTotalCost(costs) } = {}) {
  let spikes = 0;
  for (const x of r) spikes += x * rMax;
  return r.length * costs.atpRestPerMs * 1000 + spikes * perSpike;
}
