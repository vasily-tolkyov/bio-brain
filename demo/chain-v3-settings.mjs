/** v3 固定的四个候选；CONFIG 仍为 v2，设置作为独立指纹的一部分。 */
export const CANDIDATES = Object.freeze({
  L1a: { eePlastic: { tauN: 80 }, plasticityOverrides: {} },
  L1b: { eePlastic: { tauN: 120 }, plasticityOverrides: {} },
  L1c: { eePlastic: {}, plasticityOverrides: { post2Tau: 50, eta: 2.4 } },
  L1d: { eePlastic: { tauN: 80 }, plasticityOverrides: { post2Tau: 50, eta: 2.4 } },
});
export function candidateSettings(name) {
  if (name === 'L0') return {};
  if (!Object.hasOwn(CANDIDATES, name)) throw new Error(`未知 v3 候选 ${name}`);
  return CANDIDATES[name];
}
