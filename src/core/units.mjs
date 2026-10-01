/** 物理常数与单位换算。
 * 约定：HH 模型按单位面积（µA/cm²、µF/cm²、mS/cm²、mV、ms）；点神经元与突触用（pF、nS、mV、ms、pA）。 */

export const E_CHARGE = 1.602176634e-19; // C
export const UM2_TO_CM2 = 1e-8;

/** 钠钾泵：每个 ATP 泵出 3 个 Na⁺、泵入 2 个 K⁺ */
export const ATP_PER_NA = 1 / 3;
/** 质膜钙泵（PMCA）：每个 ATP 泵出 1 个 Ca²⁺ */
export const ATP_PER_CA = 1;
/** Cl⁻ 经 KCC2 与 K⁺ 共转运排出，K⁺ 再由钠钾泵补回（2 个 K⁺ / ATP） */
export const ATP_PER_CL = 1 / 2;

/** µA/cm² × ms = nC/cm² → 每 cm² 的离子数 */
export const ionsFromNanoCoulomb = (nC) => (nC * 1e-9) / E_CHARGE;
/** pA × ms = fC → 离子数 */
export const ionsFromFemtoCoulomb = (fC) => (fC * 1e-15) / E_CHARGE;
