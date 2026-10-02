/** 第 4 层（脉冲版）验证：多层网络里前馈 ↑、层内有向通道 →、反馈 ↓ 都是真实突触，每层都是完整的脉冲网络。
 *   ① 第一层学到物体内部的有向通道，第二层学到物体级吸引子与物体级有向通道；
 *   ② 第二层的预测：只给 X，第二层 Y 的组被提前拉起（与"上层没经历物体序列"相比）；
 *   ③ 反馈：上层把下层补全成完整的物体（下层不变性升高、部件补全增强），切断反馈即消失；"想象"时只驱动上层也能在第一层重建 X；
 *   ④ 层数：不变性 2 层起饱和，能耗随层数增加。
 * 已知不成立的一环（打印为说明）：上层预测的 Y 传不到正在感知 X 的第一层——X 的爆发经全局抑制把第一层其余神经元压低十几 mV。
 * 读 cache/hier-*.json（没有缓存时先逐层训练约 20 分钟），测试约 6 分钟。 */
import { runHierarchy } from '../demo/fig10-hierarchy.mjs';
import { checker, fmt } from '../src/core/check.mjs';

export function verifyL4Hierarchy({ quiet = false } = {}) {
  const c = checker('第 4 层（脉冲版）多层网络：前馈、有向通道、反馈');
  const { structure: st, results } = runHierarchy({ log: quiet ? () => {} : (s) => console.log(`  ${s}`), figures: true });
  const [r1, r2, r3, r4, rCut, rNo] = results;
  c.check('第一层学到物体内部的有向通道（A→D、C→B 强于反向）', st[0].innerFwd > 1.5 * st[0].innerBwd, `正向 ${fmt(st[0].innerFwd)}，反向 ${fmt(st[0].innerBwd)}`);
  c.check('第二层学到物体级吸引子（同一物体的两个组互相强连接）与物体级有向通道（X→Y 强于 Y→X）', st[1].within > 0.3 && st[1].fwd > 2 * st[1].bwd,
    `物体内 ${fmt(st[1].within)}，X→Y ${fmt(st[1].fwd)}，Y→X ${fmt(st[1].bwd)}；第一层跨物体 ${fmt(st[0].fwd)} / ${fmt(st[0].bwd)}`);
  c.check('第二层的预测：只给 X 时第二层 Y 的组被提前拉起，没经历物体序列的第二层不会', r2.pred[1] > rNo.pred[1] + 1, `${fmt(r2.pred[1], 1)} / ${fmt(rNo.pred[1], 1)} Hz`);
  c.check('反馈让第一层的表示更像物体：同一物体两种不重叠刺激的响应更相关，切断反馈回到单层水平', r2.invariance[0] > r1.invariance[0] + 0.2 && Math.abs(rCut.invariance[0] - r1.invariance[0]) < 0.1,
    `单层 ${fmt(r1.invariance[0])}，两层 ${fmt(r2.invariance[0])}，切断反馈 ${fmt(rCut.invariance[0])}`);
  c.check('反馈增强第一层的部件补全，切断反馈回到单层水平', r2.complete[0] > r1.complete[0] + 1.5 && Math.abs(rCut.complete[0] - r1.complete[0]) < 1,
    `单层 ${fmt(r1.complete[0], 1)}，两层 ${fmt(r2.complete[0], 1)}，切断反馈 ${fmt(rCut.complete[0], 1)} Hz`);
  c.check('想象：第一层没有刺激、只驱动第二层 X 的组，反馈在第一层重建出 X；切断反馈则没有', r2.imagery.x1 > 5 && rCut.imagery.x1 < 1,
    `第一层 X 的组 ${fmt(r2.imagery.x1, 1)} Hz（切断反馈 ${fmt(rCut.imagery.x1, 1)} Hz）`);
  c.check('层数：最上层的不变性 2 层起饱和；整个网络的能耗随层数单调增加', [r2, r3, r4].every((r) => r.invariance[r.K - 1] > 0.95) && r1.atp < r2.atp && r2.atp < r3.atp && r3.atp < r4.atp,
    `不变性 ${[r1, r2, r3, r4].map((r) => fmt(r.invariance[r.K - 1])).join(' / ')}；ATP ${[r1, r2, r3, r4].map((r) => fmt(r.atp / 1e9, 0)).join(' / ')} ×10⁹`);
  c.note(`没有成立的一环：感知 X 时第一层 Y 的组膜电位比静息低 ${fmt(-r2.vMean)} mV（没经历序列 ${fmt(-rNo.vMean)} mV），预测带来的差别只有 ${fmt(r2.vMean - rNo.vMean)} mV；想象时 ${fmt(r2.imagery.vY - rNo.imagery.vY)} mV`);
  c.note(`第 3、4 层没有学到正向的物体级通道：X→Y ${fmt(st[2].fwd)} / ${fmt(st[3].fwd)}，Y→X ${fmt(st[2].bwd)} / ${fmt(st[3].bwd)}`);
  return { ok: c.done() };
}

if (import.meta.url === `file:///${process.argv[1].replace(/\\/g, '/')}`) process.exit(verifyL4Hierarchy().ok ? 0 : 1);
