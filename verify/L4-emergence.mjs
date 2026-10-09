/** 第 4 层（脉冲版）验证：外界刺激下，由真实神经元与突触组成的网络能否
 *   ① 形成特征性激活；② 在反复经历"相似但不同"的刺激后，靠赫布学习涌现以原型为中心的吸引子（能量极小）；
 *   ③ 在相邻刺激组按顺序连续出现后，于吸引子之间长出有向通道。
 * 先验证快速网络实现与第 1–3 层逐个突触的实现逐拍一致，再跑 demo/fig2-network.mjs 的完整实验。约 4 分钟。 */
import { SpikingNetwork } from '../src/L4-network/spiking-network.mjs';
import { SpikingCircuit } from '../src/L3-plasticity/circuit.mjs';
import { GB_DP, GB_NMDA, GB_RATE, pairingChange } from '../src/L3-plasticity/calcium-rule.mjs';
import { mammalianCosts } from '../src/L1-neuron/hh.mjs';
import { runExperiment, CFG } from '../demo/fig2-network.mjs';
import { checker, fmt } from '../src/core/check.mjs';

export function verifyL4Emergence({ quiet = false } = {}) {
  const c = checker('第 4 层（脉冲版）外界刺激下的涌现');
  const costs = mammalianCosts();
  const pairs = [[0, 1], [1, 2], [2, 0], [0, 2]];
  const extra = (t) => [t % 200 < 60 ? 400 : 0, t % 300 < 40 ? 350 : 0, (t + 50) % 250 < 50 ? 380 : 0];
  for (const [name, gb] of [['GB_DP', GB_DP], ['GB_NMDA', GB_NMDA], ['GB_RATE', GB_RATE]]) {
    const plastic = { sMin: 0.1, sMax: 4, rho0: 0.3, accel: 5 };
    const obj = new SpikingCircuit({ costs, seed: 1, neurons: [{}, {}, {}], synapses: pairs.map(([a, b]) => ({ pre: a, post: b, contacts: 10, pRelease: 0.3, tauRec: 100, plastic: { ...plastic, p: gb } })) });
    const fast = new SpikingNetwork({ costs, pops: [{ name: 'E', n: 3 }], plasticity: gb });
    fast.connect({ pairs, contacts: 10, pRelease: 0.3, tauRec: 100, plastic });
    let same = true, steps = 0, firstMismatch = null, maxRhoError = 0, maxAtpRelativeError = 0;
    fast.run(3000, { extra, onStep: () => {
      obj.run(obj.dt, { extra }); steps++;
      const sameSpikes = obj.neurons.every((n, i) => n.spikes.length === fast.spikes[i].length && n.spikes.at(-1) === fast.spikes[i].at(-1));
      const rhoError = Math.max(...obj.syn.map((s, k) => Math.abs(s.pl.rho - fast.groups[0].rho[k])));
      const atpError = Math.abs(obj.ledger.total() - fast.ledger.total()) / obj.ledger.total();
      maxRhoError = Math.max(maxRhoError, rhoError); maxAtpRelativeError = Math.max(maxAtpRelativeError, atpError);
      if (!sameSpikes || rhoError >= 1e-12 || atpError >= 1e-12) { same = false; firstMismatch ??= steps; }
    } });
    c.check(`快速网络与逐个突触的实现逐拍一致（钙规则 ${name}）：放电时刻、突触效能、ATP 全部相同`, same,
      `${steps} 步；${obj.neurons.map((n) => n.spikes.length).join('/')} 次放电；最大 ρ 误差 ${maxRhoError}，ATP 相对误差 ${maxAtpRelativeError}${firstMismatch ? `，首个差异在第 ${firstMismatch} 步` : ''}`);
  }

  const w = (d, p = {}) => pairingChange(d, { p: GB_NMDA, ...p });
  const base = w(-100);
  c.check('网络版钙规则保留经典 STDP 的不对称窗：前在后前增强，后在前前特异性减弱（超出基础水平）', w(5) > 0.02 && w(10) > 0.02 && w(-5) < base - 0.05,
    `+5 ms ${fmt(w(5), 3)}，+10 ms ${fmt(w(10), 3)}，−5 ms ${fmt(w(-5), 3)}，−100 ms（基础）${fmt(base, 3)}`);

  const r = runExperiment(CFG, { figures: true, log: quiet ? () => {} : (s) => console.log(`  ${s}`) });
  const { before: B, after1: A } = r;
  c.check('特征性激活：学习后，同一组的不同变体引起的全网响应更一致，不同组之间更不同', A.resp.within > B.resp.within + 0.15 && A.resp.between < B.resp.between,
    `同组相关 ${fmt(B.resp.within)} → ${fmt(A.resp.within)}，异组 ${fmt(B.resp.between)} → ${fmt(A.resp.between)}`);
  c.check('吸引子（补全）：学习前只有被刺激的神经元放电；学习后只给原型的 8 个神经元，其余成员被拉起', B.cue === 0 && A.cue > 0.3,
    `其余 12 个被拉起 ${fmt(B.cue * 100, 0)}% → ${fmt(A.cue * 100, 0)}%`);
  const e0 = B.energy, e1 = A.energy;
  c.check('能量极小落在从未出现过的原型上：学习后维持原型所需的外部驱动最少，少于亲身经历过的变体与随机模式', e1['原型'].drive < e1['经历过的变体'].drive && e1['原型'].drive < e1['随机'].drive && e1['原型'].drive < e0['原型'].drive - 1,
    `原型 ${fmt(e0['原型'].drive, 1)} → ${fmt(e1['原型'].drive, 1)} Hz，变体 ${fmt(e0['经历过的变体'].drive, 1)} → ${fmt(e1['经历过的变体'].drive, 1)} Hz，随机 ${fmt(e0['随机'].drive, 1)} → ${fmt(e1['随机'].drive, 1)} Hz`);
  c.check('代谢账同序：维持原型的整个系统 ATP 也最低', e1['原型'].atp < e1['经历过的变体'].atp && e1['原型'].atp < e1['随机'].atp,
    `${fmt(e1['原型'].atp)} / ${fmt(e1['经历过的变体'].atp)} / ${fmt(e1['随机'].atp)} ATP/s`);
  const diag = r.B1.map((row, a) => row[a]), off = r.B1.flatMap((row, a) => row.filter((_, b) => b !== a));
  const mean = (x) => x.reduce((s, v) => s + v, 0) / x.length;
  c.check('突触结构从均匀长成组块：组内效能明显高于组间', mean(diag) > 2 * mean(off), `组内 ${fmt(mean(diag))}，组间 ${fmt(mean(off))}（起点全部 ${fmt(r.B0[0][0])}）`);
  const fwd = mean(r.fwd), bwd = mean(r.bwd), fwd1 = mean(r.fwd1), bwd1 = mean(r.bwd1);
  c.check('有向通道（结构）：相邻刺激组按 A→B→C→D 连续出现后，正向组间突触增强、反向减弱', fwd - fwd1 > 0.05 && bwd < bwd1 && fwd > bwd + 0.05,
    `正向 ${fmt(fwd1, 3)} → ${fmt(fwd, 3)}，反向 ${fmt(bwd1, 3)} → ${fmt(bwd, 3)}`);
  const s0 = A.seq, s1 = r.seqAfter;
  c.check('有向通道（功能）：持续刺激一组时，训练后只拉起它的下一组、拉不动上一组；训练前两者都拉不起', s1.fwd > 5 && s1.bwd < 1 && s0.fwd < 4 && s0.bwd < 4,
    `下一组 ${fmt(s0.fwd, 1)} → ${fmt(s1.fwd, 1)} Hz，上一组 ${fmt(s0.bwd, 1)} → ${fmt(s1.bwd, 1)} Hz`);
  return { ok: c.done(), r };
}

if (import.meta.url === `file:///${process.argv[1].replace(/\\/g, '/')}`) process.exit(verifyL4Emergence().ok ? 0 : 1);
