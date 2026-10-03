/** 第 4 层验证：树突承载预测所用的可塑性（spiking-network.mjs 的平台规则）。
 * 20 个突触前 → 1 个带顶树突的突触后，突触落在顶树突、按"平台 × 资格迹"学习（协方差形式）：
 *   ① 行为时间尺度：突触前比突触后的爆发早几百毫秒也能写进去，早 3 秒就几乎写不进去；
 *   ② 门控：突触前不放电、回忆态、突触后没有平台（放电间隔 8 ms 以上）都不学；平台只由高频爆发（间隔约 4 ms）引出——
 *      对应 Larkum 的临界频率；
 *   ③ 协方差：突触前只有自发活动时不变，自发背景上叠加一次先后配对仍然增强；
 *   ④ 编码 / 提取切换：学习态里不减弱这组突触的传递时，已经够强的顶树突输入会自己引出平台、把自己越写越强；
 *      按 Hasselmo 减弱学习态传递后不再自我强化。约 10 秒。 */
import { SpikingNetwork } from '../src/L4-network/spiking-network.mjs';
import { mammalianCosts } from '../src/L1-neuron/hh.mjs';
import { GB_NMDA } from '../src/L3-plasticity/calcium-rule.mjs';
import { checker, fmt } from '../src/core/check.mjs';

/** 预热 5 s（滑动平均适应突触前的自发水平）→ 突触前 150 ms → 空 gap → 突触后强驱动 150 ms → 300 ms */
function trial({ gap = 300, postHz = 150, preOn = true, mod = 1, modI = 0, bgPre = 0, modSup = 0.8, rho0 = 0.3 } = {}) {
  const net = new SpikingNetwork({ costs: mammalianCosts(), seed: 3, plasticity: GB_NMDA, pops: [{ name: 'PRE', n: 20 }, { name: 'POST', n: 1, p: { b: 20, tauW: 300 }, apical: modI ? { modI } : true }] });
  const g = net.connect({ from: 'PRE', to: 'POST', p: 1, contacts: 10, pRelease: 0.3, tauRec: 100, target: 'apical', modSup, plastic: { rule: 'plateau', sMin: 0, sMax: 6, rho0, cov: true } });
  net.mod = mod;
  const post = net.idx('POST')[0], mean = () => g.rho.reduce((s, v) => s + v, 0) / g.rho.length, r0 = mean();
  const phase = (ms, preHz, pHz) => { let plat = 0; net.run(ms, { rate: (i) => (i === post ? pHz : preHz), onStep: () => { if (net.plat[post] > g.plastic.thetaPl) plat += net.dt; } }); return plat; };
  phase(5000, bgPre, 0); phase(150, preOn ? 40 : bgPre, 0); phase(gap, bgPre, 0);
  const t0 = net.t, plat = phase(150, bgPre, postHz);
  phase(300, bgPre, 0);
  const sp = net.spikes[post].filter((t) => t >= t0 && t < t0 + 150), isi = sp.slice(1).map((t, k) => t - sp[k]);
  return { d: mean() - r0, plat, n: sp.length, minIsi: isi.length ? Math.min(...isi) : Infinity };
}

export function verifyL4Dendritic() {
  const c = checker('第 4 层 树突承载预测的可塑性（平台 × 资格迹）');
  const near = trial(), mid = trial({ gap: 1000 }), far = trial({ gap: 3000 });
  c.check('行为时间尺度：突触前早 300 ms / 1 s 仍写进去，早 3 s 几乎写不进去', near.d > 0.15 && mid.d > 0.08 && Math.abs(far.d) < near.d / 4,
    `Δρ ${fmt(near.d, 3)} / ${fmt(mid.d, 3)} / ${fmt(far.d, 3)}`);
  const noPre = trial({ preOn: false }), recall = trial({ mod: 0 }), noPlat = trial({ postHz: 60 });
  c.check('门控：突触前不放电、回忆态、突触后没有平台都不学', [noPre, recall, noPlat].every((r) => Math.abs(r.d) < 0.005),
    `Δρ ${fmt(noPre.d, 3)} / ${fmt(recall.d, 3)} / ${fmt(noPlat.d, 3)}`);
  c.check('平台只由高频爆发引出（Larkum 的临界频率）', near.plat > 20 && near.minIsi < 5 && noPlat.plat === 0 && noPlat.minIsi > 7,
    `爆发：最短间隔 ${fmt(near.minIsi, 1)} ms、平台 ${fmt(near.plat, 1)} ms；${noPlat.n} 次放电、最短间隔 ${fmt(noPlat.minIsi, 1)} ms 时平台 ${fmt(noPlat.plat, 1)} ms`);
  const spont = trial({ preOn: false, bgPre: 2 }), paired = trial({ bgPre: 2 });
  c.check('协方差形式：突触前只有自发活动时不变，自发背景上的先后配对仍然增强', Math.abs(spont.d) < 0.01 && paired.d > 0.15,
    `Δρ ${fmt(spont.d, 3)} / ${fmt(paired.d, 3)}`);
  const selfOn = trial({ postHz: 30, modI: 200, modSup: 0 }), selfOff = trial({ postHz: 30, modI: 200, modSup: 0.8 });
  c.check('编码 / 提取切换：不减弱学习态传递时顶树突输入自我强化，减弱后不再自我强化', selfOn.d > 0.2 && Math.abs(selfOff.d) < 0.02,
    `Δρ ${fmt(selfOn.d, 3)}（不减弱）/ ${fmt(selfOff.d, 3)}（减弱 80%）`);
  return { ok: c.done() };
}

if (import.meta.url === `file:///${process.argv[1].replace(/\\/g, '/')}`) process.exit(verifyL4Dendritic().ok ? 0 : 1);
