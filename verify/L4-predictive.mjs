/** 第 4 层（脉冲版）验证：预测编码的两层——每层 = 表征群 + 误差群 + 预测中继群（demo/predictive-hierarchy.mjs）。
 *   ① 学习让误差下降；② 熟悉的物体：误差先冲高（上层还没认出），再回到自发水平（被上层的预测解释掉）；
 *   ③ 切断预测抑制或预测下行，误差一直很高；④ 缺一半：缺的那组误差低于自发（该来的没来）；
 *   ⑤ 没见过的组合：解释不掉的部分误差高、被预测而缺的部分低于自发；⑥ 熟悉的输入误差脉冲少；
 *   ⑦ 误差上行：输入换了，上层更完整地改口，下层误差更快消退；
 *   ⑧ 学到"X 之后是 Y"（X 的中继 → Y 的误差细胞的抑制变强）；⑨ 期待在该出现的时刻表达出来（学过的先后比没学过的反向误差小）；
 *   ⑩ 该来的没来：X 之后 Y 不来，Y 的误差细胞低于自发。
 * 用 cache/ 里训练好的网络，约 25 分钟（没有缓存时先训练：平衡层约 9 分钟、第二层约 12 分钟、序列约 15 分钟）。 */
import { runPredictive } from '../demo/predictive-hierarchy.mjs';
import { checker, fmt } from '../src/core/check.mjs';

export function verifyL4Predictive({ quiet = false } = {}) {
  const c = checker('第 4 层（脉冲版）预测编码：表征群 + 误差群');
  const r = runPredictive({ log: quiet ? () => {} : (s) => console.log(s), figure: false });
  const e = r.obj.err, t = r.obj.total, sw = r.obj.sw, f1 = (x) => fmt(x, 1);
  c.check('学习让误差下降：X 引起的第一层误差降到一半以下', r.curve.at(-1) < 0.5 * r.curve[0], `${r.curve.map(f1).join(' → ')} Hz`);
  c.check('熟悉的物体被解释掉：误差先冲高，之后回到自发水平附近', e.onset > 2 * e.fam && e.fam < e.base + 3,
    `头 25 ms ${f1(e.onset)} Hz，50–200 ms ${f1(e.fam)} Hz，自发 ${f1(e.base)} Hz`);
  c.check('解释来自上层的预测：切断预测抑制或预测下行，误差一直很高', e.cutP > 4 * e.fam && e.cutT > 4 * e.fam,
    `切断预测抑制 ${f1(e.cutP)}、切断预测下行 ${f1(e.cutT)} Hz`);
  c.check('缺一半：被预测而没出现的 D 组误差低于自发', e.missD < 0.6 * e.base, `D 组 ${f1(e.missD)} Hz，自发 ${f1(e.base)} Hz`);
  c.check('没见过的组合 A + C：解释不掉的 A、C 误差高，被预测而缺的 B、D 低于自发', e.novelAC > 1.5 * e.fam && e.novelBD < 0.6 * e.base,
    `A、C ${f1(e.novelAC)} Hz（熟悉 X ${f1(e.fam)}），B、D ${f1(e.novelBD)} Hz`);
  c.check('熟悉的输入误差脉冲少：不到切断预测时的三成', t.fam < 0.3 * t.cutP, `四组合计 ${f1(t.fam)} / ${f1(t.cutP)} Hz`);
  c.check('误差上行：输入换成 Y 后上层 Y 的组更强，下层 Y 的误差更快消退', sw.on.y > sw.off.y && sw.on.n < 0.6 * sw.off.n,
    `上层 Y 的组 ${f1(sw.on.y)} / ${f1(sw.off.y)} Hz，换后头 50 ms 下层 Y 的误差 ${f1(sw.on.n)} / ${f1(sw.off.n)} Hz（完整 / 切断误差上行）`);
  const q = r.seq.seq, o = r.seq.obj;
  c.check('学到"X 之后是 Y"：X 的中继 → Y 的误差细胞的抑制经历序列后至少翻倍，反向（Y 的中继 → X 的误差）不增加', q.w > 2 * o.w && q.wYX <= o.wYX,
    `X → Y ${fmt(o.w)} → ${fmt(q.w)}，Y → X ${fmt(o.wYX)} → ${fmt(q.wYX)}（只学物体 → 经历序列）`);
  c.check('期待在该出现的时刻表达出来：学过的先后（X → Y）后一项的误差比没学过的反向（Y → X）小，减去只学物体的网络里的同一个差仍大于 4 Hz', r.seq.expect > 4,
    `后一项出现后头 50 ms：经历序列 ${f1(q.fwd)} / ${f1(q.rev)} Hz，只学物体 ${f1(o.fwd)} / ${f1(o.rev)} Hz（X → Y / Y → X），学到的期待 ${f1(r.seq.expect)} Hz`);
  c.check('该来的没来：经历序列后，X 之后 Y 不来，Y 的误差细胞在头 50 ms 低于自发的三成；只学物体的网络没有', q.omitN < 0.3 * q.base && o.omitN > 0.6 * o.base,
    `经历序列 ${f1(q.omitN)} Hz（自发 ${f1(q.base)}），只学物体 ${f1(o.omitN)} Hz（自发 ${f1(o.base)}）`);
  c.note(`期待只是把 Y 的误差推迟、压低，没有消掉：10 ms 分箱的峰值 ${f1(q.fwdPeak)} / ${f1(q.revPeak)} Hz（X → Y / Y → X）；X 的最后 50 ms 上层 Y 的组 ${f1(q.preY)} Hz——上层并不提前转到 Y`);
  return { ok: c.done(), r };
}

if (import.meta.url === `file:///${process.argv[1].replace(/\\/g, '/')}`) process.exit(verifyL4Predictive().ok ? 0 : 1);
