/** 第 4 层（脉冲版）验证：预测编码的两层——每层 = 表征群 + 误差群 + 预测中继群（demo/predictive-hierarchy.mjs）。
 *   ① 学习让误差下降；② 熟悉的物体：误差先冲高（上层还没认出），再回到自发水平（被上层的预测解释掉）；
 *   ③ 切断预测抑制或预测下行，误差一直很高；④ 缺一半：缺的那组误差低于自发（该来的没来）；
 *   ⑤ 没见过的组合：解释不掉的部分误差高、被预测而缺的部分低于自发；⑥ 熟悉的输入误差脉冲少；
 *   ⑦ 误差上行：输入换了，上层更完整地改口，下层误差更快消退。序列期待还没做到，只记录数字（见 README"已知缺口"）。
 * 用 cache/ 里训练好的网络，约 10–15 分钟（没有缓存时先训练：平衡层约 9 分钟、第二层约 10 分钟、序列约 11 分钟）。 */
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
  c.note(`尚未做到的序列期待（已知缺口）：Y 出现后头 50 ms 第一层 Y 的误差，经历序列 ${f1(q.afterX)} / ${f1(q.afterBlank)} Hz，只学物体 ${f1(o.afterX)} / ${f1(o.afterBlank)} Hz（X 之后 / 空白之后）`);
  return { ok: c.done(), r };
}

if (import.meta.url === `file:///${process.argv[1].replace(/\\/g, '/')}`) process.exit(verifyL4Predictive().ok ? 0 : 1);
