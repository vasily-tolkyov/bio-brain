/** 第 4 层（脉冲版）验证：两层迭代激活。第一层 = 训练好的网络（吸引子 A B C D + 有向通道），
 * 第一层各吸引子的激活度按强度画成第二层的刺激；第二层是全新的同构网络，只经历物体 X = A + C、Y = B + D。
 *   ① 第二层只靠赫布学习长出"吸引子的组合"这一级吸引子；② 只给物体的一个部件，第二层补全另一个部件；
 *   ③ 物体之间竞争：第一层经有向通道带起的别家部件，在第二层被压下；没学过的组合被解释成一个学过的物体；
 *   ④ 第一层内部的细节在"绘图"这一步被丢弃：同一吸引子的两种不重叠的刺激，越往上越像。
 * 约 2 分钟（第一层训练结果缓存在 cache/，没有缓存时先训练约 4 分钟）。 */
import { runTwoLayer } from '../demo/fig8-two-layer.mjs';
import { checker, fmt } from '../src/core/check.mjs';

export function verifyL4TwoLayer({ quiet = false } = {}) {
  const c = checker('第 4 层（脉冲版）两层迭代激活');
  const r = runTwoLayer({ log: quiet ? () => {} : (s) => console.log(`  ${s}`) });
  const ix = (L) => r.names.indexOf(L), A = ix('A'), B = ix('B'), C = ix('C'), D = ix('D');
  const X2 = r.comp.find((k) => k[A] >= 15 && k[C] >= 8 && k[B] + k[D] <= 4), Y2 = r.comp.find((k) => k[B] >= 15 && k[D] >= 15 && k[A] <= 4);
  c.check('第二层只靠赫布学习长出物体级吸引子：一个由 A、C 两组组成，一个由 B、D 两组组成', X2 && Y2,
    r.comp.map((k) => k.map((n, a) => (n ? `${r.names[a]} ${n}` : '')).filter(Boolean).join(' + ')).join('；'));
  const row = (short) => r.rows.find((x) => x.short === short);
  const fill = [['ONLY A', C], ['ONLY C', A], ['ONLY B', D], ['ONLY D', B]].map(([s, a]) => ({ s, a, t: row(s).rt[a], n: row(s).rn[a] }));
  c.check('跨层补全：只给物体的一个部件，第二层把另一个部件拉起（没学过的第二层不会）', fill.every((f) => f.t > f.n + 2),
    fill.map((f) => `${f.s.replace('ONLY ', '只给 ')} → ${r.names[f.a]} 组 ${fmt(f.t, 1)} / ${fmt(f.n, 1)} Hz`).join('，'));
  const oa = row('ONLY A');
  c.check('物体之间竞争：只给 A 时第一层经有向通道带起了 B，第二层的 B 组被物体 X 压下', oa.lv[B] > 0.2 && oa.rt[B] < 0.5 * oa.rn[B],
    `第一层 B 激活度 ${fmt(oa.lv[B])}，第二层 B 组 ${fmt(oa.rt[B], 1)} / ${fmt(oa.rn[B], 1)} Hz（学过 / 没学过）`);
  const ad = row('NEW A+D');
  c.check('没学过的组合 A + D 被解释成物体 X：补出 C，压下 D', ad.rt[C] > ad.rn[C] + 3 && ad.rt[D] < ad.rn[D],
    `C 组 ${fmt(ad.rt[C], 1)} / ${fmt(ad.rn[C], 1)} Hz，D 组 ${fmt(ad.rt[D], 1)} / ${fmt(ad.rn[D], 1)} Hz`);
  const d = r.detail;
  c.check('细节在"绘图"一步被丢弃：同一吸引子的两种不重叠的刺激越往上越像', d.L1 > d.stim + 0.5 && d.drawn > 0.95 && d.L2 > d.L1,
    `刺激 ${fmt(d.stim)} → 第一层 ${fmt(d.L1)} → 画出的激活度 ${fmt(d.drawn)} → 第二层 ${fmt(d.L2)}`);
  return { ok: c.done(), r };
}

if (import.meta.url === `file:///${process.argv[1].replace(/\\/g, '/')}`) process.exit(verifyL4TwoLayer().ok ? 0 : 1);
