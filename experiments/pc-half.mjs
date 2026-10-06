// "只做到一半"：后一项出现前后每 5 ms 各群的频率——经历序列的网络里 X → Y（学过）与 Y → X（没学过），以及只学物体的网络里的 X → Y
import { pcStack, sequenceCourse } from '../demo/predictive-hierarchy.mjs';

const log = console.log, row = (a) => [...a].map((v) => v.toFixed(1).padStart(5)).join('');
const cases = [
  ['经历序列，X → Y（学过）', pcStack({ seq: true }), [['AD', 150], ['CB', 150], ['', 200]], 'AD', 'CB'],
  ['经历序列，Y → X（没学过）', null, [['CB', 150], ['AD', 150], ['', 200]], 'CB', 'AD'],
  ['只学物体，X → Y', pcStack(), [['AD', 150], ['CB', 150], ['', 200]], 'AD', 'CB'],
];
cases[1][1] = cases[0][1];
log('后一项出现前 20 ms 到后 80 ms，每 5 ms（第 5 列 = 后一项的头 5 ms）；"前" = 前一项，"后" = 后一项');
for (const [name, S, segs, a, b] of cases) {
  const tc = sequenceCourse(S, segs, { bin: 5, reps: 24 });
  const pr = (pop, k, parts) => [...tc[`${pop}${k}${parts[0]}`]].map((v, i) => (v + tc[`${pop}${k}${parts[1]}`][i]) / 2).slice(26, 46);
  log(`\n【${name}】`);
  log(`  上层 前一项的组        ${row(pr('E', 2, a))}`);
  log(`  上层 后一项的组        ${row(pr('E', 2, b))}`);
  log(`  前一项的中继           ${row(pr('S', 1, a))}`);
  log(`  后一项的中继           ${row(pr('S', 1, b))}`);
  log(`  第一层 后一项的表征    ${row(pr('E', 1, b))}`);
  log(`  第一层 后一项的误差    ${row(pr('N', 1, b))}`);
}
