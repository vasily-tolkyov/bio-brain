// 把上层学到的递归放大 G 倍（加深上层的吸引子），看不加纠正时上层是否滞留、是否自己乱跳
// 用法：G=2 node scratch/deep-scan.mjs
import { pcStack, sequenceCourse, spontaneous, grp, OBJ } from '../demo/predictive-hierarchy.mjs';
import { STIM, protoB } from '../demo/balanced-layer.mjs';
import { inputRate, spikesIn, learnOnly } from '../src/L4-network/layered-network.mjs';
import { mean } from '../demo/lib-hierarchy.mjs';
import { makeRng } from '../src/core/rng.mjs';

const G = Number(process.env.G ?? 1), NREB = Number(process.env.NREB ?? 0), log = console.log, f = (v) => v.toFixed(1).padStart(5);
const S = pcStack(), [, L2] = S.layers;
for (let k = 0; k < L2.EE.scale.length; k++) L2.EE.scale[k] *= G;
// 稳态再平衡：只让上层的抑制性可塑性学习，经历 NREB 个物体（各 200 ms + 休息 300 ms），把上层拉回目标频率
if (NREB) {
  const rng = makeRng(5), ie0 = mean([...L2.IE.scale]);
  learnOnly(S, [L2.IE]);
  for (let n = 0; n < NREB; n++) {
    const parts = OBJ[rng.pick(2) ? 'Y' : 'X'];
    S.net.run(200, { rate: inputRate(S, new Map(parts.flatMap((p) => rng.shuffle(protoB(p).slice()).slice(0, 48)).map((a) => [a, STIM]))), learn: true });
    S.net.run(300, { rate: inputRate(S), learn: true });
  }
  learnOnly(S, []);
  log(`  再平衡 ${NREB} 个物体：上层抑制平均强度 ${ie0.toFixed(2)} → ${mean([...L2.IE.scale]).toFixed(2)}`);
}
const pr = (tc, pop, k, parts) => [...tc[`${pop}${k}${parts[0]}`]].map((v, i) => (v + tc[`${pop}${k}${parts[1]}`][i]) / 2);
const seg = (arr, a, b) => mean(arr.slice(a / 10, b / 10));
const sp = spontaneous(S, 2000);
// 静息时上层各组：每 20 ms 一箱，任一组超过 10 Hz 的箱的比例（"自己亮起来"）
S.net.run(300, { rate: inputRate(S), learn: false });
const t0 = S.net.t, ms = 3000; S.net.run(ms, { rate: inputRate(S), learn: false });
let hot = 0, nb = ms / 20;
const groups = 'ABCD'.split('').map((p) => grp(S, 1, 'E', p));
for (let b = 0; b < nb; b++) if (groups.some((ids) => mean(spikesIn(S.net, ids, t0 + b * 20, t0 + (b + 1) * 20).map((s) => s.length)) / 0.02 > 10)) hot++;
// X 150 ms 后撤掉：上层 X 的组在撤掉后 0–50、50–150、150–300 ms；X 直接换成 Y：换后 25–75 ms 上层 X 的残留
const xo = sequenceCourse(S, [['AD', 150], ['', 350]], { reps: 8 }), xy = sequenceCourse(S, [['AD', 150], ['CB', 150], ['', 200]], { reps: 8 });
const X = pr(xo, 'E', 2, 'AD'), Xs = pr(xy, 'E', 2, 'AD'), Ys = pr(xy, 'E', 2, 'CB'), N1X = pr(xo, 'N', 1, 'AD');
log(`G ${G}：上层自发 ${sp[1].E.toFixed(2)} Hz（抑制 ${sp[1].I.toFixed(1)}），静息时有组超过 10 Hz 的时间 ${(100 * hot / nb).toFixed(1)}%；`
  + `X 期间上层 X ${f(seg(X, 50, 150))}，撤掉后 0–50 / 50–150 / 150–300 ms：${f(seg(X, 150, 200))}${f(seg(X, 200, 300))}${f(seg(X, 300, 450))} Hz；`
  + `换成 Y 后 25–75 ms 上层 X 残留 ${f(seg(Xs, 175, 225))}、Y ${f(seg(Ys, 175, 225))} Hz；撤掉后 0–50 ms 第一层 X 的正误差 ${f(seg(N1X, 150, 200))} Hz`);
