/** 实验：双向误差能否代替"浅"来维持稳定（src/L4-network/layered-network.mjs 的 pcNeg 选项）。
 * 平衡态的上层吸引子很浅：抑制性可塑性按每个神经元的目标频率学，连吸引子自己的递归兴奋也一起抵消掉——
 *   把上层学到的递归放大后再让抑制重新学，上层就回到原样（放大 1.25–2.5 倍，自发 2–2.7 Hz，X 撤掉后 50 ms 回到基线）。
 * 所以这里直接把上层学到的递归放大 G 倍、不再重新平衡，得到一个"深"的上层；没有纠正时它会自己点燃、撤掉输入后重新点燃、新输入来了也不让位。
 * 再加上负误差通路（预测多于证据时放电，经上层同组的抑制性细胞 J 压掉被错误预测的那一组），先学负误差的抑制，
 * 然后在同一个网络里比较"向上的纠正"（J → 上层）完整与切断。
 *
 * 结果（负误差抑制学 60 个物体后；切断 / 完整）：
 *   1.3 倍：X 撤掉后 150–300 ms 上层 X 30.6 / 0.8 Hz，换成 Y 后 X 残留 17.9 / 0.8 Hz、Y 59.1 / 24.2 Hz，
 *     只有 A 时上层凭空补出的 D 21.8 / 3.4 Hz，静息时有组自己亮起来 20.0% / 10.7%（原来的浅上层 5%）
 *   1.4 倍（两次运行）：上层自发 13.4–15.8 / 3.4–3.5 Hz，X 撤掉后重新点燃 68–96 / 0.9–1.2 Hz，换成 Y 后 X 残留 59–69 / 1.9–2.3 Hz、
 *     Y 142–148 / 35–38 Hz，凭空补出的 D 11–27 / 6 Hz；静息时自己亮起来 29–32% / 17–24%（J → 上层加强到 5 时 11%）——
 *     静息时的自发点燃只压下一部分，是最不稳的一项
 *   纠正回路要比吸引子点燃得快：预测 → 负误差只用 NMDA 时（negKind 'nmda'），1.4 倍静息时仍 32.7%，换成 Y 后 Y 冲到 122 Hz。
 *   代价：同样的纠正放在原来的浅上层上偏强（熟悉 X 的正误差 3.6 → 8.6 Hz，那时用的是只有 NMDA 的回路）——纠正的强度要与深度相配。
 * 用法：G=1.4 node demo/negative-error.mjs（第一次先学负误差的抑制约 7 分钟，结果缓存到 cache/neg-G<G>*.json；测试约 10 分钟）；
 *       MODE=cal 看负误差抑制未学习时的标定；NEG='{"sJE":5}' 覆盖负误差参数 */
import fs from 'fs';
import { PC2, OBJ, grp, sequenceCourse, timeCourse } from './predictive-hierarchy.mjs';
import { STIM, protoB } from './balanced-layer.mjs';
import { buildLayered, loadState, learnOnly, inputRate, spikesIn, saveState } from '../src/L4-network/layered-network.mjs';
import { mean } from './lib-hierarchy.mjs';
import { makeRng } from '../src/core/rng.mjs';

const G = Number(process.env.G ?? 1.4), NTR = Number(process.env.NTR ?? 60), NEG = process.env.NEG ?? '{}', log = console.log;
const S = buildLayered(2, { ...PC2, pcNeg: true, pcOver: { ...PC2.pcOver, ...JSON.parse(NEG) } }), [L1, L2] = S.layers;
const deepen = () => { for (let k = 0; k < L2.EE.scale.length; k++) L2.EE.scale[k] *= G; };
loadState(S, JSON.parse(fs.readFileSync('cache/pc-2.json', 'utf8')).state); learnOnly(S, []); deepen();
const f = (v) => v.toFixed(1).padStart(5), rateOf = (ids, t0, t1) => mean(spikesIn(S.net, ids, t0, t1).map((s) => s.length)) / ((t1 - t0) / 1000);
const pr = (tc, key, parts) => [...tc[`${key}${parts[0]}`]].map((v, i) => (v + tc[`${key}${parts[1]}`][i]) / 2);
const seg = (arr, a, b, bin = 10) => mean(arr.slice(a / bin, b / bin));
const cutJE = (fn) => { const keep = L2.JE.scale.slice(); L2.JE.scale.fill(0); try { return fn(); } finally { L2.JE.scale.set(keep); } };

function measure(label) {
  S.net.run(500, { rate: inputRate(S), learn: false });
  let t0 = S.net.t; S.net.run(2000, { rate: inputRate(S), learn: false });
  const sp = Object.fromEntries([['E1', L1.E], ['N1', L1.N], ['M1', L1.M], ['V1', L1.V], ['E2', L2.E], ['I2', L2.I], ['J2', L2.J], ['S1', L1.S]].map(([k, ids]) => [k, rateOf(ids, t0, t0 + 2000)]));
  // 静息时上层各组：每 20 ms 一箱，任一组超过 10 Hz 的箱的比例（"自己亮起来"）
  t0 = S.net.t; const ms = 3000; S.net.run(ms, { rate: inputRate(S), learn: false });
  const groups = 'ABCD'.split('').map((p) => grp(S, 1, 'E', p));
  let hot = 0; for (let b = 0; b < ms / 20; b++) if (groups.some((ids) => rateOf(ids, t0 + b * 20, t0 + (b + 1) * 20) > 10)) hot++;
  const xo = sequenceCourse(S, [['AD', 150], ['', 350]], { reps: 8 }), xy = sequenceCourse(S, [['AD', 150], ['CB', 150], ['', 200]], { reps: 8 });
  const fx = timeCourse(S, OBJ.X, { reps: 6 }), oa = timeCourse(S, ['A'], { reps: 6 });
  const X = pr(xo, 'E2', 'AD'), MX = pr(xo, 'M1', 'AD'), Xs = pr(xy, 'E2', 'AD'), Ys = pr(xy, 'E2', 'CB'), NY = pr(xy, 'N1', 'CB'), MXs = pr(xy, 'M1', 'AD');
  const r = {
    sp, hot: (100 * hot) / (ms / 20), xDuring: seg(X, 50, 150), lin: [seg(X, 150, 200), seg(X, 200, 300), seg(X, 300, 450)], mxDuring: seg(MX, 50, 150), mxAfter: seg(MX, 150, 200),
    swX: seg(Xs, 175, 225), swY: seg(Ys, 175, 225), swNY: seg(NY, 150, 200), swMX: seg(MXs, 160, 210),
    famN: mean([...pr(fx, 'N1', 'AD')].slice(2)), famM: mean([...pr(fx, 'M1', 'AD')].slice(2)), famE: mean([...pr(fx, 'E2', 'AD')].slice(2)),
    missM: mean([...oa.M1D].slice(2)), missN: mean([...oa.N1D].slice(2)), missE: mean([...oa.E2D].slice(2)), onlyAE: mean([...oa.E2A].slice(2)),
  };
  log(`  【${label}】自发：E1 ${f(sp.E1)} N1 ${f(sp.N1)} M1 ${f(sp.M1)} V1 ${f(sp.V1)} S1 ${f(sp.S1)} | E2 ${f(sp.E2)} I2 ${f(sp.I2)} J2 ${f(sp.J2)} Hz；静息时上层有组自己亮起来 ${r.hot.toFixed(1)}%`);
  log(`    X 150 ms 后撤掉：上层 X 期间 ${f(r.xDuring)}，撤掉后 0–50 / 50–150 / 150–300 ms ${f(r.lin[0])}${f(r.lin[1])}${f(r.lin[2])} Hz；负误差 X 期间 ${f(r.mxDuring)}、撤掉后 0–50 ms ${f(r.mxAfter)} Hz`);
  log(`    X 换成 Y（换后 25–75 ms）：上层 X 残留 ${f(r.swX)}、Y ${f(r.swY)} Hz；换后头 50 ms Y 的正误差 ${f(r.swNY)}、X 的负误差（10–60 ms）${f(r.swMX)} Hz`);
  log(`    熟悉的 X（50–200 ms）：正误差 ${f(r.famN)}、负误差 ${f(r.famM)}、上层 X ${f(r.famE)} Hz；只有 A：缺失的 D 负误差 ${f(r.missM)}、正误差 ${f(r.missN)}、上层 D ${f(r.missE)}、上层 A ${f(r.onlyAE)} Hz`);
  return r;
}

log(`上层递归放大 ${G} 倍，负误差参数 ${NEG}`);
if (process.env.MODE === 'cal') {
  measure('负误差抑制未学习，纠正完整');
  cutJE(() => measure('负误差抑制未学习，切断纠正'));
} else {
  const file = `cache/neg-G${G}${NEG === '{}' ? '' : '-' + NEG.replace(/[^a-zA-Z0-9.]+/g, '')}.json`;
  if (fs.existsSync(file)) { loadState(S, JSON.parse(fs.readFileSync(file, 'utf8'))); deepen(); }
  else {
    const rng = makeRng(11), t0 = Date.now();
    learnOnly(S, [L1.NEGI, L2.NEGI]);
    S.net.run(500, { rate: inputRate(S), learn: false });
    for (let n = 0; n < NTR; n++) {
      const parts = OBJ[rng.pick(2) ? 'Y' : 'X'];
      S.net.run(200, { rate: inputRate(S, new Map(parts.flatMap((p) => rng.shuffle(protoB(p).slice()).slice(0, 48)).map((a) => [a, STIM]))), learn: true });
      S.net.run(300, { rate: inputRate(S), learn: true });
      if (n % 20 === 19) log(`  负误差学习 ${n + 1}/${NTR}（${((Date.now() - t0) / 1000).toFixed(0)} s）`);
    }
    learnOnly(S, []);
    fs.writeFileSync(file, JSON.stringify(saveState(S)));
  }
  measure('纠正完整');
  cutJE(() => measure('切断纠正（负误差不往上作用）'));
}
