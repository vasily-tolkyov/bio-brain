// "想象"：第一层没有任何刺激，直接驱动第二层 X 的组 40 ms——反馈能不能把 X（重建）和预测的 Y 传到第一层
import { cfg, proto, OBJ, mean, trainedStack } from '../demo/lib-hierarchy.mjs';
import { inputRate, spikesIn } from '../src/L4-network/layered-network.mjs';

const BL = ['A', 'B', 'C', 'D'], XB = OBJ.X.map((p) => BL.indexOf(p)), YB = OBJ.Y.map((p) => BL.indexOf(p));
const blocks = BL.map(proto);
function probe(S, name, reps = 16) {
  const L2X = XB.flatMap((q) => blocks[q].map((a) => S.layers[1].E[a]));
  const extra = new Map(L2X.map((i) => [i, cfg.seqStim]));
  const out = S.layers.map(() => blocks.map(() => new Float64Array(24))), vY = new Float64Array(120), vX = new Float64Array(120);
  const ids = (q) => blocks[q].map((a) => S.layers[0].E[a]);
  const yIds = YB.flatMap(ids), xIds = XB.flatMap(ids);
  for (let r = 0; r < reps; r++) {
    S.net.run(400, { rate: inputRate(S), learn: false });
    const ts = S.net.t;
    let step = 0;
    const onStep = () => { if (step++ % 10) return; const ms = step / 10 | 0; if (ms < 120) { vY[ms] += mean(yIds.map((i) => S.net.V[i])) / reps; vX[ms] += mean(xIds.map((i) => S.net.V[i])) / reps; } };
    S.net.run(40, { rate: inputRate(S, new Map(), extra), learn: false, onStep });
    S.net.run(80, { rate: inputRate(S), learn: false, onStep });
    S.layers.forEach((L, k) => blocks.forEach((b, q) => spikesIn(S.net, b.map((a) => L.E[a]), ts, ts + 120).forEach((s) => s.forEach((t) => (out[k][q][Math.floor((t - ts) / 5)] += 1 / (20 * 0.005) / reps)))));
  }
  const g = (k, qs, a, b) => mean(qs.map((q) => mean(Array.from(out[k][q]).slice(a, b))));
  const v0 = mean(Array.from(vY).slice(0, 3)), vx0 = mean(Array.from(vX).slice(0, 3));
  console.log(`【${name}】只驱动第二层的 X 40 ms：第一层 X 的组 ${g(0, XB, 0, 12).toFixed(1)} Hz、Y 的组 ${g(0, YB, 0, 12).toFixed(1)} Hz；第二层 Y 的组 ${g(1, YB, 0, 12).toFixed(1)} Hz；`
    + `第一层 Y 的组膜电位最高比起点高 ${(Math.max(...Array.from(vY).slice(0, 60)) - v0).toFixed(2)} mV，X 的组 ${(Math.max(...Array.from(vX).slice(0, 60)) - vx0).toFixed(2)} mV`);
  console.log(`   第一层 X、Y 的组放电（5 ms 一格）：X ${XB.map((q) => Array.from(out[0][q]).slice(0, 14).map((v) => v.toFixed(0).padStart(3)).join('')).join(' |')}  Y ${YB.map((q) => Array.from(out[0][q]).slice(0, 14).map((v) => v.toFixed(0).padStart(3)).join('')).join(' |')}`);
}
probe(trainedStack(2), '2 层');
probe(trainedStack(2, { variant: 'noseq' }), '2 层，上层没经历物体序列');
const cut = trainedStack(2); cut.layers[1].FB.scale.fill(0);
probe(cut, '2 层，切断反馈');
