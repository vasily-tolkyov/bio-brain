/** 残缺 + 噪声下最上层的辨识度（(X 的组 − Y 的组)/(两者之和)）：每个网络用 8 组不同的噪声各测 3 次。
 * fig10 里每个网络只抽一组噪声，单组结果波动很大（同一网络从 −0.23 到 0.98）；这里给平均值。
 * 用法：node demo/noisy-depth.mjs（读 cache/hier-*.json，约 3 分钟） */
import { cfg, NE, proto, OBJ, mean, trainedStack } from '../demo/lib-hierarchy.mjs';
import { inputRate, spikesIn } from '../src/L4-network/layered-network.mjs';
import { makeRng } from '../src/core/rng.mjs';

const BL = ['A', 'B', 'C', 'D'], XB = OBJ.X.map((p) => BL.indexOf(p)), YB = OBJ.Y.map((p) => BL.indexOf(p));
const xIds = OBJ.X.flatMap(proto);
for (const arch of ['point', 'apical']) {
  const out = [];
  for (let K = 1; K <= 4; K++) {
    const S = trainedStack(K, { arch }), rng = makeRng(500), top = S.layers[K - 1], vals = [];
    for (let n = 0; n < 8; n++) {
      const pick = (L, m) => rng.shuffle(proto(L).slice()).slice(0, m);
      const noise = rng.shuffle([...Array(NE).keys()].filter((i) => !xIds.includes(i))).slice(0, 8);
      const drives = new Map([...pick(OBJ.X[0], 6), ...pick(OBJ.X[1], 6), ...noise].map((i) => [i, cfg.stim]));
      let x = 0, y = 0;
      for (let r = 0; r < 3; r++) {
        S.net.run(400, { rate: inputRate(S), learn: false });
        const ts = S.net.t; S.net.run(150, { rate: inputRate(S, drives), learn: false });
        const cnt = (qs) => mean(qs.map((q) => mean(spikesIn(S.net, proto(BL[q]).map((a) => top.E[a]), ts, ts + 150).map((s) => s.length))));
        x += cnt(XB); y += cnt(YB);
      }
      vals.push((x - y) / (x + y + 1e-9));
    }
    out.push(`${K} 层 ${mean(vals).toFixed(2)}（各组噪声 ${vals.map((v) => v.toFixed(2)).join(' ')}）`);
  }
  console.log(`${arch === 'point' ? '点神经元' : '带顶树突'}：${out.join('；')}`);
}
