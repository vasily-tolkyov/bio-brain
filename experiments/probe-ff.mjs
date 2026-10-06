// 调前馈强度：第一层训练好、第二层未学习时，给物体 X / Y，各层各组的放电
import { buildLayered, loadState, learnOnly, inputRate } from '../src/L4-network/layered-network.mjs';
import { cfg, proto, OBJ, mean, trainedState } from '../demo/lib-hierarchy.mjs';
import { makeRng } from '../src/core/rng.mjs';

const st = trainedState(1);
const blocks = ['A', 'B', 'C', 'D'].map(proto).concat([[...Array(20).keys()].map((i) => 80 + i)]);
for (const ffScale of (process.env.FF ?? '3,6,10').split(',').map(Number)) {
  const S = buildLayered(Number(process.env.NL ?? 2), { ...cfg, ffScale });
  loadState(S, st); learnOnly(S, []);
  const rng = makeRng(5);
  const rates = (drives, reps = 3) => {
    const acc = S.layers.map(() => blocks.map(() => 0));
    for (let r = 0; r < reps; r++) {
      S.net.run(400, { rate: inputRate(S), learn: false });
      const t0 = S.net.t;
      S.net.run(200, { rate: inputRate(S, drives), learn: false });
      S.layers.forEach((L, k) => blocks.forEach((b, q) => (acc[k][q] += mean(S.net.counts(t0, S.net.t, b.map((a) => L.E[a]))) / 0.2 / reps)));
    }
    return acc.map((row) => row.map((v) => v.toFixed(1)).join(' / ')).join('  ||  ');
  };
  const obj = (o) => new Map(OBJ[o].flatMap((p) => rng.shuffle(proto(p).slice()).slice(0, 12)).map((i) => [i, cfg.stim]));
  console.log(`前馈强度 ${ffScale}：静息 ${rates(new Map())}`);
  console.log(`              X ${rates(obj('X'))}`);
  console.log(`              Y ${rates(obj('Y'))}`);
}
