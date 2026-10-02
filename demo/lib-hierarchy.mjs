/** 多层脉冲网络的世界与逐层训练（结果缓存到 cache/hier-*.json）。
 * 世界有三级结构：
 *   基本项 A、B、C、D（第一层的吸引子，各 20 个神经元；原型从不出现，只出现变体）；
 *   物体 X = A + D、Y = C + B，物体内部有先后：A 之后是 D，C 之后是 B（第一层的有向通道，短程预测）；
 *   物体之间有先后：X 之后是 Y（物体级的有向通道，长程预测）。第一层从不经历 X→Y，所以第一层自己预测不了 Y。
 *   （物体这样配对，是因为这个随机种子的第一层在变体学习中偶然长出了 A→D、C→B 两条有向联系——
 *    让它们落在物体内部；跨物体的联系都很弱。见 README。）
 * 逐层训练：
 *   第一层：变体 150 次 → 吸引子；A→D、C→B 各 40 遍（每项 40 ms）→ 物体内部的有向通道。
 *   第 k 层（k ≥ 2）：物体 100 次（两个部件同时出现，各取原型的 12 个成员，200 ms）→ 物体级吸引子 + 反馈突触；
 *     再经历物体序列 X → Y 120 遍（每个物体 40 ms）→ 物体级有向通道。
 *     训练第 k 层时下面各层的兴奋性突触冻结，抑制性可塑性保留（稳态）。
 *   上层（k ≥ 2）的异突触预算 0.3（第一层 0.2）。 */
import fs from 'fs';
import { CFG } from './fig2-network.mjs';
import { buildLayered, learnOnly, saveState, loadState, inputRate } from '../src/L4-network/layered-network.mjs';
import { makeRng } from '../src/core/rng.mjs';

// 上层的异突触预算更大：上层吸引子由下层两个吸引子合成（40 个神经元），预算 0.2 时组内连接几乎占满，物体级通道长不起来
export const cfg = { ...CFG, upperBudget: 0.3 };
export const { NE, M } = cfg;
export const proto = (L) => Array.from({ length: M }, (_, i) => 'ABCD'.indexOf(L) * M + i);
export const OBJ = { X: ['A', 'D'], Y: ['C', 'B'] };
export const mean = (x) => x.reduce((s, v) => s + v, 0) / (x.length || 1);
export const SEQ_REPS = 120, SEQ_ITEM_UPPER = CFG.seqItemMs;
const others = (L) => [...Array(NE).keys()].filter((i) => !proto(L).includes(i));

const run = (S, ms, drives, learn) => S.net.run(ms, { rate: inputRate(S, drives), learn });
const at = (ids, hz) => new Map(ids.map((i) => [i, hz]));

function trainFirst(log) {
  const S = buildLayered(1, cfg), L = S.layers[0], rng = makeRng(cfg.seed + 2), t0 = Date.now();
  learnOnly(S, [L.EE, L.IE]);
  run(S, 500, new Map(), false);
  for (let n = 0; n < cfg.nLearn; n++) {
    const k = 'ABCD'[rng.pick(4)];
    run(S, 200, at([...rng.shuffle(proto(k).slice()).slice(0, 16), ...rng.shuffle(others(k)).slice(0, 4)], cfg.stim), true);
    run(S, 300, new Map(), true);
    if (n % 50 === 49) log(`  第一层：变体 ${n + 1}/${cfg.nLearn}（${((Date.now() - t0) / 1000).toFixed(0)} s）`);
  }
  for (let n = 0; n < 80; n++) {
    const [p, q] = n % 2 ? OBJ.Y : OBJ.X;
    run(S, cfg.seqItemMs, at(proto(p), cfg.seqStim), true);
    run(S, cfg.seqItemMs, at(proto(q), cfg.seqStim), true);
    run(S, cfg.seqRestMs, new Map(), true);
  }
  log(`  第一层训练完成（${((Date.now() - t0) / 1000).toFixed(0)} s）`);
  return saveState(S);
}

/** 三种结构：point = 点神经元、反馈落在胞体；apical = 带顶树突、反馈落在顶树突（第一层没有反馈，这两种结构的第一层行为相同，共用缓存）；
 * deep = 每层分浅层（带顶树突，吸引子竞争，往上送证据）与深层（学有向通道、发出预测），预测落到下一层浅层的顶树突与下一层深层 */
// deep1 = 同样的浅层 + 深层，但序列由浅层学（短暂、有方向），深层不学转移、只经拷贝把浅层的预测保持下来往下送
// theta = 深层结构 + θ 节律（深层抑制按 7 Hz 起伏）+ 深层神经元的后去极化：被保持的几项在每个周期里按先后重演，深层在周期内的先后上学转移
// 后去极化要是阈下的：60 pA 时序列之后的第一个周期里深层的 X 比 Y 早约 25 ms 重演、之后熄灭；90 pA 能持续但顺序第二个周期就乱了，130 pA 失控
export const THETA = { theta: { f: 7, rate: 80 }, deepAdp: { amp: 60, rise: 50, decay: 200 } };
export const archCfg = (arch = 'point') => ({ ...cfg, apical: arch !== 'point', deep: ['deep', 'deep1', 'theta'].includes(arch), ...(arch === 'theta' ? THETA : {}) });

/** 深层结构的第一层：浅层用点神经元结构训练好的第一层；深层经历 A→D、C→B 序列（只有深层学习）→ 第一层的有向通道在深层 */
function trainDeepFirst(log, arch = 'deep') {
  const S = buildLayered(1, archCfg(arch)), L = S.layers[0], t0 = Date.now();
  loadState(S, trainedState(1, { log }));
  learnOnly(S, [L.DEE, L.DIE]);
  run(S, 500, new Map(), false);
  for (let n = 0; n < 80; n++) {
    const [p, q] = n % 2 ? OBJ.Y : OBJ.X;
    run(S, cfg.seqItemMs, at(proto(p), cfg.seqStim), true);
    run(S, cfg.seqItemMs, at(proto(q), cfg.seqStim), true);
    run(S, cfg.seqRestMs, new Map(), true);
  }
  log(`  第一层深层训练完成（${((Date.now() - t0) / 1000).toFixed(0)} s）`);
  return saveState(S);
}

function trainUpper(k, below, { seq = true, log, arch = 'point' }) {
  const S = buildLayered(k, archCfg(arch)), L = S.layers[k - 1], rng = makeRng(cfg.seed + 10 * k), t0 = Date.now();
  loadState(S, below);
  const homeo = S.layers.slice(0, k - 1).flatMap((q) => [q.IE, q.DIE]);
  const deepArch = ['deep', 'deep1', 'theta'].includes(arch);
  learnOnly(S, deepArch ? [L.EE, L.IE, L.DIE, L.FBA, L.FBD, ...homeo] : [L.EE, L.IE, L.FB, ...homeo]);
  run(S, 500, new Map(), false);
  for (let n = 0; n < 100; n++) {
    const parts = OBJ[rng.pick(2) ? 'Y' : 'X'];
    run(S, 200, at(parts.flatMap((p) => rng.shuffle(proto(p).slice()).slice(0, 12)), cfg.stim), true);
    run(S, 300, new Map(), true);
    if (n % 25 === 24) log(`  第 ${k} 层：物体 ${n + 1}/100（${((Date.now() - t0) / 1000).toFixed(0)} s）`);
  }
  if (seq) {
    learnOnly(S, arch === 'deep' || arch === 'theta' ? [L.DEE, L.DIE] : [L.EE]);
    for (let n = 0; n < SEQ_REPS; n++) {
      run(S, SEQ_ITEM_UPPER, at(OBJ.X.flatMap(proto), cfg.seqStim), true);
      run(S, SEQ_ITEM_UPPER, at(OBJ.Y.flatMap(proto), cfg.seqStim), true);
      run(S, cfg.seqRestMs, new Map(), true);
    }
  }
  log(`  第 ${k} 层训练完成（${((Date.now() - t0) / 1000).toFixed(0)} s）`);
  return saveState(S);
}

/** 训练好的 k 层状态（逐层缓存）；variant = 'noseq' 时最上层不经历物体序列（对照：没有物体级有向通道） */
export function trainedState(k, { variant = '', arch = 'point', log = console.log } = {}) {
  const tag = ['deep', 'deep1', 'theta'].includes(arch) ? `-${arch}` : k > 1 && arch === 'apical' ? '-ap' : '';
  if (k === 1 && arch === 'deep1') return trainedState(1, { log }); // 第一层的深层不学转移：浅层沿用，深层保持初值
  const file = `cache/hier-${k}${tag}${variant ? '-' + variant : ''}.json`;
  if (fs.existsSync(file)) return JSON.parse(fs.readFileSync(file, 'utf8'));
  const state = k === 1 ? (arch === 'deep' || arch === 'theta' ? trainDeepFirst(log, arch) : trainFirst(log)) : trainUpper(k, trainedState(k - 1, { arch, log }), { seq: variant !== 'noseq', log, arch });
  fs.mkdirSync('cache', { recursive: true });
  fs.writeFileSync(file, JSON.stringify(state));
  return state;
}

/** 搭一个 k 层网络并装入训练好的状态（全部冻结） */
export function trainedStack(k, opts = {}) {
  const S = buildLayered(k, archCfg(opts.arch));
  loadState(S, trainedState(k, opts));
  learnOnly(S, []);
  return S;
}
