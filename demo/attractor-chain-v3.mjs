/** v3：保留 v2 配置和旧缓存，分别测动力学需求与四个固定学习候选。 */
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { ROOT, CONFIG, LABELS, EDGES, G_GRID, B_GRID, mean, buildChain, setGain, run, restMetrics,
  thresholdFor, weightMatrix, trainSequence, calibrateCondition, calibrationChecks, runTrials,
  chainChecks, controlChecks, readResults } from './attractor-chain.mjs';
import { CANDIDATES, candidateSettings } from './chain-v3-settings.mjs';
import { runRateCase } from './rate-rule.mjs';
import { Canvas, PALETTE } from '../src/core/png.mjs';

export { CANDIDATES };
export const DOSES = [1, 2, 3, 5];
export const hash = value => crypto.createHash('sha256').update(JSON.stringify(value)).digest('hex');
export const weightHash = s => hash({ configHash: s.configHash, rho: s.rho, inhibition: s.inhibition });
export const fileFor = name => path.join(ROOT, 'cache', `chain-v3-${name}.json`);
export const read = file => JSON.parse(fs.readFileSync(file, 'utf8'));
export const save = (file, value) => {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(`${file}.tmp`, JSON.stringify(value, null, 2) + '\n'); fs.renameSync(`${file}.tmp`, file);
};
const oldState = name => {
  const s = read(path.join(ROOT, 'cache', `chain-v2-${name}.json`));
  if (s.configHash !== hash(CONFIG)) throw new Error('v2 缓存的 CONFIG 指纹不符，按停止条件禁止修改配置');
  return s;
};
const report = checks => { for (const c of checks) console.log(`${c.ok ? '✓' : '✗'} ${c.id} ${c.name}: ${JSON.stringify(c.actual)}`); };
export const getV3 = () => readResults().v3 ?? { schema: 3, phases: {}, calibration: { grid: [], workingPoint: null }, dose: { results: [], kStar: null }, learning: { candidates: [], best: null }, controls: {} };
export function updateV3(patch) {
  const result = readResults(), v3 = { ...getV3(), ...patch };
  result.v3 = v3; save(path.join(ROOT, 'figures', 'chain-results.json'), result); return v3;
}
export function selectPoint(grid) {
  if (grid.length !== 24) return null;
  const ok = grid.filter(r => calibrationChecks(r).every(c => c.ok));
  if (!ok.length) return null;
  const minG = Math.min(...ok.map(r => r.G));
  const points = ok.filter(r => r.G === minG).sort((a, b) => a.b - b.b);
  return points[Math.floor(points.length / 2)];
}
export function calibrationSource(candidate = null, episodes = 160) {
  return candidate ? read(fileFor(`seq-${candidate}-${episodes}-40`)) : oldState('sequence-320-40');
}
export function runCalibration(G, b, { candidate = null, episodes = 160 } = {}) {
  if (!G_GRID.includes(G) || !B_GRID.includes(b)) throw new Error('不在 v3 固定网格');
  const state = calibrationSource(candidate, episodes);
  return calibrateCondition(state, G, b, { allowFailedSequence: true,
    cachePath: fileFor(`${candidate ? `cand-${candidate}-${episodes}-` : ''}calib-g${G}-b${b}`) });
}
export function collectCalibration({ candidate = null, episodes = 160 } = {}) {
  const source = calibrationSource(candidate, episodes), grid = [];
  for (const G of G_GRID) for (const b of B_GRID) {
    const file = fileFor(`${candidate ? `cand-${candidate}-${episodes}-` : ''}calib-g${G}-b${b}`);
    if (fs.existsSync(file)) {
      const point = read(file);
      const fingerprint = hash({ schema: 3, mode: 'calib', configHash: source.configHash, stateHash: weightHash(source), G, b, settings: source.v3Settings ?? {} });
      if (point.stateHash !== weightHash(source) || point.fingerprint !== fingerprint) throw new Error('标定网格缓存指纹不符');
      grid.push(point);
    }
  }
  const workingPoint = selectPoint(grid), r = { source: candidate ?? 'v2-320-40', sourceHash: weightHash(source), episodes: candidate ? episodes : 320, grid, workingPoint,
    status: grid.length < 24 ? 'incomplete' : workingPoint ? 'passed' : 'failed' };
  updateV3(candidate ? { candidateCalibration: r } : { calibration: r });
  console.log(`标定汇总 ${r.source}: ${grid.length}/24, 工作点 ${workingPoint ? `${workingPoint.G}/${workingPoint.b}` : '无'}`); return r;
}
export function weightStats(state) {
  const S = buildChain(state.config, state), w = weightMatrix(S);
  const forward = EDGES.map(([a, z]) => w.matrix[a][z]), backward = EDGES.map(([a, z]) => w.matrix[z][a]);
  const unexperienced = [];
  for (let a = 0; a < 8; a++) for (let z = 0; z < 8; z++) if (a !== z && !EDGES.some(([i, j]) => a === i && z === j)) unexperienced.push(w.matrix[a][z]);
  const rhoBg = mean(unexperienced), histogram = Array(10).fill(0);
  for (const rho of state.rho) histogram[Math.min(9, Math.max(0, Math.floor(rho * 10)))]++;
  return { rhoBg, forwardMean: mean(forward), backwardMean: mean(backward), Ef: mean(forward) - rhoBg, Eb: mean(backward) - rhoBg,
    withinMean: mean(w.matrix.map((row, g) => row[g])), histogram: { binWidth: 0.1, counts: histogram, total: state.rho.length, lastBinIncludesOne: true }, weights: w };
}
export function trainCandidate(candidate, episodes = 160, { shuffled = false, main = null } = {}) {
  const settings = candidateSettings(candidate);
  if (![160, 320].includes(episodes) || (candidate === 'L0' && !shuffled)) throw new Error('候选训练不在计划中');
  const file = shuffled ? (candidate === 'L0' ? 'chain-v3-shuffled' : `chain-v3-shuffled-${candidate}-${episodes}`) : `chain-v3-seq-${candidate}-${episodes}-40`;
  return trainSequence(oldState('attractors'), { episodes, itemMs: 40, learnGain: 0.6, shuffled,
    matchedPaths: main?.training.paths ?? null, v3: { file, settings,
      ...(!shuffled && episodes === 320 ? { continueFile: `chain-v3-seq-${candidate}-160-40` } : {}) } });
}
export function selectivityChecks(classes, rest) {
  const c = classes?.budgetOn?.classes, first = rest?.trace?.[0], last = rest?.trace?.at(-1);
  const delta = first && last ? Object.fromEntries(['within', 'between', 'ungrouped'].map(k => [k, last[k] - first[k]])) : null;
  return [
    { id: 'classes', name: '四类选择性', ok: !!c && c.AtoA.meanDelta >= 0.15 && c.AtoOut.meanDelta <= 0.1 * c.AtoA.meanDelta && c.OutToA.meanDelta <= 0.2 * c.AtoA.meanDelta && c.OutToOut.meanDelta <= 0.01, actual: c ?? '未运行' },
    { id: 'rest', name: '静息 30 s 各类平均 Δρ ≤ +0.005', ok: !!delta && Object.values(delta).every(x => x <= 0.005), actual: delta ?? '未运行' },
  ];
}
export function candidateChecks(state, classes, rest) {
  const d = state?.measurements.direction, ignition = state?.measurements.rest.ignitionRate;
  return [...selectivityChecks(classes, rest),
    { id: 'eachDirection', name: '七对方向差各 ≥ 0.02', ok: !!d && d.edges.every(e => e.difference >= 0.02), actual: d?.edges ?? '未运行' },
    { id: 'branch', name: 'C→D > C→G', ok: !!d && d.branchD > d.branchG, actual: d ? { D: d.branchD, G: d.branchG } : '未运行' },
    { id: 'experience', name: 'Spearman ≥ 0.5 且前七大至少五对有经历', ok: !!d && d.spearman >= 0.5 && d.top7Experienced >= 5, actual: d ? { spearman: d.spearman, top7Experienced: d.top7Experienced } : '未运行' },
    { id: 'attractors', name: '组内/组间平均比 ≥ 2，各组 ≥ 1.8', ok: !!d && d.meanRatio >= 2 && d.ratios.every(r => r >= 1.8), actual: d ? { mean: d.meanRatio, groups: d.ratios } : '未运行' },
    { id: 'ignition', name: '学习态静息点燃 ≤ 5%', ok: ignition !== undefined && ignition <= 0.05, actual: ignition ?? '未运行' },
  ];
}
export function collectLearning() {
  const candidates = [], rate = { schema: 3, candidates: {} };
  for (const candidate of Object.keys(CANDIDATES)) {
    const stateFile = fileFor(`seq-${candidate}-160-40`), cFile = fileFor(`rate-classes-${candidate}`), rFile = fileFor(`rate-rest-${candidate}`);
    if (!fs.existsSync(stateFile) || !fs.existsSync(cFile) || !fs.existsSync(rFile)) continue;
    const state = read(stateFile);
    // 读取既有候选结果时也验证规则指纹，不会重新模拟或触碰 v2 文件。
    const classes = runRateCase('classes', 'GB_RATE', { candidate }), rest = runRateCase('rest', 'GB_RATE', { candidate });
    if (hash(state.v3Settings) !== hash(candidateSettings(candidate))) throw new Error('候选训练设置不符');
    const checks = candidateChecks(state, classes, rest), stats = weightStats(state);
    candidates.push({ candidate, settings: candidateSettings(candidate), sourceHash: state.sourceHash, measurements: state.measurements, stats, checks, usable: checks.every(c => c.ok) });
    rate.candidates[candidate] = { classes, rest, checks: selectivityChecks(classes, rest) };
  }
  const usable = candidates.filter(c => c.usable).sort((a, b) => b.stats.Ef - a.stats.Ef);
  const r = { baseline: { candidate: 'L0', measurements: oldState('sequence-160-40').measurements, stats: weightStats(oldState('sequence-160-40')) }, candidates,
    best: candidates.length === 4 ? usable[0]?.candidate ?? null : null, status: candidates.length === 4 ? 'complete' : 'incomplete' };
  save(path.join(ROOT, 'figures', 'rate-rule-v3.json'), rate); updateV3({ learning: r });
  for (const c of candidates) console.log(`${c.candidate}: Ef=${c.stats.Ef.toFixed(6)}, Eb=${c.stats.Eb.toFixed(6)}, usable=${c.usable}`);
  console.log(`最佳可用候选：${r.best ?? '无'}`); return r;
}

export function runV3Mode(mode) {
  if (mode === 'v3-calib') {
    const candidate = process.env.CANDIDATE ?? null, episodes = Number(process.env.EPISODES ?? 160);
    if (process.env.COLLECT_ONLY !== '1') {
      const gs = process.env.G ? [Number(process.env.G)] : G_GRID, bs = process.env.B ? [Number(process.env.B)] : B_GRID;
      for (const G of gs) for (const b of bs) runCalibration(G, b, { candidate, episodes });
    }
    if ((!process.env.G && !process.env.B) || process.env.COLLECT_ONLY === '1') collectCalibration({ candidate, episodes });
  } else if (mode === 'v3-sequence') trainCandidate(process.env.CANDIDATE, Number(process.env.EPISODES ?? 160));
  else if (mode === 'v3-learning-collect') collectLearning();
  else throw new Error(`尚未实现的 v3 模式 ${mode}`);
  return 0;
}
