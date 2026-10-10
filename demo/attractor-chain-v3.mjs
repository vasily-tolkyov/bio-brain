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
  // 42 个背景组对，正向与其七条反向均排除；对应计划给出的 320×40 背景约 0.031。
  for (let a = 0; a < 8; a++) for (let z = 0; z < 8; z++) if (a !== z && !EDGES.some(([i, j]) => (a === i && z === j) || (a === j && z === i))) unexperienced.push(w.matrix[a][z]);
  const rhoBg = mean(unexperienced), histogram = Array(10).fill(0);
  for (const rho of state.rho) histogram[Math.min(9, Math.max(0, Math.floor(rho * 10)))]++;
  return { rhoBg, backgroundPairs: 42, backgroundDefinition: 'neither-forward-nor-reverse', forwardMean: mean(forward), backwardMean: mean(backward), Ef: mean(forward) - rhoBg, Eb: mean(backward) - rhoBg,
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

export function amplifyForward(base, k) {
  if (!DOSES.includes(k)) throw new Error('剂量不在 k=1/2/3/5 中');
  const S = buildChain(base.config, base), before = weightStats(base), rhoBg = before.rhoBg;
  const factors = EDGES.map(([a, z]) => {
    const original = before.weights.matrix[a][z], target = rhoBg + k * (original - rhoBg);
    return { from: a, to: z, original, target, multiplier: k === 1 ? 1 : target / original };
  });
  let changed = 0, untouched = 0, clipped = 0;
  for (let i = 0; i < S.EE.rho.length; i++) {
    const a = S.EE.pre[i] < 640 ? Math.floor(S.EE.pre[i] / 80) : -1;
    const z = S.EE.post[i] < 640 ? Math.floor(S.EE.post[i] / 80) : -1;
    const factor = factors.find(f => f.from === a && f.to === z);
    if (factor && k !== 1) { const target = base.rho[i] * factor.multiplier; if (target > 1) clipped++; S.EE.rho[i] = Math.min(1, target); changed++; }
    else { if (S.EE.rho[i] !== base.rho[i]) throw new Error('非目标突触发生变化'); untouched++; }
  }
  const state = { ...base, rho: Array.from(S.EE.rho), inhibition: base.inhibition.slice() };
  const stats = weightStats(state);
  const edges = factors.map(f => { const actual = stats.weights.matrix[f.from][f.to], shortfall = Math.max(0, (f.target - actual) / f.target);
    return { ...f, actual, shortfall, clippedBelowTargetOver5Percent: shortfall > 0.05 }; });
  return { state, stats, report: { k, rhoBg, edges, changed, untouched, clipped, inhibitionUnchanged: true,
    branchExcessRatioBefore: (before.weights.matrix[2][3] - rhoBg) / (before.weights.matrix[2][6] - rhoBg),
    branchExcessRatioAfter: (stats.weights.matrix[2][3] - rhoBg) / (stats.weights.matrix[2][6] - rhoBg) } };
}
export function failureTypes(result) {
  if (!result.trials?.length) return { status: 'trialsNotRun', stoppedTrialFraction: null, unexperiencedProportion: null, reverseProportion: null, uncontrolledBinFraction: null, breakAtCounts: {}, runawayRest: true };
  const transitions = result.trials.flatMap(t => t.decoded.transitions), bins = result.trials.flatMap(t => t.bins), breakAtCounts = {};
  for (const trial of result.trials) {
    if (trial.decoded.completed) continue;
    const dwells = trial.decoded.dwells;
    let group = dwells[0]?.group === 0 ? 0 : null;
    if (group !== null) for (let i = 1; i < dwells.length; i++) {
      const previous = dwells[i - 1], next = dwells[i];
      if (next.start - previous.end > 100 || !EDGES.some(([a, z]) => a === previous.group && z === next.group)) break;
      group = next.group;
    }
    const outgoing = group === null ? 'cue-A' : EDGES.filter(([a]) => a === group).map(([a, z]) => `${LABELS[a]}>${LABELS[z]}`).join('/') || `${LABELS[group]}-terminal`;
    breakAtCounts[outgoing] = (breakAtCounts[outgoing] ?? 0) + 1;
  }
  return { status: 'measured', stoppedTrialFraction: mean(result.trials.map(t => Number(t.decoded.transitions.length === 0))),
    // 未经历转移包括反向；反向同时单列，两类统计不互斥。
    unexperiencedProportion: transitions.length ? mean(transitions.map(t => Number(!t.forward))) : 0,
    reverseProportion: transitions.length ? mean(transitions.map(t => Number(t.reverse))) : 0,
    uncontrolledBinFraction: mean(bins.map(row => Number(row.filter(rate => rate > result.threshold.threshold).length >= 2))),
    breakAtCounts, runawayRest: false, transitionCount: transitions.length, bins: bins.length };
}
export function runDose(k) {
  const workingPoint = getV3().calibration.workingPoint;
  if (!workingPoint) throw new Error('D1 没有工作点，D2 按情形 C 不运行');
  const base = oldState('sequence-320-40'), amplified = amplifyForward(base, k), { G, b } = workingPoint;
  const file = fileFor(`dose-k${k}`), doseFingerprint = hash({ schema: 3, sourceHash: weightHash(base), stateHash: weightHash(amplified.state), k, G, b, backgroundDefinition: amplified.stats.backgroundDefinition });
  if (fs.existsSync(file)) { const cached = read(file); if (cached.k !== undefined) { if (cached.doseFingerprint !== doseFingerprint) throw new Error('剂量缓存指纹不符'); return cached; } }
  const S = buildChain(base.config, amplified.state, { G, b, noiseSeed: 7001 }), threshold = thresholdFor(S);
  const rest = restMetrics(S, 3000, threshold.threshold); delete rest.bins10;
  const uncontrolled = rest.ignitionRate > 0.1;
  const r = uncontrolled ? { stateHash: weightHash(amplified.state), G, b, threshold, trials: [], rejected: [], finished: false, summary: null }
    : runTrials(amplified.state, { G, b, enforceRest: true, name: `v3-dose-k${k}`, cachePath: file });
  Object.assign(r, { k, sourceHash: weightHash(base), doseFingerprint, artificial: k !== 1, amplification: amplified.report, stats: amplified.stats, rest,
    status: uncontrolled ? 'uncontrolled' : 'measured' });
  r.failures = failureTypes(r);
  r.checks = uncontrolled ? ['4.1', '4.2', '4.3', '4.4', '4.5'].map(id => ({ id, name: '静息失控，链试次未运行', ok: false, actual: '未运行' })) : chainChecks(r.summary);
  r.ok = !uncontrolled && r.finished && r.checks.every(c => c.ok); save(file, r);
  console.log(`k=${k}, ignition=${rest.ignitionRate}, ok=${r.ok}`); report(r.checks); return r;
}
export function collectDoses() {
  const v3 = getV3();
  if (!v3.calibration.workingPoint) {
    const r = { status: 'notRun', reason: 'D1 无工作点，转情形 C', results: [], kStar: null }; updateV3({ dose: r }); return r;
  }
  const results = [];
  for (const k of DOSES) if (fs.existsSync(fileFor(`dose-k${k}`))) {
    const r = read(fileFor(`dose-k${k}`));
    if (r.k !== k || r.sourceHash !== weightHash(oldState('sequence-320-40'))) throw new Error('剂量结果来源不符'); results.push(r);
  }
  const kStar = results.length === 4 ? results.filter(r => r.ok).sort((a, b) => a.k - b.k)[0]?.k ?? null : null;
  const r = { results, kStar, status: results.length === 4 ? 'complete' : 'incomplete' }; updateV3({ dose: r });
  console.log(`剂量表 ${results.length}/4, k*=${kStar ?? '无'}`); return r;
}
export function decisionFor(v3) {
  if (v3.calibration.grid.length !== 24 || v3.learning.candidates.length !== 4) throw new Error('D1 或 L1 未完整执行，不能决策');
  if (!v3.calibration.workingPoint) return { case: 'C', action: 'stop', reason: 'D1 完整网格没有工作点', kStar: null, Ereq: null };
  if (v3.dose.results.length !== 4) throw new Error('D2 四个剂量未完整执行，不能决策');
  const kStar = DOSES.find(k => v3.dose.results.find(r => r.k === k)?.ok) ?? null;
  if (kStar === null) return { case: 'C', action: 'stop', reason: 'k=1/2/3/5 均未通过，瓶颈在演化态动力学', kStar: null, Ereq: null };
  if (kStar === 1) return { case: 'A', action: 'controls', reason: 'v2 学到的权重已能驱动链，平均方向差门槛由实测动力学需求替代', kStar, Ereq: v3.dose.results.find(r => r.k === 1).stats.Ef, candidate: 'L0', episodes: 320 };
  const Ereq = v3.dose.results.find(r => r.k === kStar).stats.Ef;
  const best = v3.learning.candidates.filter(c => c.usable).sort((a, b) => b.stats.Ef - a.stats.Ef)[0];
  if (!best) return { case: 'B', action: 'stop', reason: '没有可用学习候选', kStar, Ereq, candidate: null };
  const Ef = best.stats.Ef;
  return { case: 'B', action: Ef >= Ereq ? 'candidateCalibration' : Ef >= 0.8 * Ereq ? 'continue320' : 'stop',
    reason: Ef >= Ereq ? '最佳可用候选达到动力学需求' : Ef >= 0.8 * Ereq ? '最佳候选达到需求的 80%，启用唯一一次续训补救' : '最佳可用候选低于需求的 80%，按规则停止',
    kStar, Ereq, Ef, candidate: best.candidate, episodes: 160 };
}
export function decide() {
  const v3 = getV3(), decision = decisionFor(v3); updateV3({ decision });
  if (decision.case === 'A') updateV3({ mainTest: v3.dose.results.find(r => r.k === 1) });
  console.log(`决策：${JSON.stringify(decision)}`); return decision;
}
export function remedy() {
  const decision = getV3().decision;
  if (decision?.action !== 'continue320') throw new Error('规则未授权 320 回合补救');
  const state = trainCandidate(decision.candidate, 320), stats = weightStats(state);
  const classes = runRateCase('classes', 'GB_RATE', { candidate: decision.candidate }), rest = runRateCase('rest', 'GB_RATE', { candidate: decision.candidate });
  const checks = candidateChecks(state, classes, rest), usable = checks.every(c => c.ok), attained = stats.Ef >= decision.Ereq;
  const result = { candidate: decision.candidate, episodes: 320, stats, measurements: state.measurements, checks, usable, attained };
  updateV3({ remedy: result, decision: { ...decision, episodes: 320, Ef: stats.Ef, action: attained && usable ? 'candidateCalibration' : 'stop',
    reason: attained && usable ? '唯一一次续训补救达到需求' : '唯一一次续训补救仍未达到需求或可用性条件' } }); return result;
}
function mainState() {
  const decision = getV3().decision;
  if (!decision || decision.action === 'stop' || decision.action === 'continue320') throw new Error('决策未授权主测试或对照');
  return decision.case === 'A' ? oldState('sequence-320-40') : read(fileFor(`seq-${decision.candidate}-${decision.episodes}-40`));
}
function mainPoint() {
  const v3 = getV3();
  return v3.decision.case === 'A' ? v3.calibration.workingPoint : v3.candidateCalibration?.workingPoint;
}
export function runMainTest() {
  const state = mainState(), v3 = getV3();
  if (v3.decision.case === 'A') return v3.mainTest;
  const point = mainPoint();
  if (!point) { updateV3({ decision: { ...v3.decision, action: 'stop', reason: '候选完整网格没有工作点' } }); return null; }
  const result = runTrials(state, { G: point.G, b: point.b, enforceRest: true, name: 'v3-main', cachePath: fileFor(`main-${v3.decision.candidate}-${v3.decision.episodes}`) });
  result.checks = chainChecks(result.summary); result.ok = result.finished && result.checks.every(c => c.ok);
  updateV3({ mainTest: result, ...(!result.ok ? { decision: { ...v3.decision, action: 'stop', reason: '学到的候选权重未通过主链测试' } } : {}) }); report(result.checks); return result;
}
export function runV3Control(kind) {
  if (!['noSequence', 'shuffled', 'noAdaptation', 'lowNoise'].includes(kind)) throw new Error('对照不在 5a–5d 中');
  const v3 = getV3();
  if (!v3.mainTest?.ok) throw new Error('主测试未通过，按规则不运行对照');
  const main = mainState(), point = mainPoint(), { G, b } = point;
  let state = main;
  if (kind === 'noSequence') state = oldState('attractors');
  else if (kind === 'shuffled') state = trainCandidate(v3.decision.candidate, v3.decision.episodes, { shuffled: true, main });
  const file = fileFor(`control-${kind}-${v3.decision.candidate}-${v3.decision.episodes}`);
  const r = runTrials(state, { G, b: kind === 'noAdaptation' ? 0 : b, lowNoise: kind === 'lowNoise', name: `v3-${kind}`, cachePath: file });
  if (kind === 'shuffled') r.direction = state.measurements.direction;
  r.mainSourceHash = weightHash(main); save(file, r); return r;
}
export function collectV3Controls() {
  const v3 = getV3(), controls = {};
  for (const kind of ['noSequence', 'shuffled', 'noAdaptation', 'lowNoise']) {
    const file = fileFor(`control-${kind}-${v3.decision.candidate}-${v3.decision.episodes}`);
    if (fs.existsSync(file)) { const r = read(file); if (r.mainSourceHash !== weightHash(mainState())) throw new Error('对照主状态来源不符'); controls[kind] = r; }
  }
  const checks = controlChecks(controls, v3.mainTest?.summary).slice(0, 4); updateV3({ controls, controlChecks: checks }); report(checks); return { controls, checks };
}

const missing = (ids, name) => ids.map(id => ({ id, name: `${name}未运行`, ok: false, actual: '未运行' }));
export function v3Checks(v3 = getV3()) {
  const grid = v3.calibration.grid, expectedPoint = selectPoint(grid), point = v3.calibration.workingPoint;
  const learning = v3.learning, candidates = learning.candidates;
  const usable = candidates.filter(c => c.usable).sort((a, b) => b.stats.Ef - a.stats.Ef);
  const doses = v3.dose.results;
  const expectedK = DOSES.find(k => doses.find(d => d.k === k)?.ok) ?? null;
  let decisionOk = false, decisionActual;
  try {
    const expected = decisionFor(v3), actual = v3.decision;
    if (v3.remedy) {
      const r = v3.remedy;
      const action = r.attained && r.usable ? 'candidateCalibration' : 'stop';
      decisionOk = expected.action === 'continue320' && actual.case === 'B' && actual.candidate === expected.candidate && actual.episodes === 320
        && actual.Ereq === expected.Ereq && r.stats.Ef === actual.Ef && r.attained === (r.stats.Ef >= expected.Ereq) && actual.action === action;
      if (actual.reason === '学到的候选权重未通过主链测试' || actual.reason === '候选完整网格没有工作点') decisionOk = expected.action === 'continue320' && actual.action === 'stop' && r.attained && r.usable;
    } else {
      decisionOk = actual?.case === expected.case && actual?.kStar === expected.kStar && actual?.Ereq === expected.Ereq && actual?.candidate === expected.candidate
        && (actual.action === expected.action || (expected.case === 'B' && expected.action === 'candidateCalibration' && actual.action === 'stop'
          && ['学到的候选权重未通过主链测试', '候选完整网格没有工作点'].includes(actual.reason)));
    }
    decisionActual = { expected, actual };
  } catch (error) { decisionActual = error.message; }
  const allGrid = grid.length === 24 && G_GRID.every(G => B_GRID.every(b => grid.filter(p => p.G === G && p.b === b).length === 1));
  const main = v3.mainTest, controls = v3.controls ?? {};
  return [
    { id: 'D1.0', name: '24 个标定点完整且不重复', ok: allGrid, actual: grid.length },
    { id: 'D1.select', name: '工作点按最小 G、同 G 中间 b 选取', ok: (expectedPoint?.G ?? null) === (point?.G ?? null) && (expectedPoint?.b ?? null) === (point?.b ?? null), actual: point ? { G: point.G, b: point.b } : '无工作点' },
    ...(point ? calibrationChecks(point).map(c => ({ ...c, id: `D1.${c.id}` })) : missing(['D1.3.1', 'D1.3.2', 'D1.3.3', 'D1.3.4'], '工作点条件')),
    { id: 'D2.0', name: '四个剂量及失败类型均有结果', ok: doses.length === 4 && DOSES.every(k => doses.some(d => d.k === k && !!d.failures && d.checks?.length === 5)), actual: v3.dose.status === 'notRun' ? '未运行：D1 无工作点' : doses.length },
    { id: 'D2.select', name: 'k* 为最小通过剂量或确定无', ok: doses.length === 4 && v3.dose.kStar === expectedK, actual: v3.dose.status === 'notRun' ? '未运行' : v3.dose.kStar ?? '无' },
    { id: 'L1.0', name: '四个候选训练、复查与直方图完整', ok: candidates.length === 4 && Object.keys(CANDIDATES).every(id => candidates.some(c => c.candidate === id && c.checks.length === 7 && c.stats.histogram.counts.reduce((a, b) => a + b, 0) === c.stats.histogram.total)), actual: candidates.length },
    { id: 'L1.usability', name: '候选可用性按全部七条条件确定', ok: candidates.length === 4 && candidates.every(c => c.usable === c.checks.every(p => p.ok)), actual: candidates.map(c => ({ candidate: c.candidate, usable: c.usable, failed: c.checks.filter(p => !p.ok).map(p => p.id) })) },
    { id: 'L1.best', name: '最佳候选为可用者中的最大 E_f 或确定无', ok: candidates.length === 4 && learning.best === (usable[0]?.candidate ?? null), actual: learning.best ?? '无' },
    { id: 'decision', name: '决策及唯一补救符合预先规则', ok: decisionOk, actual: decisionActual },
    ...(v3.decision?.case === 'B' && v3.candidateCalibration ? [
      { id: 'B.grid', name: '候选重新标定完整 24 点', ok: v3.candidateCalibration.grid.length === 24, actual: v3.candidateCalibration.grid.length },
      ...(v3.candidateCalibration.workingPoint ? calibrationChecks(v3.candidateCalibration.workingPoint).map(c => ({ ...c, id: `B.${c.id}` })) : missing(['B.3.1', 'B.3.2', 'B.3.3', 'B.3.4'], '候选工作点条件')),
    ] : []),
    ...(main ? [{ id: '4.0', name: '主测试采用学到的权重、40 个有效试次', ok: main.finished && main.trials.length === 40 && !main.artificial, actual: { count: main.trials.length, artificial: main.artificial ?? false } }, ...chainChecks(main.summary)]
      : missing(['4.0', '4.1', '4.2', '4.3', '4.4', '4.5'], '学到的权重主测试')),
    ...controlChecks(controls, main?.summary).slice(0, 4),
  ];
}
export function drawV3(v3 = getV3()) {
  const c = new Canvas(1800, 760, [247, 249, 252]), ink = [30, 45, 60], muted = [90, 105, 120];
  const blue = [31, 119, 180], green = [44, 160, 44], orange = [255, 127, 14], red = [214, 39, 40];
  c.text(26, 22, 'DIRECTED ATTRACTOR CHAIN V3 - DOSE AND LEARNING', ink, 3);
  c.text(26, 58, `CASE ${v3.decision?.case ?? '-'}  K-STAR ${v3.dose.kStar ?? 'NONE'}  ${v3.outcome ?? 'IN PROGRESS'}  V2 320X40 SOURCE - LEARNING G 0.6`, muted, 2);
  const panel = (x, title) => { c.rect(x, 98, 565, 625, [255, 255, 255]); c.text(x + 16, 116, title, ink, 2); };
  panel(25, 'A. DOSE RESPONSE'); panel(615, 'B. LEARNED EXCESS RHO'); panel(1205, 'C. DECODED STATES - 20 TRIALS');
  const leftX = 86, top = 215, bottom = 526, width = 430;
  const doses = v3.dose.results;
  if (doses.length) {
    for (let n = 0; n <= 4; n++) {
      const y = bottom - n / 4 * (bottom - top); c.line(leftX, y, leftX + width, y, [229, 235, 240]);
      c.text(leftX - 48, y - 5, n * 25, muted, 1); c.text(leftX + width + 12, y - 5, (n * 1.25).toFixed(2), muted, 1);
    }
    c.text(leftX - 30, 183, 'PERCENT', muted, 1); c.text(leftX + width - 55, 183, 'CHAIN MEDIAN', muted, 1);
    const xpos = k => leftX + (k - 1) / 4 * width;
    const series = [d => d.summary?.completionRate, d => d.summary?.forwardProportion, d => d.summary ? d.summary.chainMedian / 5 : null];
    for (let n = 0; n < series.length; n++) {
      let previous = null;
      for (const d of doses) {
        const value = series[n](d), color = [blue, green, orange][n];
        if (value === undefined || value === null) { previous = null; continue; }
        const x = xpos(d.k), y = bottom - value * (bottom - top);
        if (previous) c.line(previous.x, previous.y, x, y, color, 3);
        c.rect(x - 4, y - 4, 9, 9, color); previous = { x, y };
      }
    }
    for (const d of doses) { c.text(xpos(d.k) - 6, 549, d.k, ink, 2); if (d.status === 'uncontrolled') c.text(xpos(d.k) - 35, 570, 'NO TRIALS', red, 1); }
    c.text(274, 580, 'DOSE K', ink, 2);
    for (const [n, label] of ['COMPLETION', 'FORWARD', 'CHAIN MEDIAN - RIGHT AXIS'].entries()) { c.rect(65, 620 + n * 24, 20, 12, [blue, green, orange][n]); c.text(98, 621 + n * 24, label, muted, 1); }
  } else c.text(58, 340, 'NOT RUN - NO WORKING POINT', muted, 2);
  const bars = [v3.learning.baseline, ...v3.learning.candidates].filter(Boolean);
  if (bars.length) {
    const req = v3.decision?.Ereq, values = bars.flatMap(b => [b.stats.Ef, b.stats.Eb]);
    const min = Math.min(0, ...values) - 0.01, max = Math.max(0.05, ...values, req ?? 0) * 1.12;
    const yval = v => bottom - (v - min) / (max - min) * (bottom - top), zero = yval(0);
    for (let i = 0; i <= 4; i++) { const value = min + (max - min) * i / 4, y = yval(value); c.line(677, y, 1135, y, [229, 235, 240]); c.text(636, y - 3, value.toFixed(2), muted, 1); }
    bars.forEach((b, i) => {
      const x = 706 + i * 85;
      for (const [j, value] of [b.stats.Ef, b.stats.Eb].entries()) { const y = yval(value); c.rect(x + j * 29, Math.min(y, zero), 24, Math.max(1, Math.abs(y - zero)), j ? orange : blue); }
      c.text(x, 549, b.candidate, ink, 2); c.text(x, 576, b.candidate === 'L0' ? 'BASE' : b.usable ? 'OK' : 'FAIL', b.usable ? green : muted, 1);
    });
    if (Number.isFinite(req)) { const y = yval(req); for (let x = 680; x < 1138; x += 14) c.line(x, y, x + 8, y, red, 2); c.text(691, y - 21, `E-REQ ${req.toFixed(4)}`, red, 1); }
    else c.text(683, 180, 'E-REQ NOT MEASURED', muted, 1);
    c.rect(665, 624, 20, 12, blue); c.text(699, 625, 'E-F FORWARD MINUS BACKGROUND', muted, 1);
    c.rect(665, 654, 20, 12, orange); c.text(699, 655, 'E-B REVERSE MINUS BACKGROUND', muted, 1);
    c.text(665, 690, 'BACKGROUND: 42 NON-ADJACENT ORDERED PAIRS', muted, 1);
  } else c.text(660, 340, 'LEARNING SCREEN NOT RUN', muted, 2);
  const raster = v3.mainTest ?? doses.find(d => d.k === v3.dose.kStar);
  if (raster?.trials?.length) {
    c.text(1230, 162, v3.mainTest ? 'LEARNED MAIN STATE' : `ARTIFICIAL K=${raster.k} - DIAGNOSTIC`, muted, 1);
    const x = 1265, y = 206, rowHeight = 18, rasterWidth = 452, colors = [...PALETTE, [23, 170, 165]];
    raster.trials.slice(0, 20).forEach((trial, i) => { c.text(x - 31, y + i * rowHeight, i + 1, muted, 1);
      trial.decoded.states.forEach((g, t) => { if (g >= 0) c.rect(x + t / trial.decoded.states.length * rasterWidth, y + i * rowHeight, Math.ceil(rasterWidth / trial.decoded.states.length), 14, colors[g]); }); });
    for (let s = 0; s <= 3; s++) c.text(x + s / 3.04 * rasterWidth, 585, `${s}S`, muted, 1);
    for (let g = 0; g < 8; g++) { const px = 1240 + (g % 4) * 125, py = 633 + Math.floor(g / 4) * 28; c.rect(px, py, 19, 14, colors[g]); c.text(px + 27, py, LABELS[g], ink, 2); }
  } else c.text(1235, 340, 'NO MAIN OR PASSING DOSE', muted, 2);
  const file = path.join(ROOT, 'figures', 'fig14-chain-dose.png'); c.save(file); return file;
}
export function finalizeV3() {
  const v3 = getV3(); v3.checks = v3Checks(v3);
  v3.outcome = v3.checks.every(c => c.ok) ? 'passed' : 'partial';
  v3.stopReason = v3.decision?.action === 'stop' ? v3.decision.reason : v3.outcome === 'partial' ? '主测试或对照未全部通过' : null;
  updateV3(v3); drawV3(v3); report(v3.checks); return v3;
}
export function runV3All() {
  for (const G of G_GRID) for (const b of B_GRID) runCalibration(G, b);
  const calib = collectCalibration();
  for (const candidate of Object.keys(CANDIDATES)) { trainCandidate(candidate); for (const mode of ['classes', 'rest']) runRateCase(mode, 'GB_RATE', { candidate }); }
  collectLearning();
  if (calib.workingPoint) for (const k of DOSES) runDose(k);
  collectDoses(); let decision = decide();
  if (decision.action === 'continue320') { remedy(); decision = getV3().decision; }
  if (decision.action === 'candidateCalibration') {
    for (const G of G_GRID) for (const b of B_GRID) runCalibration(G, b, { candidate: decision.candidate, episodes: decision.episodes });
    collectCalibration({ candidate: decision.candidate, episodes: decision.episodes }); runMainTest();
  }
  if (getV3().mainTest?.ok) { for (const kind of ['noSequence', 'shuffled', 'noAdaptation', 'lowNoise']) runV3Control(kind); collectV3Controls(); }
  return finalizeV3();
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
  else if (mode === 'v3-dose') {
    if (process.env.COLLECT_ONLY !== '1') for (const k of process.env.K ? [Number(process.env.K)] : DOSES) runDose(k);
    if (!process.env.K || process.env.COLLECT_ONLY === '1') collectDoses();
  } else if (mode === 'v3-decision') decide();
  else if (mode === 'v3-remedy') remedy();
  else if (mode === 'v3-test') { const r = runMainTest(); return r?.ok ? 0 : 2; }
  else if (mode === 'v3-controls') {
    if (process.env.COLLECT_ONLY !== '1') for (const kind of process.env.CONTROL ? [process.env.CONTROL] : ['noSequence', 'shuffled', 'noAdaptation', 'lowNoise']) runV3Control(kind);
    if (!process.env.CONTROL || process.env.COLLECT_ONLY === '1') { const r = collectV3Controls(); return r.checks.every(c => c.ok) ? 0 : 2; }
  } else if (mode === 'v3-fig') return finalizeV3().outcome === 'passed' ? 0 : 2;
  else if (mode === 'v3-all') return runV3All().outcome === 'passed' ? 0 : 2;
  else throw new Error(`尚未实现的 v3 模式 ${mode}`);
  return 0;
}
