/** 网络层规则验收。MODE=classes/direction/rest，RULE=GB_RATE/GB_NMDA/GB_NMDA_ETA0。
 * 每个工作进程只写独立缓存；MODE=collect 聚合九项结果，MODE=all 无缓存时全部重做。
 * 复现诊断网络的 pIn=pOut=0.1、背景 30/12 Hz、网络/噪声/抽样种子 1/1001/3。
 */
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { pathToFileURL } from 'node:url';
import { ROOT, CONFIG, RULES, buildChain, mean, resumableRng } from './attractor-chain.mjs';

export const RATE_MODES = ['classes', 'direction', 'rest'];
export const RATE_RULES = ['GB_RATE', 'GB_NMDA', 'GB_NMDA_ETA0'];
const fileFor = (mode, rule) => path.join(ROOT, 'cache', `rate-rule-${mode}-${rule.toLowerCase()}.json`);
const save = (file, r) => { fs.mkdirSync(path.dirname(file), { recursive: true }); fs.writeFileSync(`${file}.tmp`, JSON.stringify(r, null, 2) + '\n'); fs.renameSync(`${file}.tmp`, file); };
const clear = S => { for (const s of S.net.spikes) s.length = 0; };
const group = (i, cfg) => i < cfg.K * cfg.M ? Math.floor(i / cfg.M) : -1;
const proto = (g, cfg) => Array.from({ length: cfg.M }, (_, k) => g * cfg.M + k);
const frequency = (S, ids, start, ms) => mean(ids.map(i => S.net.spikes[i].filter(t => t >= start && t < start + ms).length)) * 1000 / ms;
const run = (S, ms, { ids = [], hz = S.cfg.stim, learn = false } = {}) => {
  const rates = new Float64Array(S.net.N); rates.fill(S.cfg.bg, 0, S.cfg.NE); rates.fill(S.cfg.bgI, S.cfg.NE);
  for (const i of ids) rates[i] = hz;
  S.net.run(ms, { rate: i => rates[i], learn });
};
const stats = S => {
  const acc = { within: [0, 0], between: [0, 0], ungrouped: [0, 0] };
  for (let k = 0; k < S.EE.rho.length; k++) {
    const a = group(S.EE.pre[k], S.cfg), z = group(S.EE.post[k], S.cfg), key = a < 0 || z < 0 ? 'ungrouped' : a === z ? 'within' : 'between';
    acc[key][0] += S.EE.rho[k]; acc[key][1]++;
  }
  return Object.fromEntries(Object.entries(acc).map(([key, [sum, n]]) => [key, sum / n]));
};
const deltas = (S, initial, classify) => {
  const acc = {};
  for (let k = 0; k < S.EE.rho.length; k++) {
    const name = classify(k); const a = acc[name] ??= { sum: 0, n: 0, up: 0 };
    const d = S.EE.rho[k] - initial[k]; a.sum += d; a.n++; if (d > 0.05) a.up++;
  }
  return Object.fromEntries(Object.entries(acc).map(([name, a]) => [name, { meanDelta: a.sum / a.n, n: a.n, fracUp: a.up / a.n }]));
};

export function runRateCase(mode, rule) {
  if (!RATE_MODES.includes(mode) || !RATE_RULES.includes(rule)) throw new Error('规则或模式不在预定九项测试中');
  const cfg = { ...CONFIG, pIn: 0.1, pOut: 0.1, bg: 30, bgI: 12, initialGain: 1, rule };
  const fingerprint = crypto.createHash('sha256').update(JSON.stringify({ version: 1, cfg, plasticity: RULES[rule] })).digest('hex');
  const file = fileFor(mode, rule);
  if (fs.existsSync(file)) { const r = JSON.parse(fs.readFileSync(file, 'utf8')); if (r.fingerprint !== fingerprint) throw new Error('规则验收缓存配置不符'); return r; }
  const r = { mode, rule, cfg, plasticity: RULES[rule], fingerprint }, t0 = Date.now();
  if (mode === 'classes') {
    for (const budgetOn of [true, false]) {
      const S = buildChain(cfg), rng = resumableRng(3), rates = { A: [], outside: [] };
      if (!budgetOn) S.EE.plastic.budget = undefined;
      run(S, 500); const initial = S.EE.rho.slice();
      for (let n = 0; n < 20; n++) {
        const A = proto(0, cfg), outside = S.E.filter(i => group(i, cfg) !== 0);
        const ids = [...rng.shuffle(A.slice()).slice(0, 64), ...rng.shuffle(outside.slice()).slice(0, 16)], stimulated = new Set(ids);
        clear(S); const start = S.net.t;
        S.EE.frozen = false; run(S, 200, { ids, learn: true });
        rates.A.push(frequency(S, A.filter(i => stimulated.has(i)), start, 200));
        rates.outside.push(frequency(S, outside.filter(i => !stimulated.has(i)), start, 200));
        S.EE.frozen = true; S.EE.z.fill(0); run(S, 300, { learn: true });
        if (n % 5 === 4) console.log(`${rule} classes budget=${budgetOn}: ${n + 1}/20`);
      }
      const classes = deltas(S, initial, k => {
        const a = group(S.EE.pre[k], cfg) === 0, z = group(S.EE.post[k], cfg) === 0;
        return a && z ? 'AtoA' : a ? 'AtoOut' : z ? 'OutToA' : 'OutToOut';
      });
      r[budgetOn ? 'budgetOn' : 'budgetOff'] = { classes, rateA: mean(rates.A), rateOutside: mean(rates.outside) };
    }
  } else if (mode === 'direction') {
    const S = buildChain(cfg); run(S, 500); const initial = S.EE.rho.slice();
    for (let n = 0; n < 40; n++) {
      run(S, 40, { ids: proto(0, cfg), hz: 58, learn: true });
      run(S, 40, { ids: proto(1, cfg), hz: 58, learn: true });
      run(S, 400, { learn: true }); clear(S);
      if (n % 5 === 4) console.log(`${rule} direction: ${n + 1}/40`);
    }
    r.classes = deltas(S, initial, k => {
      const a = group(S.EE.pre[k], cfg), z = group(S.EE.post[k], cfg), ab = g => g === 0 || g === 1;
      if (a === 0 && z === 1) return 'AtoB'; if (a === 1 && z === 0) return 'BtoA';
      if (a === 0 && z === 0) return 'AtoA'; if (a === 1 && z === 1) return 'BtoB';
      return ab(a) && !ab(z) ? 'ABtoOut' : !ab(a) && ab(z) ? 'OutToAB' : 'OutToOut';
    });
    r.direction = Object.fromEntries(Object.entries(r.classes).map(([k, v]) => [k, v.meanDelta]));
  } else {
    const S = buildChain(cfg); run(S, 500); r.trace = [{ seconds: 0, ...stats(S) }];
    for (let k = 1; k <= 6; k++) {
      clear(S); const start = S.net.t; run(S, 5000, { learn: true });
      r.trace.push({ seconds: k * 5, rE: frequency(S, S.E, start, 5000), rI: frequency(S, S.I, start, 5000), ...stats(S) });
      console.log(`${rule} rest: ${k * 5}/30 s，${JSON.stringify(r.trace.at(-1))}`);
    }
  }
  r.seconds = (Date.now() - t0) / 1000; save(file, r); return r;
}

export function rateRuleChecks(results) {
  const c = results.cases?.GB_RATE?.classes?.budgetOn?.classes, d = results.cases?.GB_RATE?.direction?.direction;
  const trace = results.cases?.GB_RATE?.rest?.trace, first = trace?.[0], last = trace?.at(-1);
  const restDelta = last && first ? Object.fromEntries(['within', 'between', 'ungrouped'].map(k => [k, last[k] - first[k]])) : null;
  return [
    { id: 'R3.1', name: '四类：组内 ≥ +0.15，输出/输入非特异增强 ≤ 10%/20%，组外 ≤ +0.01',
      ok: !!c && c.AtoA.meanDelta >= 0.15 && c.AtoOut.meanDelta <= 0.1 * c.AtoA.meanDelta && c.OutToA.meanDelta <= 0.2 * c.AtoA.meanDelta && c.OutToOut.meanDelta <= 0.01, actual: c ?? '未运行' },
    { id: 'R3.2', name: '方向差 ≥ 0.03，三个未经历类别平均 Δρ 各 ≤ +0.01',
      ok: !!d && d.AtoB - d.BtoA >= 0.03 && ['ABtoOut', 'OutToAB', 'OutToOut'].every(k => d[k] <= 0.01), actual: d ?? '未运行' },
    { id: 'R3.3', name: '静息 30 s：各类 Δρ ≤ +0.005，组内/组间差 ≤ 0.005',
      ok: !!restDelta && Object.values(restDelta).every(v => v <= 0.005) && Math.abs(restDelta.within - restDelta.between) <= 0.005, actual: restDelta ?? '未运行' },
    { id: 'R3.0', name: '三种规则的九项测试全部保存', ok: RATE_RULES.every(rule => RATE_MODES.every(mode => !!results.cases?.[rule]?.[mode])), actual: results.cases ? Object.values(results.cases).reduce((n, a) => n + Object.keys(a).length, 0) : 0 },
  ];
}
export function collectRateResults() {
  const r = { schema: 1, config: { ...CONFIG, pIn: 0.1, pOut: 0.1 }, cases: {} };
  for (const rule of RATE_RULES) {
    r.cases[rule] = {};
    for (const mode of RATE_MODES) if (fs.existsSync(fileFor(mode, rule))) r.cases[rule][mode] = JSON.parse(fs.readFileSync(fileFor(mode, rule), 'utf8'));
  }
  r.checks = rateRuleChecks(r); r.ok = r.checks.every(c => c.ok);
  const old = r.cases.GB_NMDA;
  r.referenceControls = { AtoOutFraction: old.classes ? old.classes.budgetOn.classes.AtoOut.meanDelta / old.classes.budgetOn.classes.AtoA.meanDelta : null,
    restAt5s: old.rest?.trace?.[1] ?? null, gate: false };
  save(path.join(ROOT, 'figures', 'rate-rule-results.json'), r); return r;
}
export function ensureRateResults() {
  for (const rule of RATE_RULES) for (const mode of RATE_MODES) runRateCase(mode, rule);
  return collectRateResults();
}
if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  const mode = process.env.MODE ?? 'all';
  try {
    const r = mode === 'all' ? ensureRateResults() : mode === 'collect' ? collectRateResults() : runRateCase(mode, process.env.RULE ?? 'GB_RATE');
    if (r.checks) for (const c of r.checks) console.log(`${c.ok ? '✓' : '✗'} ${c.id} ${c.name}: ${JSON.stringify(c.actual)}`);
    process.exitCode = r.ok === false ? 2 : 0;
  } catch (e) { console.error(e.stack); process.exitCode = 1; }
}
