/** L4g v3：执行完整性、固定决策和学到的权重主测试。剂量与候选失败如实单列。
 * 部分成功退出 1；FAST 由 all.mjs 跳过。验证不训练、不修改旧缓存。
 */
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { pathToFileURL } from 'node:url';
import { ROOT, CONFIG, chainChecks, summarizeTrials, decodeRates } from '../demo/attractor-chain.mjs';
import { getV3, v3Checks, hash, weightHash, read, save, fileFor, candidateChecks, failureTypes } from '../demo/attractor-chain-v3.mjs';
import { candidateSettings } from '../demo/chain-v3-settings.mjs';
import { checker } from '../src/core/check.mjs';

const ARCHIVE = path.join(ROOT, 'experiments', 'logs', '2026-10-09-attractor-chain-v3');
const sha = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex').toUpperCase();
export function verifyL4Chain() {
  const r = getV3(), checks = v3Checks(r), c = checker('第 4 层（脉冲版）有向通道演化链 v3');
  for (const item of checks) c.check(`${item.id} ${item.name}`, item.ok, JSON.stringify(item.actual));
  for (const dose of r.dose.results) {
    c.note(`诊断 k=${dose.k}（人工放大结果不作为学到权重的成功证据）：静息点燃 ${dose.rest.ignitionRate}`);
    for (const item of dose.checks) console.log(`${item.ok ? '✓' : '✗'} 诊断 k=${dose.k} ${item.id} ${item.name}: ${JSON.stringify(item.actual)}`);
    console.log(`  · 失败类型：${JSON.stringify(dose.failures)}`);
    const expectedOk = dose.status !== 'uncontrolled' && dose.finished && dose.trials.length === 40 && dose.rest.ignitionRate <= 0.1 && chainChecks(dose.summary).every(p => p.ok);
    let consistent = expectedOk === dose.ok;
    if (dose.trials.length) {
      consistent &&= hash(summarizeTrials(dose.trials)) === hash(dose.summary) && hash(failureTypes(dose)) === hash(dose.failures);
      consistent &&= dose.trials.every(t => hash(decodeRates(t.bins, dose.threshold.threshold)) === hash(t.decoded));
    }
    c.check(`D2.k${dose.k}.data 剂量判据、解码和失败统计与原始分箱一致`, consistent, `ok=${dose.ok}, valid=${dose.trials.length}`);
  }
  for (const candidate of r.learning.candidates) {
    c.note(`候选 ${candidate.candidate}：E_f=${candidate.stats.Ef}, E_b=${candidate.stats.Eb}, 可用=${candidate.usable}`);
    for (const item of candidate.checks) console.log(`${item.ok ? '✓' : '✗'} 候选 ${candidate.candidate} ${item.name}: ${JSON.stringify(item.actual)}`);
    try {
      const state = read(fileFor(`seq-${candidate.candidate}-160-40`));
      const classes = read(fileFor(`rate-classes-${candidate.candidate}`)), rest = read(fileFor(`rate-rest-${candidate.candidate}`));
      const source = read(path.join(ROOT, 'cache', 'chain-v2-attractors.json'));
      const settings = candidateSettings(candidate.candidate);
      const expectedSource = hash({ weights: weightHash(source), learnGain: 0.6, settings, itemMs: 40, matchedPaths: null });
      const actual = candidateChecks(state, classes, rest);
      c.check(`L1.${candidate.candidate}.data 训练来源、设置与复查一致`, state.sourceHash === expectedSource && state.configHash === hash(CONFIG)
        && hash(state.v3Settings) === hash(settings) && hash(actual) === hash(candidate.checks), candidate.candidate);
    } catch (error) { c.check(`L1.${candidate.candidate}.data 训练来源、设置与复查一致`, false, error.message); }
  }
  if (r.stopReason) c.note(r.stopReason);
  try {
    const names = execFileSync('git', ['diff', '--name-only', 'attempt/2/attractor-chain-v2', '--', 'src'], { cwd: ROOT, encoding: 'utf8' }).trim();
    const untracked = execFileSync('git', ['ls-files', '--others', '--exclude-standard', '--', 'src'], { cwd: ROOT, encoding: 'utf8' }).trim();
    c.check('V.src 相对 v2 标签没有任何 src 改动', !names && !untracked, names || untracked || '无改动');
    const original = execFileSync('git', ['show', 'attempt/2/attractor-chain-v2:demo/attractor-chain.mjs'], { cwd: ROOT, encoding: 'utf8' });
    const current = fs.readFileSync(path.join(ROOT, 'demo', 'attractor-chain.mjs'), 'utf8');
    const configText = s => s.match(/export const CONFIG = Object.freeze\(\{[\s\S]*?\n\}\);/)[0].replace(/\r/g, '');
    c.check('V.config CONFIG 对象保持 v2 原样', CONFIG.version === 2 && configText(current) === configText(original), `version=${CONFIG.version}`);
  } catch (error) { c.check('V.src/config 原始源码与配置可验证', false, error.message); }
  let audit;
  try {
    const baseline = read(path.join(ARCHIVE, 'cache-before.json'));
    const entries = baseline.map(before => {
      const file = path.join(ROOT, 'cache', before.name), exists = fs.existsSync(file);
      const after = exists ? sha(file) : null;
      return { name: before.name, before: before.sha256, after, unchanged: before.sha256 === after };
    });
    audit = { checkedUtc: new Date().toISOString(), expectedCount: 74, baselineCount: baseline.length, unchanged: entries.filter(e => e.unchanged).length, entries };
    save(path.join(ARCHIVE, 'cache-integrity.json'), audit);
    c.check('V.cache 原有 74 个缓存 SHA-256 全部不变', baseline.length === 74 && audit.unchanged === 74, `${audit.unchanged}/${baseline.length}`);
    const old = new Set(baseline.map(b => b.name)), added = fs.readdirSync(path.join(ROOT, 'cache')).filter(name => name.endsWith('.json') && !old.has(name));
    c.check('V.prefix 新缓存全部使用 chain-v3- 前缀', added.every(name => name.startsWith('chain-v3-')), `${added.length} 个新缓存`);
  } catch (error) { c.check('V.cache 原有 74 个缓存 SHA-256 全部不变', false, error.message); }
  return { ok: c.done(), r, checks, audit };
}
if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) process.exitCode = verifyL4Chain().ok ? 0 : 1;
