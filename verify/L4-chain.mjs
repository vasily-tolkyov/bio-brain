/** L4g：有向通道演化链 v2，完整列出 R1–R3 与 S1–S5 的通过、失败和缺失项。
 * 有链缓存时读取其验收数据；无吸引子缓存时按阶段门槛先训练。
 * FAST=1 由 all.mjs 跳过本实验。部分成功保留非零退出码。
 */
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { execFileSync } from 'node:child_process';
import crypto from 'node:crypto';
import { ROOT, cacheFile, readResults, environmentConfig, acceptanceChecks, runExperiment } from '../demo/attractor-chain.mjs';
import { checker } from '../src/core/check.mjs';

export function verifyL4Chain({ trainMissing = true } = {}) {
  const cfg = environmentConfig();
  let r = readResults();
  const attractorFile = cacheFile('attractors', cfg);
  if (!fs.existsSync(attractorFile) && !r.stopReason && trainMissing) r = runExperiment(cfg);
  if (fs.existsSync(attractorFile)) {
    const state = JSON.parse(fs.readFileSync(attractorFile, 'utf8'));
    const configHash = crypto.createHash('sha256').update(JSON.stringify(cfg)).digest('hex');
    if (state.configHash !== configHash) throw new Error('吸引子缓存与所选配置不符');
    r = { ...r, attractors: state.measurements };
  }
  const rateFile = path.join(ROOT, 'figures', 'rate-rule-results.json');
  if (fs.existsSync(rateFile)) r.ruleAcceptance = JSON.parse(fs.readFileSync(rateFile, 'utf8'));
  const sequenceFile = cacheFile('main', cfg);
  if (fs.existsSync(sequenceFile)) {
    const state = JSON.parse(fs.readFileSync(sequenceFile, 'utf8'));
    r.sequence = state.measurements;
  }
  const c = checker('第 4 层（脉冲版）有向通道演化链 v2');
  const checks = acceptanceChecks(r);
  for (const item of checks) c.check(`${item.id} ${item.name}`, item.ok, JSON.stringify(item.actual));
  if (r.stopReason) c.note(r.stopReason);
  try {
    const bases = ['refs/heads/stage/2-spiking-network', 'refs/remotes/origin/stage/2-spiking-network'];
    const base = bases.find(ref => {
      try { execFileSync('git', ['rev-parse', '--verify', ref], { cwd: ROOT, stdio: 'ignore' }); return true; }
      catch { return false; }
    });
    if (!base) throw new Error('缺少 stage/2-spiking-network 或 origin/stage/2-spiking-network 作为 src 比较基线');
    const names = execFileSync('git', ['diff', '--name-only', base, '--', 'src'], { cwd: ROOT, encoding: 'utf8' }).trim().split(/\r?\n/).filter(Boolean);
    const allowed = ['src/L3-plasticity/calcium-rule.mjs', 'src/L4-network/spiking-network.mjs'];
    c.check('6.3 共享 src 改动仅限获准的两个文件', names.every(n => allowed.includes(n)), names.join(', ') || '与集成分支一致');
  } catch (error) { c.check('6.3 共享 src 改动仅限获准的两个文件', false, error.message); }
  return { ok: c.done(), r, checks };
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) process.exitCode = verifyL4Chain().ok ? 0 : 1;
