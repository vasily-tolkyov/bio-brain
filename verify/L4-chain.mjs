/** L4g：有向通道演化链，完整列出步骤 1–5 的通过、失败和缺失项。
 * 有链缓存时读取其验收数据；无吸引子缓存时按阶段门槛先训练。
 * FAST=1 由 all.mjs 跳过本实验。部分成功保留非零退出码。
 */
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { execFileSync } from 'node:child_process';
import crypto from 'node:crypto';
import { ROOT, readResults, environmentConfig, acceptanceChecks, runExperiment } from '../demo/attractor-chain.mjs';
import { checker } from '../src/core/check.mjs';

export function verifyL4Chain() {
  const cfg = environmentConfig(), suffix = cfg.pIn === 0.1 ? '' : '-pin025';
  let r = readResults();
  const attractorFile = path.join(ROOT, 'cache', `chain-attractors${suffix}.json`);
  if (!fs.existsSync(attractorFile)) r = runExperiment(cfg);
  else {
    const state = JSON.parse(fs.readFileSync(attractorFile, 'utf8'));
    const configHash = crypto.createHash('sha256').update(JSON.stringify(cfg)).digest('hex');
    if (state.configHash !== configHash) throw new Error('吸引子缓存与所选配置不符');
    r = { ...r, attractors: state.measurements };
  }
  const c = checker('第 4 层（脉冲版）有向通道演化链');
  const checks = acceptanceChecks(r);
  for (const item of checks) c.check(`${item.id} ${item.name}`, item.ok, JSON.stringify(item.actual));
  if (r.stopReason) c.note(r.stopReason);
  try {
    const diff = execFileSync('git', ['diff', '--stat', 'stage/2-spiking-network', '--', 'src'], { cwd: ROOT, encoding: 'utf8' }).trim();
    c.check('6.3 共享 src 无改动', diff === '', diff || '与 stage/2-spiking-network 一致');
  } catch (error) { c.check('6.3 共享 src 无改动', false, error.message); }
  return { ok: c.done(), r, checks };
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) process.exitCode = verifyL4Chain().ok ? 0 : 1;
