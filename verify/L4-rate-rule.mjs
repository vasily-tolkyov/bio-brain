/** L4h：GB_RATE 的四类、方向和静息验收；原规则与 eta=0 只记录。 */
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { ensureRateResults } from '../demo/rate-rule.mjs';
import { checker } from '../src/core/check.mjs';

export function verifyL4RateRule() {
  const r = ensureRateResults(), c = checker('第 4 层（脉冲版）频率依赖规则');
  for (const row of r.checks) c.check(`${row.id} ${row.name}`, row.ok, JSON.stringify(row.actual));
  c.note(`原规则对照只记录：A→组外 / A→A = ${r.referenceControls.AtoOutFraction}；5 s 静息 ${JSON.stringify(r.referenceControls.restAt5s)}`);
  return { ok: c.done(), r };
}
if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) process.exitCode = verifyL4RateRule().ok ? 0 : 1;
