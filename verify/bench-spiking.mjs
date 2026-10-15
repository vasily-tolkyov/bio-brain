/** SpikingNetwork 的性能基准、逐位状态摘要与 ATP 对照。 */
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { performance } from 'node:perf_hooks';
import { buildE1, E1_DEFAULTS } from '../src/loop/e1.mjs';
import { buildNetwork, CFG } from '../demo/fig2-network.mjs';
import { PC2 } from '../demo/predictive-hierarchy.mjs';
import { buildLayered, inputRate } from '../src/L4-network/layered-network.mjs';

const ROOT = fileURLToPath(new URL('../', import.meta.url));
const hash = value => crypto.createHash('sha256').update(value).digest('hex');
const vectorHash = value => hash(Buffer.from(value.buffer, value.byteOffset, value.byteLength));
export function summarizeState(net) {
  const vectors = {};
  for (const key of ['V','w','ref','Rin','x','y','y2','nmda','xs','ys','xsl','Vd','wd','bapUntil'])
    if (net[key]) vectors[key] = vectorHash(net[key]);
  for (const key of ['ra','rb','ea','eb','dra','drb'])
    if (net[key]) net[key].forEach((value, i) => vectors[key + '.' + i] = vectorHash(value));
  net.groups.forEach((group, i) => {
    for (const key of ['R','rho','scale','z','nm'])
      if (group[key]) vectors['groups.' + i + '.' + key] = vectorHash(group[key]);
  });
  return { t: net.t, vectors, spikes: hash(JSON.stringify(net.spikes)),
    pending: vectorHash(Float64Array.from(net.pending)), pendHead: net.pendHead,
    frozen: net.groups.map(g => g.frozen ?? false), atp: Object.fromEntries(net.ledger.atp) };
}
export function makeBenchmark(name) {
  if (name === 'e1') {
    const S = buildE1();
    S.EE.frozen = false;
    return { net:S.net, milliseconds:2000, run:() => S.net.run(2000,{learn:true,rate:i => i < S.cfg.NE ? S.cfg.bg : S.cfg.bgI}) };
  }
  if (name === 'emergence' || name === 'nmda') {
    const S = buildNetwork(CFG), ms = name === 'emergence' ? 2000 : 1000;
    return { net:S.net, milliseconds:ms, run:() => S.net.run(ms,{learn:true,rate:i => i < CFG.NE ? CFG.bg : 0}) };
  }
  if (name === 'predictive') {
    const S = buildLayered(2,PC2);
    return { net:S.net, milliseconds:500, run:() => S.net.run(500,{learn:true,rate:inputRate(S,new Map())}) };
  }
  throw new Error('Unknown benchmark ' + name);
}
export function runBenchmarks(names=['e1','emergence','nmda','predictive']) {
  const results = {};
  for (const name of names) {
    const job = makeBenchmark(name), t0 = performance.now(), cpu0 = process.cpuUsage();
    job.run();
    const seconds = (performance.now()-t0)/1000, cpu = process.cpuUsage(cpu0);
    results[name] = { milliseconds:job.milliseconds, neurons:job.net.N,
      synapses:job.net.groups.reduce((n,g)=>n+g.pre.length,0), wallSeconds:seconds,
      wallSecondsPerSimulatedSecond:seconds/(job.milliseconds/1000),
      cpuSeconds:(cpu.user+cpu.system)/1e6, state:summarizeState(job.net) };
    console.log(name + ': ' + seconds.toFixed(3) + ' wall s; ' + results[name].wallSecondsPerSimulatedSecond.toFixed(3) + ' wall s / simulated s');
  }
  return {schema:1,createdAtUtc:new Date().toISOString(),node:process.version, e1Defaults:E1_DEFAULTS,cases:results};
}
export function compareBenchmarks(before,after) {
  const comparisons = [];
  for (const name of Object.keys(before.cases)) {
    const a=before.cases[name],b=after.cases[name];
    if (!b) {comparisons.push({name,ok:false,error:'Missing benchmark'});continue;}
    const {atp:atpA,...stateA}=a.state, {atp:atpB,...stateB}=b.state;
    const stateIdentical=JSON.stringify(stateA)===JSON.stringify(stateB);
    const errors=Object.fromEntries([...new Set([...Object.keys(atpA),...Object.keys(atpB)])].map(key=>[key,Math.abs((atpA[key]??0)-(atpB[key]??0))/Math.max(Math.abs(atpA[key]??0),Number.MIN_VALUE)]));
    const atpOk=Object.values(errors).every(error=>error<=1e-12);
    comparisons.push({name,ok:stateIdentical&&atpOk,stateIdentical,atpRelativeErrors:errors,
      speedup:a.wallSeconds/b.wallSeconds, wallTimeReduction:1-b.wallSeconds/a.wallSeconds});
  }
  return {ok:comparisons.every(row=>row.ok),comparisons};
}
if (process.argv[1] && import.meta.url===pathToFileURL(path.resolve(process.argv[1])).href) {
  const names=process.env.BENCH_CASE ? process.env.BENCH_CASE.split(',') : undefined;
  const result=runBenchmarks(names);
  const out=process.env.BENCH_OUT ?? path.join(ROOT,'experiments','logs','2026-10-10-loop-w0-perf','bench-'+(process.env.PHASE??'before')+'.json');
  fs.mkdirSync(path.dirname(out),{recursive:true}); fs.writeFileSync(out,JSON.stringify(result,null,2)+'\n');
}
