/** L4i：慢受体、ATP 与 JSON 完整续算；FAST 包含本组。 */
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import assert from 'node:assert/strict';
import {SpikingNetwork} from '../src/L4-network/spiking-network.mjs';
import {GB_RATE} from '../src/L3-plasticity/calcium-rule.mjs';
import {RECEPTORS} from '../src/L2-synapse/synapse.mjs';
import {ATP_PER_NA,ATP_PER_CA,ionsFromFemtoCoulomb} from '../src/core/units.mjs';
import {checker} from '../src/core/check.mjs';
import {runBenchmarks,compareBenchmarks} from './bench-spiking.mjs';
import {buildE1,captureRuntime,restoreRuntime} from '../src/loop/e1.mjs';
const ROOT=fileURLToPath(new URL('../',import.meta.url));
const zeroCosts={atpPerSpike:0,atpRestPerMs:0};
const peak=(r,d)=>r*d/(d-r)*Math.log(d/r),norm=(r,d)=>1/(Math.exp(-peak(r,d)/d)-Math.exp(-peak(r,d)/r));
const relative=(a,b)=>Math.abs(a-b)/Math.max(Math.abs(b),Number.MIN_VALUE);
function pulseNetwork(params){
  const net=new SpikingNetwork({costs:zeroCosts,pops:[{name:'pre',n:1,p:{C:100,b:0}},{name:'post',n:1,p:{C:Infinity,Vth:Infinity,b:0}}]});
  for(const p of params)net.connect({pairs:[[0,1]],kind:'slow',contacts:1,pRelease:1,scale:1,...p});
  net.run(.1,{extra:()=>[20000,0],learn:false});return net;
}
function jointNetwork(){
  const net=new SpikingNetwork({seed:5,costs:{atpPerSpike:1000,atpRestPerMs:100},plasticity:GB_RATE,
    pops:[{name:'E',n:3,p:{b:20,tauW:300}},{name:'I',n:2,p:{C:100,tRef:1}},{name:'A',n:2,apical:true}]});
  net.connect({from:'E',to:'E',p:1,contacts:10,plastic:{sMin:.1,sMax:9,rho0:.2,accel:5,budget:.2,tauN:30}});
  net.connect({from:'I',to:'E',p:1,kind:'inhFast',scale:2,istdp:{tauPre:200,eta:.02,target:3,wMax:40}});
  net.connect({from:'E',to:'I',p:1,scale:1});
  net.connect({from:'E',to:'E',p:1,kind:'slow',scale:.1,tauR:10,tauD:30});
  net.connect({from:'E',to:'A',p:1,kind:'slow',target:'apical',scale:.002,tauR:9200,tauD:18500});
  net.connect({from:'E',to:'A',p:1,kind:'nmda',target:'apical',scale:.5});
  net.groups[3].R[6]=.7; // 未参与释放但外部预置过的资源也必须逐位恢复。
  net.rng.normal();return net;
}
export function verifyL4Slow({write=true}={}){
  const c=checker('第 4 层 慢受体与完整续算'),metrics={};
  const reference=JSON.parse(fs.readFileSync(path.join(ROOT,'experiments/logs/2026-10-10-loop-w0-perf/bench-after.json'),'utf8'));
  const current=runBenchmarks(['e1','nmda','predictive']);
  const before={...reference,cases:Object.fromEntries(['e1','nmda','predictive'].map(name=>[name,reference.cases[name]]))};
  metrics.noSlow=compareBenchmarks(before,current);
  c.check('不用慢受体：三种 W0 配置的放电与全部状态逐位相同，ATP 误差 ≤ 1e-12',metrics.noSlow.ok,JSON.stringify(metrics.noSlow.comparisons.map(x=>({name:x.name,stateIdentical:x.stateIdentical,atpRelativeErrors:x.atpRelativeErrors}))));
  const analytical=[];
  for(const [tauR,tauD] of [[10,30],[9200,18500]]){
    const net=pulseNetwork([{tauR,tauD}]),g=net.groups[0],spike=net.spikes[0][0],rows=[];
    const amplitude=RECEPTORS.AMPA.q*norm(tauR,tauD);
    for(const factor of [.05,.1,.25,.5,.75,1,1.5,2,3,4]){
      const target=spike+factor*peak(tauR,tauD);
      if(target>net.t)net.run(Math.ceil((target-net.t)/net.dt)*net.dt,{learn:false});
      const elapsed=net.t-spike,actual=Math.max(0,g.slow.a[1]-g.slow.b[1]),expected=amplitude*(Math.exp(-elapsed/tauD)-Math.exp(-elapsed/tauR));
      rows.push({elapsed,actual,expected,relativeError:relative(actual,expected)});
    }
    analytical.push({tauR,tauD,samples:rows,maxRelativeError:Math.max(...rows.map(row=>row.relativeError)),preSpikes:net.spikes[0].length});
  }
  metrics.analytical=analytical;
  c.check('10/30 ms 与 9.2/18.5 s：各十个采样时刻符合解析双指数，相对误差 ≤ 1e-6',analytical.every(x=>x.preSpikes===1&&x.maxRelativeError<=1e-6),JSON.stringify(analytical.map(x=>({tauR:x.tauR,tauD:x.tauD,maxRelativeError:x.maxRelativeError}))));
  const p1={tauR:10,tauD:30,scale:.7},p2={tauR:20,tauD:70,scale:.3},one=pulseNetwork([p1]),two=pulseNetwork([p2]),both=pulseNetwork([p1,p2]);
  let superpositionError=0;
  for(let k=0;k<1000;k++){
    for(const net of [one,two,both])net.run(net.dt,{learn:false});
    const conductance=net=>net.groups.reduce((sum,g)=>sum+Math.max(0,g.slow.a[1]-g.slow.b[1]),0);
    superpositionError=Math.max(superpositionError,relative(conductance(both),conductance(one)+conductance(two)));
  }
  metrics.superposition={maxRelativeError:superpositionError,atpRelativeError:relative(both.ledger.get('syn_post'),one.ledger.get('syn_post')+two.ledger.get('syn_post'))};
  c.check('同一突触后群两组慢受体线性叠加，电导及电荷误差 ≤ 1e-9',Object.values(metrics.superposition).every(x=>x<=1e-9),JSON.stringify(metrics.superposition));
  const energy=pulseNetwork([{tauR:10,tauD:30}]);energy.run(300,{learn:false});
  const g=energy.groups[0],m=Math.round((energy.t-energy.spikes[0][0])/energy.dt),dr=Math.exp(-energy.dt/g.slow.tauR),dd=Math.exp(-energy.dt/g.slow.tauD);
  const charge=65*energy.dt*RECEPTORS.AMPA.q*g.slow.k*(dd*(1-dd**m)/(1-dd)-dr*(1-dr**m)/(1-dr));
  const expected=ionsFromFemtoCoulomb(charge)*(RECEPTORS.AMPA.na*ATP_PER_NA+RECEPTORS.AMPA.ca/2*ATP_PER_CA),actual=energy.ledger.get('syn_post');
  metrics.energy={steps:m,charge,expected,actual,relativeError:relative(actual,expected)};
  c.check('慢电流 ATP 等于离散解析积分电荷按 AMPA 比例换算，相对误差 ≤ 1e-9',metrics.energy.relativeError<=1e-9,JSON.stringify(metrics.energy));
  const continuous=jointNetwork(),first=jointNetwork(),rate=i=>i<3?30:i<5?12:8,extra=t=>[t%100<2?1200:0,0,0,0,0,0,0];
  const onStep=()=>{};
  continuous.run(2000,{learn:true,rate,extra,onStep});first.run(1000,{learn:true,rate,extra,onStep,onStepWritesResources:false});
  const serialized=JSON.stringify(first.exportState()),restored=jointNetwork();restored.importState(JSON.parse(serialized));restored.run(1000,{learn:true,rate,extra,onStep,onStepWritesResources:false});
  let jointIdentical=true,error=null;try{assert.deepEqual(restored.exportState(),continuous.exportState());assert.equal(restored.rng.normal(),continuous.rng.normal());}catch(e){jointIdentical=false;error=e.message;}
  metrics.resume={identical:jointIdentical,error,bytes:Buffer.byteLength(serialized),neurons:first.N,groups:first.groups.length};
  c.check('慢受体 + GB_RATE + ISTDP + 顶树突：1 s 导出 / JSON 导入再 1 s 与连续 2 s 逐位相同',jointIdentical,error??JSON.stringify(metrics.resume));
  const a=buildE1(),b=buildE1();a.net.run(10,{rate:i=>i<a.cfg.NE?30:12});const state=captureRuntime(a);restoreRuntime(b,JSON.parse(JSON.stringify(state)));
  let e1Identical=true;try{assert.deepEqual(a.net.exportState(),b.net.exportState());}catch{e1Identical=false;}
  c.check('E1 运行时工具调用完整状态接口，JSON 往返保留全部状态',e1Identical);
  const ok=c.done(),result={schema:1,package:'w2-slow',createdAtUtc:new Date().toISOString(),ok,metrics};
  if(write){const dir=path.join(ROOT,'experiments/logs/2026-10-10-loop-w2-slow');fs.mkdirSync(dir,{recursive:true});fs.writeFileSync(path.join(dir,'slow-results.json'),JSON.stringify(result,null,2)+'\n');}
  return {ok,result};
}
if(process.argv[1]&&import.meta.url===pathToFileURL(path.resolve(process.argv[1])).href)process.exit(verifyL4Slow().ok?0:1);
