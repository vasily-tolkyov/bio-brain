/** W4-diag4：固定十四点；完整记录，先选机理配置，再选单通道 L 增益。 */
import fs from 'node:fs';
import path from 'node:path';
import {spawn,execFileSync} from 'node:child_process';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';
import {gzipSync,gunzipSync} from 'node:zlib';
import {buildLoop} from '../src/loop/build.mjs';
import {warmup,trainPredictions,readCheckpoint,writeCheckpoint,checkpointExists,measureRest,restLevels} from '../src/loop/protocol.mjs';
import {structure,validState,verifyUnits,verifyProtocolResume} from '../verify/loop-build.mjs';
import {binGroups,groupRate,meanPsth,poissonComparison,excessFraction,stateHash,connectionHash,loadEvents,boundaryDirection} from '../src/loop/readout.mjs';
import {E1_DEFAULTS,buildE1,run as runE1} from '../src/loop/e1.mjs';
import {mgBlock} from '../src/L2-synapse/synapse.mjs';
export const ROOT=fileURLToPath(new URL('../',import.meta.url));
export const DIR=path.join(ROOT,'experiments/logs/2026-10-14-loop-w4-diag4');
const CACHE=path.join(ROOT,'cache'),FROZEN={learn:false,e1Learn:false,prototypeLearn:false,tdLearn:false};
export const MULTIPLIERS=Object.freeze([.5,.75,1,1.5,2]);
export const RUNS=Object.freeze([
  ...[40,80].flatMap(relaySize=>[20,10,5].flatMap(e1Adaptation=>[2,4].map(localSEq=>({id:`s1-r${relaySize}-b${e1Adaptation}-e${localSEq}`,scene:1,relaySize,e1Adaptation,localSEq})))),
  ...[20,10].map(e1Adaptation=>({id:`s4-r40-b${e1Adaptation}-e4`,scene:4,relaySize:40,e1Adaptation,localSEq:4})),
]);
const sourceFiles=['demo/loop-diag.mjs','src/loop/build.mjs','src/loop/e1.mjs','src/loop/e2.mjs','src/loop/errors.mjs','src/loop/protocol.mjs','src/loop/readout.mjs','src/loop/context.mjs','src/loop/actions.mjs','src/loop/expect.mjs','verify/loop-build.mjs','verify/all.mjs','src/loop/world.mjs','verify/loop-world.mjs','src/L4-network/spiking-network.mjs'];
export const sourceHash=()=>stateHash(sourceFiles.map(name=>[name,fs.readFileSync(path.join(ROOT,name),'utf8')]));
const clone=value=>JSON.parse(JSON.stringify(value)),sum=values=>values.reduce((s,x)=>s+x,0),mean=a=>a.length?sum(a)/a.length:null;
const ratio=(value,reference)=>reference>0?value/reference:value===0?'undefined(0/0)':'Infinity';
const finitePass=(v,limit,lower=false)=>typeof v==='number'&&Number.isFinite(v)&&(lower?v>=limit:v<=limit);
const atomic=writeCheckpoint,read=readCheckpoint;
const fileFor=(id,suffix)=>path.join(CACHE,'loop-w4diag4-'+id+'-'+suffix+'.json');
const testGroups=S=>S.spec.scene===4?['红','甜','苦']:S.spec.features.slice();
function extraRun(c){return {id:`s4-r${c.relaySize}-b${c.e1Adaptation}-e${c.localSEq}-selected`,scene:4,relaySize:c.relaySize,e1Adaptation:c.e1Adaptation,localSEq:c.localSEq};}
function selected(id){
  const run=RUNS.find(r=>r.id===id);if(run)return run;
  const file=path.join(DIR,'summary.json');
  if(checkpointExists(file)){const s=read(file),r=s.entry.authorizedRuns.find(r=>r.id===id);if(r&&s.entry.initialRunsComplete)return r;}
  throw new Error('运行不属于固定网格或唯一场景四复核：'+id);
}
export function make(run){
  const S=buildLoop({scene:run.scene,seed:43,localTauR:25,localSEq:run.localSEq,localMEq:run.localSEq,relaySize:run.relaySize,e1Adaptation:run.e1Adaptation,loadGain:1,shadowL:[.5,.75,1.5,2],e2Recurrent:4.5});
  attachTrace(S);return S;
}
function assertFinite(S){
  if(!validState(S.net)||!Number.isFinite(S.net.t)||![...S.net.ledger.atp.values()].every(Number.isFinite)||S.net.spikes.some(row=>row.length&&!Number.isFinite(row.at(-1)))||!S.e2.W.every(row=>row.every(Number.isFinite)))throw Object.assign(new Error('非有限网络状态；停止诊断'),{code:'NONFINITE'});
}
/** 只读电导采样，保存为压缩证据；不改变电位、噪声或突触资源。 */
function attachTrace(S){
  const net=S.net;let active=null;
  const receptor=(ids,qs)=>{let value=0;for(const i of ids)for(const q of qs)value+=Math.max(0,net.ra[q][i]-net.rb[q][i])*(q===1?mgBlock(net.V[i]):1);return value/ids.length;};
  const recorder={onStepWritesResources:false,
    start(features,limitMs=Infinity){active={start:net.t,end:Number.isFinite(limitMs)?net.t+limitMs:null,times:[],features:Object.fromEntries(features.map(f=>[f,{gExcN:[],gInhN:[],gSlowM:[],gInhM:[]}]))};},
    onStep(t){
      if(!active||(active.end!==null&&t>=active.end-1e-8)||Math.round(t*10)%10!==0)return;
      active.times.push(Math.round(t));
      for(const [f,row] of Object.entries(active.features)){
        const N=S.errors.N[f],M=S.errors.M[f],slow=S.errors.handles.localNegative.find(g=>g.loopSlow.feature===f).slow;
        const values={gExcN:receptor(N,[0,1]),gInhN:receptor(N,[2,3]),gSlowM:sum(M.map(i=>Math.max(0,slow.a[i]-slow.b[i])))/M.length,gInhM:receptor(M,[2,3])};
        for(const [key,value] of Object.entries(values)){if(!Number.isFinite(value))throw Object.assign(new Error('电导记录出现非有限值'),{code:'NONFINITE'});row[key].push(+value.toFixed(4));}
      }
    },
    stop(){const result=active;active=null;return result;},exportState:()=>clone(active),importState:saved=>{active=clone(saved);},
  };
  S.hooks.add('diagnosticTrace',recorder);S.trace=recorder;
}
function saveTrace(id,key,data){
  const relative=`traces/${id}-${key}.json.gz`,file=path.join(DIR,relative),buffer=gzipSync(Buffer.from(JSON.stringify(data)),{level:1});
  fs.mkdirSync(path.dirname(file),{recursive:true});
  if(fs.existsSync(file)){if(!fs.readFileSync(file).equals(buffer))throw new Error('重复电导证据与恢复运行不一致：'+relative);}else fs.writeFileSync(file,buffer,{flag:'wx'});
  return {file:relative,sha256:createHash('sha256').update(buffer).digest('hex'),samples:data.times.length,start:data.start,end:data.times.at(-1)??data.start};
}
export function windowMean(bins,width,start,end){let total=0;for(let k=0;k<bins.length;k++)total+=bins[k]*Math.max(0,Math.min(end,(k+1)*width)-Math.max(start,k*width));return total/(end-start);}
export function windowPeak(bins,width,start,end){return Math.max(0,...bins.filter((_,k)=>(k+.5)*width>=start&&(k+.5)*width<=end));}
const loadGroups=(S,gain)=>gain===1?S.context.L:S.context.shadows[gain];
const loadName=(gain,key)=>gain===1?'L:'+key:'ShadowL:'+gain+':'+key;
function completeConnections(S){
  const pops=new Array(S.net.N);for(const [name,p] of Object.entries(S.net.pops))for(let i=p.start;i<p.start+p.n;i++)pops[i]=name;
  return S.net.groups.map((g,index)=>({index,route:g.loopRoute??null,load:g.loopLoad??null,prePopulations:[...new Set([...g.pre].map(i=>pops[i]))],postPopulations:[...new Set([...g.post].map(i=>pops[i]))],kind:g.kind,edges:g.pre.length,strengthMin:g.scale.length?g.scale.reduce((a,b)=>Math.min(a,b)):null,strengthMax:g.scale.length?g.scale.reduce((a,b)=>Math.max(a,b)):null,contacts:g.contacts,pRelease:g.pRelease,tauRec:g.tauRec,tauR:g.slow?.tauR??null,tauD:g.slow?.tauD??null,istdp:g.istdp?{...g.istdp}:null,istdpFrozen:!!g.frozen,e2Idle:g.e2Idle??null}));
}
function trialRest(id,feature,index){return 1000+1000*parseInt(stateHash({id,feature,index}).slice(0,8),16)/0x100000000;}
export function newProbe(S,feature,kind,baseline){return {feature,kind,baseline:clone(baseline),trials:[],loadApplicable:!!S.context.L[feature+':±']};}
export function probeTrial(S,probe,id,{trace=true}={}){
  const start=S.net.t,duration=probe.kind==='held'?10000:1000,restMs=trialRest(id,probe.feature,probe.trials.length),groups={N:S.errors.N[probe.feature],M:S.errors.M[probe.feature],S:S.errors.S[probe.feature]};
  if(trace)S.trace.start(testGroups(S));
  S.run(duration,{...FROZEN,features:[probe.feature]});const off=S.net.t;S.run(restMs,FROZEN);assertFinite(S);
  const end=S.net.t,bins=binGroups(S.net,groups,start,duration+1000,20),row={trial:probe.trials.length,start,off,end,duration,restMs,bins};
  if(trace)row.trace=saveTrace(id,`${S.spec.features.indexOf(probe.feature)}-${row.trial}`,S.trace.stop());
  probe.trials.push(row);return row;
}
export function probeMetrics(S,probe){
  const {feature,kind,trials,baseline}=probe,width=20,psth=Object.fromEntries(Object.keys(trials[0].bins).map(key=>[key,meanPsth(trials.map(r=>r.bins[key]))])),duration=trials[0].duration;
  const baseN=baseline.baseline['N:'+feature],baseM=baseline.baseline['M:'+feature],baseS=baseline.baseline['S:'+feature];
  const nPeak=windowPeak(psth.N,width,0,50),mPeak=windowPeak(psth.M,width,duration,duration+60),sMean=windowMean(psth.S,width,50,1000);
  const nm=Object.fromEntries([['N',baseN,nPeak],['M',baseM,mPeak]].map(([type,base,peak])=>{
    const result={peak,baseline:base.mu,peakExcess:peak-base.mu,requiredPeak:base.mu+10,peakOK:peak>=base.mu+10};
    if(kind==='held'){
      const hz=windowMean(psth[type],width,150,10000),residual=hz-base.mu,excess=excessFraction(psth[type].filter((_,k)=>(k+.5)*width>=150&&(k+.5)*width<10000),base),limit=.25*result.peakExcess;
      Object.assign(result,{rate:hz,residual,residualRatio:ratio(residual,result.peakExcess),residualLimit:limit,residualOK:residual<=limit,excess,peakReference:'same five held trials',ok:residual<=limit&&excess.ok});
    }
    return [type,result];
  }));
  return {feature,kind,trialCount:trials.length,binMs:width,N:nm.N,M:nm.M,S:{rate:sMean,baseline:baseS.mu,ratioToRest:ratio(sMean,baseS.mu),ok:sMean>=2*baseS.mu},psth};
}
function probeStep(S,ctx,key,kind,count){
  const rows=ctx.data[key]??=[],index=rows.filter(r=>r.done).length;if(index===ctx.features.length)return true;
  let row=rows[index];if(!row){row=newProbe(S,ctx.features[index],kind,(key==='D4before'?ctx.data.D3:ctx.data.D3prime).rest);rows.push(row);ctx.data[key]=rows;}
  probeTrial(S,row,ctx.id+'-'+key);if(row.trials.length===count){row.metrics=probeMetrics(S,row);row.done=true;}
  return rows.filter(r=>r.done).length===ctx.features.length;
}
/** 与探查相同：全局整数毫秒计数；分析窗取最近整数毫秒。 */
export function spikeSeries(net,ids){
  const counts=new Uint32Array(Math.ceil(net.t)+2);for(const i of ids)for(const t of net.spikes[i])counts[Math.floor(t)]++;
  const prefix=new Float64Array(counts.length+1);for(let i=0;i<counts.length;i++)prefix[i+1]=prefix[i]+counts[i];
  const count=(a,b)=>prefix[Math.min(counts.length,Math.max(0,Math.round(b)))]-prefix[Math.min(counts.length,Math.max(0,Math.round(a)))];
  return {counts,prefix,count,rate:(a,b)=>count(a,b)*1000/(ids.length*(b-a))};
}
export function volleyEvents(series,segments){
  const events=[];
  for(const [a,b] of segments){let last=-Infinity;for(let t=Math.round(a);t+5<=Math.round(b);t++){
    if(series.count(t,t+5)>=5&&t-last>20){let best=t,value=series.count(t,t+5);for(let u=t;u<Math.min(Math.round(b)-5,t+10);u++){const v=series.count(u,u+5);if(v>value){value=v;best=u;}}events.push(best);last=best;t=best+20;}
  }}
  return events;
}
function fano(series,segments){const counts=[];for(const [a,b] of segments)for(let t=Math.round(a);t+20<=Math.round(b);t+=20)counts.push(series.count(t,t+20));const mu=mean(counts);return mu>0?sum(counts.map(v=>(v-mu)**2))/(counts.length-1)/mu:'undefined(0/0)';}
function restSegments(data){
  const out=[data.D3,data.D3prime].map(r=>[r.rest.start,r.rest.end]);
  for(const key of ['D4before','D4after']){const trials=data[key].flatMap(p=>p.trials).sort((a,b)=>a.start-b.start);trials.forEach((r,k)=>out.push([r.off+300,trials[k+1]?.start??r.end]));}
  return out.filter(([a,b])=>b>a).map(r=>r.map(Math.round));
}
const segmentRate=(series,segments,n)=>sum(segments.map(([a,b])=>series.count(a,b)))*1000/(n*sum(segments.map(([a,b])=>b-a)));
const MECHANISM_LIMITS={restVolleyN:.5,restVolleyM:.5,presVolleyN:1,presVolleyM:1,restFanoN:2,restFanoM:2,mWaveRatio:1.5,nReboundRatio:1.5,e1OnsetMargin:2,e1OffsetMargin:1.5};
export function mechanismMetrics(S,data,features){
  const rests=restSegments(data),rows=[];
  for(const f of features){
    const event=data.D4after.find(p=>p.feature===f),held=data.D6after.find(p=>p.feature===f),trials=[...event.trials,...held.trials],pres=held.trials.map(r=>[Math.round(r.start)+300,Math.round(r.off)]);
    const N=spikeSeries(S.net,S.errors.N[f]),M=spikeSeries(S.net,S.errors.M[f]),E=spikeSeries(S.net,S.e1.featureIds[f]);
    const restN=volleyEvents(N,rests),restM=volleyEvents(M,rests),presN=volleyEvents(N,pres),presM=volleyEvents(M,pres),rs=sum(rests.map(([a,b])=>b-a))/1000,ps=sum(pres.map(([a,b])=>b-a))/1000;
    const stat=t=>E.rate(t,t+40)-E.rate(t-200,t),on=trials.map(r=>Math.max(...Array.from({length:40},(_,i)=>stat(Math.round(r.start)-10+i)))),off=trials.map(r=>Math.min(...Array.from({length:60},(_,i)=>stat(Math.round(r.off)-10+i))));
    let maxPositive=0,minNegative=0;for(const [a,b] of [...rests,...pres])for(let t=a+200;t+40<=b;t+=2){const v=stat(t);maxPositive=Math.max(maxPositive,v);minNegative=Math.min(minNegative,v);}
    const values={restVolleyN:restN.length/rs,restVolleyM:restM.length/rs,presVolleyN:presN.length/ps,presVolleyM:presM.length/ps,restFanoN:fano(N,rests),restFanoM:fano(M,rests),
      mWaveRatio:ratio(segmentRate(M,event.trials.map(r=>[Math.round(r.start)+100,Math.round(r.start)+300]),S.errors.M[f].length),segmentRate(M,held.trials.map(r=>[Math.round(r.start)+1000,Math.round(r.start)+5000]),S.errors.M[f].length)),
      nReboundRatio:ratio(segmentRate(N,event.trials.map(r=>[Math.round(r.off)+200,Math.round(r.off)+400]),S.errors.N[f].length),data.D3prime.rest.baseline['N:'+f].mu),
      e1OnsetMargin:ratio(Math.min(...on),maxPositive),e1OffsetMargin:ratio(Math.min(...off.map(Math.abs)),Math.abs(minNegative))};
    rows.push({feature:f,...values,passes:Object.fromEntries(Object.entries(values).map(([key,v])=>[key,finitePass(v,MECHANISM_LIMITS[key],key.startsWith('e1'))])),events:{restN,restM,presN,presM},segments:{rest:rests,pres},step:{on,off,maxPositive,minNegative,trialCount:trials.length,trialScope:'D4-after and D6; all frozen post-training trials'},rates:{restN:data.D3prime.rest.baseline['N:'+f].mu},upstream:{N:event.metrics.N,M:event.metrics.M,S:event.metrics.S,heldN:held.metrics.N,heldM:held.metrics.M}});
  }
  const metrics=Object.fromEntries(Object.entries(MECHANISM_LIMITS).map(([key,limit])=>{const values=rows.map(r=>({feature:r.feature,value:r[key],pass:r.passes[key]})),finite=values.every(v=>typeof v.value==='number'),worst=finite?(key.startsWith('e1')?Math.min(...values.map(v=>v.value)):Math.max(...values.map(v=>v.value))):null;return [key,{byFeature:values,worst,limit,pass:values.every(v=>v.pass)}];}));
  return {...metrics,features:rows,definition:'probe analyze.mjs/phases.mjs: 1 ms counts, sample-variance Fano, local-maximum 5 ms volley windows with >20 ms spacing'};
}
export function loadMetrics(S,data,features){
  return MULTIPLIERS.map(gain=>{
    const rows=[];
    for(const f of features){
      const key=f+':±',ids=loadGroups(S,gain)[key];if(!ids)continue;
      const events=loadEvents(ids.map(i=>S.net.spikes[i])),event=data.D4after.find(p=>p.feature===f),held=data.D6after.find(p=>p.feature===f);
      const inWindow=(a,b)=>events.filter(e=>e.time>=a-1e-8&&e.time<b-1e-8),trials=event.trials.map(r=>{
        const on=inWindow(r.start,r.start+100),off=inWindow(r.off,r.off+100),double=(t)=>{const es=inWindow(t,t+150);return es.slice(1).some(e=>e.time>=t+50-1e-8);};
        return {start:r.start,off:r.off,end:r.end,on,offset:off,directions:[...on.map(e=>({time:e.time,expected:'+',actual:boundaryDirection(S.net,S.e1.featureIds[f],e.time)})),...off.map(e=>({time:e.time,expected:'−',actual:boundaryDirection(S.net,S.e1.featureIds[f],e.time)}))],falsePresence:inWindow(r.start+150,r.off).length>0,falseRest:inWindow(r.off+150,r.end).length>0,doubleOn:double(r.start),doubleOff:double(r.off)};
      });
      const directions=trials.flatMap(t=>t.directions),fraction=fn=>trials.filter(fn).length/trials.length,base=data.D3prime.rest.baseline[loadName(gain,key)],heldSeconds=sum(held.trials.map(r=>(r.off-r.start-150)/1000)),heldHz=sum(held.trials.map(r=>groupRate(S.net,ids,r.start+150,r.off)*(r.off-r.start-150)))/(heldSeconds*1000),comparison=poissonComparison(heldHz,base.mu,ids.length,heldSeconds,3);
      rows.push({feature:f,gain,detectOn:fraction(t=>t.on.length>0),detectOff:fraction(t=>t.offset.length>0),direction:directions.length?directions.filter(d=>d.actual===d.expected).length/directions.length:0,directionCount:directions.length,falsePresence:fraction(t=>t.falsePresence),falseRest:fraction(t=>t.falseRest),double:sum(trials.map(t=>Number(t.doubleOn)+Number(t.doubleOff)))/(2*trials.length),heldAboveRest:heldHz-base.mu>comparison.limit,held:{hz:heldHz,baseline:base.mu,seconds:heldSeconds,comparison},lRest:[data.D3,data.D3prime].map(r=>r.rest.rates[loadName(gain,key)]),trials});
    }
    const minimum=key=>rows.length?Math.min(...rows.map(r=>r[key])):null,maximum=key=>rows.length?Math.max(...rows.map(r=>r[key])):null;
    return {gain,applicable:rows.length>0,detectOn:minimum('detectOn'),detectOff:minimum('detectOff'),direction:minimum('direction'),falsePresence:maximum('falsePresence'),falseRest:maximum('falseRest'),double:maximum('double'),heldAboveRest:rows.some(r=>r.heldAboveRest),lRest:rows.length?Math.max(...rows.flatMap(r=>r.lRest)):null,features:rows};
  });
}
export function e2SourceStep(S,rows){
  const mode=['normal','N-to-E2-zero','E1-to-E2-zero'][rows.length];if(!mode)return true;
  const affected=mode==='normal'?[]:S.e2.handles.paths.filter(r=>r.kind===(mode==='N-to-E2-zero'?'up':'recognition')).map(r=>r.group),saved=affected.map(g=>g.scale.slice()),start=S.net.t;
  try{affected.forEach(g=>g.scale.fill(0));S.run(10000,FROZEN);assertFinite(S);rows.push({mode,start,end:S.net.t,units:S.e2.labels.flatMap((label,k)=>{if(label===null)return [];const g=S.e2.handles.inhibition[k],weights=[...g.scale].sort((a,b)=>a-b),mid=weights.length/2;return [{k,label,hz:groupRate(S.net,S.e2.units[k],start,S.net.t),inhibition:{mean:mean(weights),median:weights.length%2?weights[Math.floor(mid)]:(weights[mid-1]+weights[mid])/2,nearMaxFraction:weights.filter(w=>w>=.95*g.istdp.wMax).length/weights.length,wMax:g.istdp.wMax}}];})});}
  finally{affected.forEach((g,i)=>g.scale.set(saved[i]));}
  return rows.length===3;
}
const stages=['D1','D3-warmup','D3-rest','D4-before','D5','D3-prime-rest','D4-after','D6-after','D7','D8','D9'];
export function runDiagnostic(id,{maxSteps=Infinity,maxWarmChunks=Infinity}={}){
  const sessionStart=Date.now(),run=selected(id),S=make(run),identity=stateHash({schema:4,run,seed:43,sources:sourceHash()}),checkpoint=fileFor(id,'checkpoint'),output=path.join(DIR,id+'-results.json');
  let ctx={schema:4,package:'w4-diag4',id,run,identity,features:testGroups(S),trainingFeatures:testGroups(S),stage:0,data:{},startedAtUtc:new Date().toISOString(),activeWallSeconds:0,done:false};
  if(checkpointExists(checkpoint)){const saved=read(checkpoint);if(saved.identity!==identity||saved.id!==id)throw new Error('诊断检查点参数或源码不匹配');ctx=saved.context;S.loadState(saved.state);}
  const previousWall=ctx.activeWallSeconds;
  if(checkpointExists(output)){const existing=read(output);if(existing.identity!==identity)throw new Error('结果文件与本轮源码不匹配');if(existing.done)return existing;}
  const progress=extra=>{
    ctx.activeWallSeconds=previousWall+(Date.now()-sessionStart)/1000;
    atomic(path.join(DIR,id+'-progress.json'),{id,identity,stage:stages[ctx.stage]??'complete',simulationSeconds:S.net.t/1000,activeWallSeconds:ctx.activeWallSeconds,done:ctx.done,updatedAtUtc:new Date().toISOString(),...extra});
  };
  const save=()=>{progress();atomic(checkpoint,{schema:4,identity,id,context:ctx,state:S.saveState()});atomic(output,{...ctx,parameters:{...S.config,background:{...S.background},seed:43},simulationSeconds:S.net.t/1000,stageName:stages[ctx.stage]??'complete',baseline:clone(S.baseline),finite:validState(S.net),etaRestored:S.net.istdpGroups.every(g=>g.istdp.eta===.02)});};
  const preserve=suffix=>atomic(fileFor(id,suffix),{schema:4,identity,id,parameters:{...S.config,background:{...S.background},seed:43},state:S.saveState()});
  const measure=(key)=>{S.trace.start(ctx.features);const row=measureRest(S);assertFinite(S);row.trace=saveTrace(id,key,S.trace.stop());return row;};
  let steps=0;
  try{
    while(!ctx.done&&steps<maxSteps){
      let advance=true;
      if(ctx.stage===0){ctx.data.D1={...structure(S),allConnections:completeConnections(S)};if(!ctx.data.D1.ok)throw new Error('D1 结构检查失败');}
      else if(ctx.stage===1){
        const preparation=warmup(S,{etaMultiplier:5,checkpointPath:fileFor(id,'warmup'),checkpointEvery:5,maxChunks:maxWarmChunks,onChunk:(row,info)=>{assertFinite(S);if(row.second%5===0){progress({warmupSecond:row.second,converged:info.converged});console.log(JSON.stringify({id,phase:'D3',second:row.second,converged:info.converged}));}}});
        ctx.data.D3={warmup:preparation};if(!preparation.done){advance=false;steps=maxSteps;}
      }else if(ctx.stage===2){Object.assign(ctx.data.D3,measure('D3-rest'));preserve('prepared');}
      else if(ctx.stage===3)advance=probeStep(S,ctx,'D4before','feature',10);
      else if(ctx.stage===4){
        ctx.data.D5=trainPredictions(S,S.spec.scene,{seed:20261011,etaMultiplier:5,features:ctx.trainingFeatures,checkpointPath:fileFor(id,'predictions'),checkpointEvery:1,onChunk:(row,info)=>{assertFinite(S);progress({trainingIndex:info.index,trainingTotal:info.total});if(info.index%5===0)console.log(JSON.stringify({id,phase:'D5',index:info.index,total:info.total}));}});
      }else if(ctx.stage===5){ctx.data.D3prime=measure('D3-prime-rest');preserve('trained');}
      else if(ctx.stage===6)advance=probeStep(S,ctx,'D4after','feature',10);
      else if(ctx.stage===7)advance=probeStep(S,ctx,'D6after','held',5);
      else if(ctx.stage===8)ctx.data.D7={mechanism:mechanismMetrics(S,ctx.data,ctx.features),load:loadMetrics(S,ctx.data,ctx.features)};
      else if(ctx.stage===9){if(run.scene===4){ctx.data.D8??=[];advance=e2SourceStep(S,ctx.data.D8);}else ctx.data.D8={applicable:false,reason:'scene 4 only'};}
      else if(ctx.stage===10){ctx.done=true;ctx.finishedAtUtc=new Date().toISOString();}
      assertFinite(S);if(advance)ctx.stage++;steps++;
      ctx.data.D9={simulationSeconds:S.net.t/1000,activeWallSeconds:previousWall+(Date.now()-sessionStart)/1000,wallSecondsPerSimSecond:(previousWall+(Date.now()-sessionStart)/1000)/(S.net.t/1000||1)};
      save();console.log(JSON.stringify({id,stage:stages[ctx.stage]??'complete',simulationSeconds:S.net.t/1000,done:ctx.done,steps}));
    }
    return read(output);
  }catch(error){progress({fatal:true,error:error.message,code:error.code??'PROGRAM_ERROR'});atomic(path.join(DIR,id+'-fatal.json'),{id,identity,stage:stages[ctx.stage],message:error.message,stack:error.stack,code:error.code??'PROGRAM_ERROR',at:new Date().toISOString()});throw error;}
}
export function stepOne(result){
  const failures=[],data=result.data,add=(item,feature,detail)=>failures.push({item,feature,detail});
  if(!result.done||!data.D7){add('complete',null,'diagnostic incomplete');return {pass:false,failures};}
  if(result.run.scene===1&&!data.D3.warmup.converged)add('1.convergence',null,data.D3.warmup.seconds);
  for(const [phase,level] of [['prepared',data.D3],['trained',data.D3prime]]){
    if(!level.restInRange)add('1.levels',null,{phase,failures:level.selectionRestingFailures});
    if(level.ignition.fraction>.05)add('2.ignition',null,{phase,value:level.ignition.fraction});
  }
  for(const row of data.D7.mechanism.features){
    for(const [key,passed] of Object.entries(row.passes))if(!passed)add(key.startsWith('e1')?'5.'+key:key.includes('Wave')||key.includes('Rebound')?'4.'+key:'3.'+key,row.feature,row[key]);
    for(const key of ['N','M'])if(!row.upstream[key].peakOK)add('6.peak'+key,row.feature,row.upstream[key]);
    for(const key of ['heldN','heldM'])if(!row.upstream[key].ok)add('6.'+key,row.feature,row.upstream[key]);
    if(!row.upstream.S.ok)add('6.Scontrast',row.feature,row.upstream.S);
  }
  return {pass:failures.length===0,failures};
}
export function stepTwo(load){
  const failures=[];if(!load?.applicable)return {pass:false,failures:[{item:'no-built-L'}]};
  for(const r of load.features){
    for(const [key,limit,lower] of [['detectOn',.9,true],['detectOff',.9,true],['direction',.95,true],['falsePresence',.05,false],['falseRest',.05,false],['double',.05,false]])if(!finitePass(r[key],limit,lower))failures.push({item:key,feature:r.feature,value:r[key],limit});
    if(r.heldAboveRest)failures.push({item:'heldAboveRest',feature:r.feature,value:r.held});
    if(!r.lRest.every(v=>finitePass(v,.5)))failures.push({item:'lRest',feature:r.feature,value:r.lRest,limit:.5});
  }
  return {pass:failures.length===0,failures};
}
export function compareCandidates(a,b){return a.maxFalse-b.maxFalse||b.minDetect-a.minDetect||a.relaySize-b.relaySize||b.e1Adaptation-a.e1Adaptation||a.localSEq-b.localSEq||Math.abs(a.gain-1)-Math.abs(b.gain-1);}
function pointMetrics(r){
  const step1=stepOne(r),loads=r.data.D7?.load??[];
  return {...r.run,done:!!r.done,finite:r.finite??null,etaRestored:r.etaRestored??null,converged:r.data.D3?.warmup?.converged??false,warmupSeconds:r.data.D3?.warmup?.seconds??null,
    rest:[r.data.D3,r.data.D3prime].map(x=>x?{rates:x.rest?.rates??null,e2:x.e2Rest??null,restInRange:x.restInRange??false,failures:x.selectionRestingFailures??[],ignition:x.ignition??null}:null),mechanism:r.data.D7?.mechanism??null,load:loads,
    selection:{step1,step2:loads.map(load=>({gain:load.gain,upstreamEligible:step1.pass,...stepTwo(load)}))},D8:r.data.D8??null,D9:r.data.D9??null,error:r.error??null};
}
function resultFor(run){
  const file=path.join(DIR,run.id+'-results.json'),r=checkpointExists(file)?read(file):{id:run.id,run,features:run.scene===4?['红','甜','苦']:['闪电','雷声','鸟叫'],data:{},done:false};
  const fatal=path.join(DIR,run.id+'-fatal.json');if(!r.done&&checkpointExists(fatal))r.error=read(fatal);
  if(r.done&&(!r.finite||!r.etaRestored))throw new Error('完成运行的状态无效：'+run.id);return r;
}
export function summarize(){
  const results=RUNS.map(resultFor),points=results.map(pointMetrics),initialRunsComplete=points.every(p=>p.done),eligible=[],authorizedRuns=[],pendingRuns=[];
  for(const p of points.filter(p=>p.scene===1&&p.selection.step1.pass))for(const load of p.load)if(stepTwo(load).pass)eligible.push({...p,gain:load.gain,maxFalse:Math.max(load.falsePresence,load.falseRest,load.double),minDetect:Math.min(load.detectOn,load.detectOff)});
  eligible.sort(compareCandidates);const chosen=eligible[0]??null;let scene4=null;
  if(initialRunsComplete&&chosen){
    let run=RUNS.find(r=>r.scene===4&&r.relaySize===chosen.relaySize&&r.e1Adaptation===chosen.e1Adaptation&&r.localSEq===chosen.localSEq);
    if(!run){run=extraRun(chosen);authorizedRuns.push(run);const r=resultFor(run);results.push(r);points.push(pointMetrics(r));if(!r.done)pendingRuns.push(run);}
    const p=points.find(p=>p.id===run.id),load=p.load.find(l=>l.gain===chosen.gain);
    scene4={id:run.id,done:p.done,step1:p.selection.step1,step2:stepTwo(load),pass:p.done&&p.selection.step1.pass&&stepTwo(load).pass,D8:p.D8};
  }
  const complete=initialRunsComplete&&!pendingRuns.length&&points.every(p=>p.done),upstreamEligibleCount=points.filter(p=>p.scene===1&&p.selection.step1.pass).length;
  const entry={initialRunsComplete,complete,upstreamEligibleCount,eligibleCount:eligible.length,authorizedRuns,pendingRuns,selected:chosen?{id:chosen.id,scene:1,relaySize:chosen.relaySize,e1Adaptation:chosen.e1Adaptation,localSEq:chosen.localSEq,gain:chosen.gain,localTauR:25}:null,scene4,
    ready:complete&&!!scene4?.pass,stop:complete&&!scene4?.pass,reason:!initialRunsComplete?'initial-runs-incomplete':pendingRuns.length?'selected-scene4-required':!upstreamEligibleCount?'no-upstream-qualified-run':!chosen?'no-qualified-L-gain':!scene4.pass?'scene4-failed':'ready-for-w4-r4',
    unresolved:points.filter(p=>!p.done).map(p=>({id:p.id,error:p.error})),rule:'boundary reconstruction plan section 4; no emergency search or biological early stop'};
  const best=points.filter(p=>p.scene===1&&p.done).sort((a,b)=>a.selection.step1.failures.length-b.selection.step1.failures.length||a.id.localeCompare(b.id))[0]?.id??null;
  const summary={schema:4,package:'w4-diag4',sourceHash:sourceHash(),createdAtUtc:new Date().toISOString(),entry,bestStep1ForFigure:chosen?.id??best,runs:points};
  atomic(path.join(DIR,'summary.json'),summary);console.log(JSON.stringify({phase:'summary',completeRuns:points.filter(p=>p.done).length,upstreamEligibleCount,eligible:eligible.length,reason:entry.reason}));return summary;
}
export async function runPool(items,worker,limit=4){
  const results=new Array(items.length);let next=0;
  await Promise.all(Array.from({length:Math.min(limit,items.length)},async()=>{while(next<items.length){const index=next++;results[index]=await worker(items[index]);}}));
  return results;
}
async function all(){
  fs.mkdirSync(DIR,{recursive:true});const children=[],startedAtUtc=new Date().toISOString(),expectedSourceHash=sourceHash(),cacheBefore=read(path.join(DIR,'cache-before.json'));
  const lock=path.join(DIR,'scheduler.lock');fs.writeFileSync(lock,JSON.stringify({pid:process.pid,startedAtUtc,sourceHash:expectedSourceHash}),{flag:'wx'});
  const originalStats=new Map(),outcomes=[];
  const sha256=file=>{const h=createHash('sha256'),fd=fs.openSync(file,'r'),buf=Buffer.allocUnsafe(4*1024*1024);try{let n;while((n=fs.readSync(fd,buf,0,buf.length,null))>0)h.update(buf.subarray(0,n));return h.digest('hex');}finally{fs.closeSync(fd);}};
  const audit=(full=false)=>{
    for(const entry of cacheBefore.files){const file=path.join(CACHE,entry.name);if(!fs.existsSync(file))throw Object.assign(new Error('已有缓存消失：'+entry.name),{code:'CACHE_INTEGRITY'});const stat=fs.statSync(file),previous=originalStats.get(entry.name);if((full||!previous||previous.mtimeMs!==stat.mtimeMs||previous.size!==stat.size)&&sha256(file)!==entry.sha256)throw Object.assign(new Error('已有缓存哈希改变：'+entry.name),{code:'CACHE_INTEGRITY'});originalStats.set(entry.name,{mtimeMs:stat.mtimeMs,size:stat.size});}
    if(sourceHash()!==expectedSourceHash)throw Object.assign(new Error('运行期间源码改变'),{code:'SOURCE_INTEGRITY'});
  };
  audit(true);let hardFailure=null;
  const manifest=()=>atomic(path.join(DIR,'process-manifest.json'),{parentPid:process.pid,maxWorkers:4,startedAtUtc,updatedAtUtc:new Date().toISOString(),sourceHash:expectedSourceHash,children:children.map(c=>({id:c.diagnosticId,attempt:c.diagnosticAttempt,pid:c.pid,exitCode:c.exitCode,signalCode:c.signalCode}))});
  const halt=error=>{hardFailure=error;for(const c of children)if(c.exitCode===null)c.kill();};
  const timer=setInterval(()=>{try{audit();manifest();}catch(error){halt(error);}},60000);
  const launch=(run,attempt)=>new Promise(resolve=>{
    const log=fs.openSync(path.join(DIR,run.id+'.log'),'a'),child=spawn(process.execPath,[fileURLToPath(import.meta.url),'--run',run.id],{cwd:ROOT,stdio:['ignore',log,log],windowsHide:true});fs.closeSync(log);children.push(child);child.diagnosticId=run.id;child.diagnosticAttempt=attempt;manifest();let settled=false;
    const finish=(code,error=null)=>{
      if(settled)return;settled=true;
      let fatal=null,result={};try{const f=path.join(DIR,run.id+'-fatal.json');fatal=code!==0&&checkpointExists(f)?read(f):null;result=resultFor(run);}catch(e){error=e.message;}
      if(['NONFINITE','BITWISE','CACHE_INTEGRITY','SOURCE_INTEGRITY'].includes(fatal?.code))halt(Object.assign(new Error(fatal.message),{code:fatal.code}));
      outcomes.push({id:run.id,attempt,code,signal:child.signalCode,error:error??fatal,completedStage:result.stageName??null,done:!!result.done,finishedAtUtc:new Date().toISOString()});
      atomic(path.join(DIR,'process-exit-results.json'),{createdAtUtc:new Date().toISOString(),outcomes});resolve(outcomes.at(-1));
    };
    child.once('error',e=>finish(1,e.message));child.once('exit',code=>finish(code));
  });
  const batch=async runs=>{
    const first=await runPool(runs,async r=>{if(hardFailure)return {id:r.id,code:1,done:false,error:'hard-stop before launch'};return launch(r,1);});summarize();if(hardFailure)throw hardFailure;
    const failed=first.filter(r=>r.code!==0||!r.done);if(failed.length)await runPool(failed,row=>launch(runs.find(r=>r.id===row.id),2));
    const summary=summarize();if(hardFailure)throw hardFailure;
    const twiceFailed=failed.filter(row=>{const r=outcomes.filter(r=>r.id===row.id).at(-1);return r.code!==0||!r.done;});if(twiceFailed.length)throw new Error('一次重试后仍失败：'+twiceFailed.map(r=>r.id).join(', '));return summary;
  };
  try{let summary=await batch(RUNS);while(summary.entry.pendingRuns.length)summary=await batch(summary.entry.pendingRuns);audit(true);atomic(path.join(DIR,'cache-integrity.json'),{ok:true,checked:cacheBefore.files.length,s0Verified:cacheBefore.s0Verified,createdAtUtc:new Date().toISOString()});return summary;}
  finally{clearInterval(timer);manifest();atomic(path.join(DIR,'process-exit-results.json'),{createdAtUtc:new Date().toISOString(),hardFailure:hardFailure?{code:hardFailure.code,message:hardFailure.message}:null,outcomes});}
}
export function arithmeticTests(){
  const events=loadEvents([[0,2,4,49.9,70,72,74],[1,3,50,71,73,74.9]]);
  const load=events.length===2&&events[0].time===4&&events[0].strength===6&&events[1].time===74&&events[1].strength===6;
  const directionNet={spikes:[[2,10,110,120,130]]},direction=boundaryDirection(directionNet,[0],100)==='+'&&boundaryDirection(directionNet,[0],150)==='−';
  const net={t:250,spikes:[[50,51,52,53,54,100,101,102,103,104]]},series=spikeSeries(net,[0]),volleys=volleyEvents(series,[[0,200]]);
  const volley=stateHash(volleys)===stateHash([50,100]);
  const passLoad={applicable:true,features:[{feature:'x',detectOn:.9,detectOff:1,direction:.95,falsePresence:.05,falseRest:0,double:.05,heldAboveRest:false,lRest:[.5,.1]}]};
  const bad=clone(passLoad);bad.features[0].direction=.949;
  const candidates=[{maxFalse:0,minDetect:1,relaySize:80,e1Adaptation:20,localSEq:2,gain:1},{maxFalse:0,minDetect:1,relaySize:40,e1Adaptation:5,localSEq:4,gain:.75},{maxFalse:0,minDetect:1,relaySize:40,e1Adaptation:20,localSEq:2,gain:1}];
  candidates.sort(compareCandidates);
  const selection=stepTwo(passLoad).pass&&!stepTwo(bad).pass&&candidates[0].relaySize===40&&candidates[0].e1Adaptation===20;
  const window={mean:windowMean([0,20,10],20,10,50),peak:windowPeak([0,20,10],20,0,50)};
  const result={ok:load&&direction&&volley&&selection&&window.mean===12.5&&window.peak===20,load,direction,volley,selection,events,volleys,window};
  atomic(path.join(DIR,'arithmetic-tests.json'),result);return result;
}
export function selfTest(){
  fs.mkdirSync(DIR,{recursive:true});const started=Date.now(),proofs={};
  const must=(ok,message,code='PROGRAM_ERROR')=>{if(!ok)throw Object.assign(new Error(message),{code});};
  const before=JSON.parse(execFileSync('git',['show','attempt/loop/w4-build-r3:experiments/logs/2026-10-11-loop-w4-build-r3/e1-default-before.json'],{cwd:ROOT,encoding:'utf8',maxBuffer:64*1024*1024}));
  const rows=[{...E1_DEFAULTS},{...E1_DEFAULTS,K:3,NE:320,NI:80,seed:43},{...E1_DEFAULTS,K:14,NE:1200,NI:300,seed:44}].map(cfg=>{const S=buildE1(cfg),connection=connectionHash(S.net),initial=stateHash(S.net.exportState());runE1(S,100,{ids:S.E.slice(0,64),learn:true});runE1(S,100,{learn:true});return {cfg,connection,initial,afterLearning:stateHash(S.net.exportState()),spikes:stateHash(S.net.spikes)};});
  proofs.defaultProof={ok:JSON.stringify(rows)===JSON.stringify(before.rows),rows,before:before.rows};atomic(path.join(DIR,'e1-default-comparison.json'),proofs.defaultProof);must(proofs.defaultProof.ok,'默认 E1 逐位一致检查失败','BITWISE');
  const units=verifyUnits({write:true,freshRest:false});proofs.units=units.ok;must(units.ok,'结构、确定性、网络续算或原型单元检查失败');
  proofs.arithmetic=arithmeticTests();must(proofs.arithmetic.ok,'独立统计手算检查失败');
  const protocol=verifyProtocolResume({write:true});proofs.protocol=protocol.ok;must(protocol.ok,'准备协议逐位续算失败','BITWISE');
  proofs.grid=RUNS.map(run=>{const row=structure(make(run));return {id:run.id,ok:row.ok,sharing:row.sharing,relayByGroup:row.relayByGroup,e1Adaptation:row.e1Adaptation,loadParameters:row.loadParameters,indegreeFailures:row.indegreeFailures,slowTable:row.slowTable,loadTable:row.loadTable};});must(proofs.grid.every(r=>r.ok),'固定网格结构检查失败');
  const empty=buildLoop({scene:1,seed:43}),explicit=buildLoop({scene:1,seed:43,shadowL:[]}),shadow=make(RUNS[0]);
  proofs.defaultOff={ok:stateHash(empty.saveState())===stateHash(explicit.saveState())&&Object.keys(empty.context.shadows).length===0};
  const primaryTopology=empty.net.groups.every((g,i)=>stateHash({pre:[...g.pre],post:[...g.post],scale:[...g.scale]})===stateHash({pre:[...shadow.net.groups[i].pre],post:[...shadow.net.groups[i].post],scale:[...shadow.net.groups[i].scale]}));
  proofs.shadow={ok:primaryTopology&&structure(shadow).shadowIsolation,primaryTopology};must(proofs.defaultOff.ok&&proofs.shadow.ok,'默认影子关闭或影子隔离检查失败');
  proofs.resume=[];
  for(const run of [RUNS[0],RUNS[11],RUNS[12]]){
    const a=make(run),b=make(run),c=make(run),feature=testGroups(a)[0];
    a.trace.start([feature]);b.trace.start([feature]);a.run(400,{...FROZEN,features:[feature]});b.run(200,{...FROZEN,features:[feature]});c.loadState(clone(b.saveState()));c.run(200,{...FROZEN,features:[feature]});
    const hashes=[stateHash(a.saveState()),stateHash(c.saveState())],traceHashes=[stateHash(a.trace.stop()),stateHash(c.trace.stop())];
    const row={id:run.id,hashes,traceHashes,ok:hashes[0]===hashes[1]&&traceHashes[0]===traceHashes[1]};proofs.resume.push(row);atomic(path.join(DIR,'trace-resume-tests.json'),proofs.resume);must(row.ok,'带电导采样的网络逐位续算失败','BITWISE');
  }
  const actionA=buildLoop({scene:4,seed:43}),actionB=buildLoop({scene:4,seed:43});actionA.run(60,{...FROZEN,actions:['吃']});actionB.loadState(clone(actionA.saveState()));actionA.run(60,{...FROZEN,actions:['吃']});actionB.run(60,{...FROZEN,actions:['吃']});
  proofs.action={hashes:[stateHash(actionA.saveState()),stateHash(actionB.saveState())],closed:actionA.actions.handles['吃'].scale.every(v=>v===0)};proofs.action.ok=proofs.action.hashes[0]===proofs.action.hashes[1]&&proofs.action.closed;must(proofs.action.ok,'动作开始门控续算失败','BITWISE');
  const a=make(RUNS[0]);const baseline=a.rest(3000);baseline.baseline=clone(a.baseline);const probe=newProbe(a,testGroups(a)[0],'feature',baseline);probeTrial(a,probe,'unit-probe',{trace:false});
  const snapshot=clone({state:a.saveState(),probe});probeTrial(a,probe,'unit-probe',{trace:false});probe.metrics=probeMetrics(a,probe);
  const b=make(RUNS[0]),resumedProbe=snapshot.probe;b.loadState(snapshot.state);probeTrial(b,resumedProbe,'unit-probe',{trace:false});resumedProbe.metrics=probeMetrics(b,resumedProbe);
  proofs.probeResume={network:[stateHash(a.saveState()),stateHash(b.saveState())],metrics:[stateHash(probe),stateHash(resumedProbe)]};proofs.probeResume.ok=proofs.probeResume.network[0]===proofs.probeResume.network[1]&&proofs.probeResume.metrics[0]===proofs.probeResume.metrics[1];atomic(path.join(DIR,'probe-resume-test.json'),proofs.probeResume);must(proofs.probeResume.ok,'测量逐位续算失败','BITWISE');
  proofs.levels={atLimit:restLevels(empty,{'E1:x':3,I1:12,'N:x':3,'M:x':3,'L:x:±':.5}).restInRange,overLimit:restLevels(empty,{'E1:x':3,I1:12,'N:x':3,'M:x':3,'L:x:±':.501}).restInRange};must(proofs.levels.atLimit&&!proofs.levels.overLimit,'L 静息边界检查失败');
  proofs.io=[];for(const code of ['EPERM','EBUSY','EACCES']){
    let calls=0;const sleeps=[],file=path.join(DIR,'io-self-test-'+code+'-'+Date.now()+'.json'),data={code,proof:'fallback readable'};
    const actual=writeCheckpoint(file,data,{rename:()=>{calls++;throw Object.assign(new Error('injected'),{code});},sleep:ms=>sleeps.push(ms),onFallback:()=>{}});
    proofs.io.push({code,calls,sleeps,actual,ok:calls===6&&sleeps.length===5&&actual!==file&&stateHash(read(file))===stateHash(data)});
  }must(proofs.io.every(r=>r.ok),'JSON 另存恢复检查失败');
  const result={schema:4,ok:true,sourceHash:sourceHash(),createdAtUtc:new Date().toISOString(),wallSeconds:(Date.now()-started)/1000,proofs};atomic(path.join(DIR,'diagnostic-self-test.json'),result);console.log(JSON.stringify({phase:'self-test',ok:true,wallSeconds:result.wallSeconds,sourceHash:result.sourceHash}));return result;
}
if(process.argv[1]&&import.meta.url===pathToFileURL(path.resolve(process.argv[1])).href){
  const args=process.argv.slice(2),parent=args.includes('--all');let error=null;
  try{
    if(parent)await all();else if(args.includes('--summary'))summarize();else if(args.includes('--self-test'))selfTest();else if(args.includes('--arithmetic')){if(!arithmeticTests().ok)process.exitCode=1;}
    else if(args.includes('--run')){const value=name=>args[args.indexOf(name)+1],r=runDiagnostic(value('--run'),{maxSteps:args.includes('--max-steps')?Number(value('--max-steps')):Infinity,maxWarmChunks:args.includes('--max-warm-chunks')?Number(value('--max-warm-chunks')):Infinity});if(!r.done)process.exitCode=2;}
    else console.log('node demo/loop-diag.mjs --self-test | --all | --run <fixed-id> | --summary');
  }catch(e){error={message:e.message,code:e.code??'PROGRAM_ERROR',stack:e.stack};console.error(e.stack);process.exitCode=1;if(!parent&&!args.includes('--run'))atomic(path.join(DIR,'preflight-failure-'+Date.now()+'.json'),error);}
  finally{if(parent)atomic(path.join(DIR,'parent-exit-result.json'),{pid:process.pid,code:process.exitCode??0,finishedAtUtc:new Date().toISOString(),error});}
}
