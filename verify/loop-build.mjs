/** 场景一至四的结构、模块状态、原型规则与静息标定。 */
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {checker} from '../src/core/check.mjs';
import {buildLoop,BACKGROUNDS,BACKGROUND_RANGES,slowCharge,slowStrength,NMDA_REST_CHARGE,SHARED_ROUTES} from '../src/loop/build.mjs';
import {LOAD_BASE_STRENGTH} from '../src/loop/context.mjs';
import {connectionHash,stateHash,binGroups,groupRate,binMsFor,meanPsth,poissonComparison,excessFraction} from '../src/loop/readout.mjs';
import {E1_DEFAULTS,buildE1} from '../src/loop/e1.mjs';
import {cosine} from '../src/loop/e2.mjs';
import {warmup,trainPredictions,convergence,readCheckpoint,writeCheckpoint} from '../src/loop/protocol.mjs';
const ROOT=fileURLToPath(new URL('../',import.meta.url)),PACKAGE=process.env.LOOP_BUILD_PACKAGE??'w4-diag4',DIR=path.join(ROOT,'experiments/logs/2026-10-14-loop-'+PACKAGE);
const PHASE='slow-local-persistence-idle-e2-off-mixed-duration';
const rt=x=>JSON.parse(JSON.stringify(x));
function buildForCalibration(opts){
  const background={};for(const [name,value] of Object.entries(opts.background??{})){
    if(BACKGROUND_RANGES[name])background[name]=value;
    else if(value!==BACKGROUNDS[name])throw new Error('不允许改变固定背景 '+name);
  }
  return buildLoop({...opts,background});
}
export function restingFailures(rates,S,{e2Rates=rates}={}){
  const failures=[];
  for(const [name,shortRate] of Object.entries(rates)){
    const type=name.split(':')[0],rate=type==='E2'?e2Rates[name]:shortRate,range=type==='E1'?[2,5]:type==='I1'?[5,20]:['N','M'].includes(type)?[2,5]:type==='C'?[0,2]:type==='L'?[0,.5]:type==='E2'?(S.e2.labels[Number(name.split(':')[1])]!==null?[.5,2]:[0,1]):['P','D','Up','Down'].includes(type)?[0,5]:null;
    if(range&&(!(rate>=range[0]&&rate<=range[1])||!Number.isFinite(rate)))failures.push({name,rate,range});
  }
  return failures;
}
export function validState(net){
  const vectors=['V','w','Rin','x','y','y2','nmda','xs','ys','xsl','Vd','wd'].map(name=>net[name]).filter(Boolean);
  for(const name of ['ra','rb','ea','eb','dra','drb'])if(net[name])vectors.push(...net[name]);
  for(const g of net.groups){for(const name of ['R','rho','scale','z','nm'])if(g[name])vectors.push(g[name]);if(g.slow)vectors.push(g.slow.a,g.slow.b);}
  return vectors.every(a=>a.every(Number.isFinite))&&['ref','bapUntil'].every(name=>!net[name]||net[name].every(x=>!Number.isNaN(x)));
}
export function structure(S){
  const forbidden=[],E=new Set([...S.e1.E,...S.e2.units.flat()]),P=new Set([...Object.values(S.expect.P).flat(),...Object.values(S.expect.Q).flat()]);
  const firstError=new Set([...Object.values(S.errors.N).flat(),...Object.values(S.errors.M).flat()]),expectation=new Set(Object.values(S.expect.P).flat()),C=new Set(Object.values(S.context.C).flat(2));
  for(let gi=0;gi<S.net.groups.length;gi++){const g=S.net.groups[gi];for(let k=0;k<g.pre.length;k++)if((E.has(g.pre[k])&&P.has(g.post[k]))||(expectation.has(g.pre[k])&&firstError.has(g.post[k]))||(C.has(g.pre[k])&&C.has(g.post[k])))forbidden.push({group:gi,pre:g.pre[k],post:g.post[k]});}
  const expected={E1:Math.max(720,(S.spec.features.length+1)*80),I1:Math.max(720,(S.spec.features.length+1)*80)/4,I2:S.spec.e2Units*5};
  for(let k=0;k<S.spec.e2Units;k++){expected['E2:'+k]=20;expected['J:'+k]=5;}
  for(const g of S.spec.features)for(const [name,n] of [['N',20],['S',S.config.relaySize],['M',20],['V',S.config.relaySize]])expected[name+':'+g]=n;
  for(const ch of [...S.spec.context,...S.spec.outputs])expected['L:'+ch.g+':'+ch.c]=10;
  for(const ch of S.spec.context)for(let k=0;k<8;k++)expected['C:'+ch.g+':'+ch.c+':'+k]=6;
  for(const ch of S.spec.outputs)for(const [name,n] of [['P',20],['Q',10],['Z',10],['D',20],['Up',10],['Down',10],['IUp',10],['IDown',10]])expected[name+':'+ch.g+':'+ch.c]=n;
  for(const g of S.spec.actions)expected['A:'+g]=20;
  for(const multiplier of S.config.shadowL)for(const key of Object.keys(S.context.L))expected['ShadowL:'+multiplier+':'+key]=10;
  const sizes=Object.keys(expected).length===Object.keys(S.net.pops).length&&Object.entries(expected).every(([name,n])=>S.net.pops[name]?.n===n);
  const modules=Object.values(S.modules).every(module=>['readout','saveState','loadState','selfTest'].every(name=>typeof module[name]==='function'));
  const table=S.table(),indegreeFailures=[];
  for(const row of table.indegrees){
    const valid=Math.abs(row.meanIndegree-row.expectedIndegree)<=.1*row.expectedIndegree;
    if(!valid)indegreeFailures.push({route:row.route,mean:row.meanIndegree,expected:row.expectedIndegree});
  }
  for(const row of table.e1.indegrees)if(Math.abs(row.meanIndegree-row.reference)>.1*row.reference)indegreeFailures.push({route:'E1:'+row.route,mean:row.meanIndegree,expected:row.reference});
  const v2Probabilities=S.spec.scene!==4||Object.entries(table.e1.probabilities).every(([name,value])=>value===E1_DEFAULTS[name]);
  const strengths=S.net.groups.every(g=>{
    if(!g.loopRoute)return true;const d=g.loopRoute;
    const shared=S.e2.handles.paths.find(row=>row.group===g);
    const expected=d.strength*d.multiplier*(shared?S.e2.W[shared.k][shared.g]:1);
    return g.scale.every(x=>x===expected)&&(d.p!==1||g.pre.length===d.nPre*d.nPost);
  });
  const local=S.net.groups.filter(g=>['localPrediction','localNegative'].includes(g.loopRoute?.route));
  const slowTable=local.map(g=>{
    const {tauR,tauD}=g.slow,sEq=g.loopRoute.route==='localPrediction'?S.config.localSEq:S.config.localMEq,sSlow=g.loopRoute.strength,expected=slowStrength(sEq,tauR,tauD);
    return {route:g.loopRoute.route,feature:g.loopSlow.feature,tauR,tauD,sEq,sSlow,expected,charge:slowCharge(tauR,tauD),referenceCharge:NMDA_REST_CHARGE,error:Math.abs(sSlow-expected),ok:Math.abs(sSlow-expected)<=1e-12};
  });
  const localRatio=S.config.localMEq/S.config.localSEq;
  const localPaths=local.length===2*S.spec.features.length&&localRatio>=1&&localRatio<=2&&slowTable.every(row=>row.ok)&&local.every(g=>g.kind==='slow'&&g.slow.tauR===S.config.localTauR&&g.slow.tauD===3*S.config.localTauR&&!g.plastic&&!g.istdp&&g.loopRoute.reference===24&&g.loopSlow.sSlow===g.loopRoute.strength);
  const idleE2=S.e2.units.every((_,k)=>S.e2.labels[k]!==null||S.e2.handles.recurrent[k].scale.every(x=>x===0)&&S.e2.handles.inhibition[k].frozen&&S.e2.handles.inhibition[k].scale.every(x=>x===4));
  const loadGains={gain:S.config.loadGain};
  const loadTable=S.context.handles.load.map(g=>({...g.loopLoad,edges:g.pre.length,actualIndegree:g.pre.length/10,strength:g.scale[0]}));
  const loadSensitivity=S.context.handles.load.every(g=>g.loopRoute.k_ref===7&&g.loopRoute.k===7&&g.kind==='slow'&&g.slow.tauR===5&&g.slow.tauD===30&&g.scale.every(x=>Math.abs(x-LOAD_BASE_STRENGTH*S.config.loadGain)<=1e-12));
  const sharing=S.net.groups.filter(g=>g.loopRoute).every(g=>{const d=g.loopRoute,k=Math.min(d.k_ref,d.n_pre,SHARED_ROUTES.includes(d.route)?Math.floor(.35*d.n_pre):Infinity);return d.k===k&&d.p===k/d.n_pre&&d.compensation===d.k_ref/k&&d.multiplier===d.compensation&&(!SHARED_ROUTES.includes(d.route)||d.sharingRatio<=.35);});
  const relayByGroup=S.errors.handles.relay.every((g,k)=>g.pre.every(i=>S.errors.S[S.spec.features[k]].includes(i))&&g.post.every(i=>S.errors.N[S.spec.features[k]].includes(i)));
  const allLoads=[...Object.values(S.context.L),...Object.values(S.context.shadows).flatMap(copy=>Object.values(copy))];
  const loadParameters=allLoads.every(ids=>ids.every((id,q)=>{const p=S.net.par[id];return p.C===100&&p.b===200&&p.tauW===50&&Math.abs(p.Vth-p.EL-30*(1+.1*(2*q/9-1)))<1e-12&&S.backgroundRate()(id)===S.background.L;}));
  const e1Adaptation=S.e1.b===S.config.e1Adaptation&&S.e1.E.every(i=>S.net.par[i].b===S.config.e1Adaptation);
  const featureChannels=[...new Map([...S.spec.context,...S.spec.outputs].filter(ch=>S.spec.features.includes(ch.g)).map(ch=>[ch.g,ch])).values()];
  const singleBoundary=featureChannels.every(ch=>ch.c==='±'&&S.context.L[ch.g+':±']&&!S.context.L[ch.g+':+']&&!S.context.L[ch.g+':−'])&&(S.spec.scene!==1||Object.keys(S.context.L).length===3);
  const i2Quarter=S.net.pops.I2.n===S.e2.units.flat().length/4;
  const shadowIds=new Set(Object.values(S.context.shadows).flatMap(copy=>Object.values(copy).flat()));
  const shadowIsolation=S.net.groups.every(g=>g.pre.every(id=>!shadowIds.has(id)))&&Object.keys(S.context.shadows).length===S.config.shadowL.length&&S.context.shadowHandles.every(g=>(!g.loopLoad||g.scale.every(x=>Math.abs(x-LOAD_BASE_STRENGTH*g.loopLoad.gain)<=1e-12))&&g.post.every(id=>shadowIds.has(id)))&&Object.values(S.context.shadows).every(copy=>Object.keys(copy).length===Object.keys(S.context.L).length);
  return {ok:S.selfTest()&&modules&&sizes&&forbidden.length===0&&indegreeFailures.length===0&&strengths&&v2Probabilities&&localPaths&&idleE2&&loadSensitivity&&shadowIsolation&&i2Quarter&&sharing&&relayByGroup&&loadParameters&&e1Adaptation&&singleBoundary,forbidden,modules,sizes,strengths,localPaths,slowTable,localRatio,idleE2,loadSensitivity,loadGains,loadTable,i2Quarter,shadowIsolation,sharing,relayByGroup,loadParameters,e1Adaptation,singleBoundary,defaultHasNoShadows:S.config.shadowL.length===0&&shadowIds.size===0,v2Probabilities,indegreeFailures,expected,table};
}
function prototypeCheck(){
  const S=buildLoop({scene:3,seed:43}),m=S.e2.learner;
  S.e2.W[0].fill(.2);S.e2.W[1].fill(.6);m.sync();m.eta=.1;m.referenceHz.fill(20);m.maxHz.fill(40);m.enabled=true;
  const expected=S.e2.W.map(row=>[...row]),rates=S.spec.features.map((_,i)=>i%2?10:20),activity=S.e2.units.map((_,k)=>k===0?20:0);
  expected[0]=expected[0].map((w,g)=>w+.05*(rates[g]/20-w));m.update(rates,activity);
  const error=Math.max(...S.e2.W.flatMap((row,k)=>[...row].map((x,g)=>Math.abs(x-expected[k][g]))));
  const scales=m.paths.every(row=>row.group.scale.every(x=>x===row.strength*m.W[row.k][row.g]));
  const inhibition=m.lateral.every(row=>row.group.scale.every(x=>x===row.strength*cosine(m.W[row.k],m.W[row.l])));
  const hold=stateHash(m.exportState());m.update(rates.map(()=>0),activity);
  return {ok:error<=1e-9&&scales&&inhibition&&hold===stateHash(m.exportState()),error,scales,inhibition,gate:hold===stateHash(m.exportState())};
}

function writeResult(name,result){fs.mkdirSync(DIR,{recursive:true});fs.writeFileSync(path.join(DIR,name+'.json'),JSON.stringify(result,null,2)+'\n');return result;}
function saveProbe(file,data){
  if(!path.basename(file).startsWith('loop-'))throw new Error('探针检查点必须以 loop- 开头');
  writeCheckpoint(file,data);
}
function loadProbe(file,identity){
  if(!fs.existsSync(file))return null;
  const saved=readCheckpoint(file);
  if(saved.schema!==3||saved.identity!==identity)throw new Error('探针检查点参数不匹配');
  return saved;
}
function resumeCheck(S,{fromState=null}={}){
  const initial=fromState??S.saveState(),fresh=()=>buildLoop({scene:S.spec.scene,seed:S.seed,...S.config});
  const a=fresh(),b=fresh(),c=fresh();a.loadState(rt(initial));b.loadState(rt(initial));
  const drives=new Map([[a.context.C['闪电:±'][0][0],58],[a.expect.Up['雷声:±'][0],25]]);
  const options={features:['闪电'],drives,learn:true,e1Learn:true,prototypeLearn:true,tdLearn:true};
  a.run(400,options);b.run(200,options);const json=JSON.stringify(b.saveState());c.loadState(JSON.parse(json));c.run(200,options);
  const one=stateHash(a.saveState()),two=stateHash(c.saveState());
  return {ok:one===two,continuous:one,resumed:two,bytes:Buffer.byteLength(json)};
}
export function verifyProtocolResume({write=true}={}){
  const seed=20261011,etaMultiplier=5,make=()=>buildLoop({scene:1,seed:43}),file=path.join(DIR,'loop-w4diag4-protocol-unit-warmup-'+process.pid+'.json');
  const a=make(),b=make(),c=make();
  const wa=warmup(a,{seed:43,etaMultiplier,maxChunks:2,checkpointPath:null}),wb=warmup(b,{seed:43,etaMultiplier,maxChunks:1,checkpointPath:file,resume:false}),wc=warmup(c,{seed:43,etaMultiplier,maxChunks:1,checkpointPath:file,resume:true});
  const warmupHashes=[stateHash(a.saveState()),stateHash(c.saveState())],warmupCurves=[stateHash(wa.curve),stateHash(wc.curve)];
  const d=make(),e=make(),f=make(),predFile=path.join(DIR,'loop-w4diag4-protocol-unit-predictions-'+process.pid+'.json');
  const pa=trainPredictions(d,1,{seed,etaMultiplier,maxChunks:2}),pb=trainPredictions(e,1,{seed,etaMultiplier,maxChunks:1,checkpointPath:predFile,resume:false}),pc=trainPredictions(f,1,{seed,etaMultiplier,maxChunks:1,checkpointPath:predFile,resume:true});
  const predictionHashes=[stateHash(d.saveState()),stateHash(f.saveState())],predictionCurves=[stateHash(pa.curve),stateHash(pc.curve)];
  const etaRestored=[a,b,c,d,e,f].every(S=>S.net.istdpGroups.every(g=>g.istdp.eta===.02));
  const durationCounts=Object.fromEntries(d.spec.features.map(g=>[g,Object.fromEntries([1000,3000,10000].map(ms=>[ms,pa.schedule.filter(row=>row.features[0]===g&&row.ms===ms).length]))]));
  const scheduleOK=Object.values(durationCounts).every(counts=>Object.values(counts).every(n=>n===10))&&pa.schedule.every(row=>row.restMs>=1000&&row.restMs<=2000);
  const idleOff=[a,b,c,d,e,f].every(S=>S.e2.handles.inhibition.every(g=>g.frozen&&g.scale.every(x=>x===4))&&S.e2.handles.recurrent.every(g=>g.scale.every(x=>x===0)));
  const result={schema:3,createdAtUtc:new Date().toISOString(),ok:warmupHashes[0]===warmupHashes[1]&&warmupCurves[0]===warmupCurves[1]&&predictionHashes[0]===predictionHashes[1]&&predictionCurves[0]===predictionCurves[1]&&etaRestored&&scheduleOK&&idleOff,warmupHashes,warmupCurves,predictionHashes,predictionCurves,durationCounts,scheduleOK,idleOff,etaMultiplier,etaRestored,networkSeed:43,scheduleSeed:seed};
  if(write)writeResult('protocol-resume-results',result);console.log('协议检查点续算 '+JSON.stringify(result));return {ok:result.ok,result};
}
export function verifyUnits({write=true,freshRest=true}={}){
  const c=checker('W4-diag4 结构、确定性、续算与原型'),metrics={structures:[],freshRest:[]};
  const finish=()=>{const result={schema:3,package:PACKAGE,createdAtUtc:new Date().toISOString(),ok:c.done(),metrics};if(write)writeResult('unit-results',result);return {ok:result.ok,result};};
  for(const scene of [1,2,3,4]){
    const S=buildLoop({scene,seed:43}),row=structure(S);metrics.structures.push(row);
    c.check('场景 '+scene+'：规模、十二条通路入度与强度、禁止连接',row.ok,JSON.stringify({neurons:S.net.N,indegreeFailures:row.indegreeFailures,strengths:row.strengths,localPaths:row.localPaths}));
    console.table(row.table.indegrees.map(({route,reference,meanIndegree,expectedIndegree})=>({route,reference,meanIndegree,expectedIndegree})));
    console.table(row.table.e1.indegrees);
    console.table(row.table.populations.map(({name,n})=>({name,n})));
    if(!row.ok)return finish();
    if(freshRest){const start=S.net.t;S.run(2000,{learn:true,e1Learn:false,prototypeLearn:false,tdLearn:false});S.measureBaseline(start,2000);metrics.freshRest.push({scene,background:S.background,learning:'inhibitory-only (original W4 measurement)',start,end:S.net.t,rates:Object.fromEntries(Object.entries(S.groups).map(([name,ids])=>[name,groupRate(S.net,ids,start,start+2000)])),finite:validState(S.net)});}
  }
  const one=buildLoop({scene:1,seed:43}),same=buildLoop({scene:1,seed:43}),different=buildLoop({scene:1,seed:44});
  const original=buildE1(),normalized=buildE1(E1_DEFAULTS,null,{inDegree:'v2'});
  metrics.e1V2At720={original:stateHash(original.net.exportState()),normalized:stateHash(normalized.net.exportState())};
  if(!c.check('720 个 E1 的 v2 入度选项与默认构建逐位相同',metrics.e1V2At720.original===metrics.e1V2At720.normalized))return finish();
  metrics.e1Default=JSON.parse(fs.readFileSync(path.join(DIR,'e1-default-comparison.json'),'utf8'));
  if(!c.check('默认关闭时三配置构建、学习与放电逐位不变',metrics.e1Default.ok))return finish();
  const flat=Array.from({length:60},(_,i)=>({second:i+1,rates:{E1:3,'N:group':3,'M:group':3,'S:group':6}})),sizes={E1:720,'N:group':20,'M:group':20,'S:group':10};
  const stable=convergence(flat,sizes);flat.slice(-20).forEach(r=>{r.rates['N:group']=4;});const shifted=convergence(flat,sizes);
  metrics.convergenceArithmetic={stable,shifted};
  if(!c.check('两窗口判据对稳定/漂移数据与泊松手算一致',stable.ok&&!shifted.ok&&Math.abs(shifted.changes['N:group'].se-Math.sqrt(7/400))<1e-15))return finish();
  metrics.determinism={one:connectionHash(one.net),same:connectionHash(same.net),different:connectionHash(different.net)};
  if(!c.check('同种子哈希相同，异种子不同',metrics.determinism.one===metrics.determinism.same&&metrics.determinism.one!==metrics.determinism.different,JSON.stringify(metrics.determinism)))return finish();
  metrics.resume=resumeCheck(one);
  if(!c.check('新建完整网络 JSON 续跑 200 ms 逐位相同',metrics.resume.ok,JSON.stringify(metrics.resume)))return finish();
  metrics.prototype=prototypeCheck();c.check('新入度结构上原型更新、四路径与侧抑制手算',metrics.prototype.ok,JSON.stringify(metrics.prototype));
  return finish();
}
function nextCalibration(row){
  const background={...row.background},old={background:row.background,recurrent:row.recurrent,etaMultiplier:row.etaMultiplier};
  let recurrent=row.recurrent,etaMultiplier=row.etaMultiplier;
  const failures=row.failures,highE2=failures.filter(f=>f.name.startsWith('E2:')&&f.rate>f.range[1]),lowE2=failures.filter(f=>f.name.startsWith('E2:')&&f.rate<f.range[0]);
  if(highE2.length){
    if(background.E2>10)background.E2=Math.max(10,background.E2-1);
    else if(recurrent>2)recurrent=Math.max(2,recurrent-.5);
    else if(background.E1>20)background.E1=Math.max(20,background.E1-1);
  }else if(lowE2.length)background.E2=Math.min(20,background.E2+1);
  for(const type of ['E1','I1','N','M','S','C','P','Up','Down']){
    const own=failures.filter(f=>f.name.split(':')[0]===type);if(!own.length)continue;
    const direction=own.every(f=>f.rate<f.range[0])?1:own.every(f=>f.rate>f.range[1])?-1:0;
    if(direction){const [lo,hi]=BACKGROUND_RANGES[type];background[type]=Math.min(hi,Math.max(lo,background[type]+direction));}
  }
  if(!row.warmup.converged&&failures.length===0)etaMultiplier=etaMultiplier===5?3:etaMultiplier===3?1:1;
  const next={background,recurrent,etaMultiplier};
  return JSON.stringify(next)===JSON.stringify(old)?null:next;
}
export function verifyScene(scene,{maxAttempts=12,maxChunks=Infinity,write=true,calibrationRequest=null,variant=null}={}){
  if(variant!==null&&!/^[a-z0-9-]{1,32}$/.test(variant))throw new Error('标定名称无效');
  const stem='scene-'+scene+(variant?'-'+variant:''),file=path.join(DIR,stem+'-results.json'),saved=fs.existsSync(file)?JSON.parse(fs.readFileSync(file,'utf8')):null;
  if(saved?.done&&!calibrationRequest)return {ok:saved.ok,result:saved};
  if(calibrationRequest&&saved&&!saved.done)throw new Error('下一组标定参数须等当前运行结束，不能覆盖正在运行的检查点');
  let background=calibrationRequest?.background??saved?.next?.background??saved?.background??{...BACKGROUNDS};
  let recurrent=calibrationRequest?.recurrent??saved?.next?.recurrent??saved?.recurrent??4.5,etaMultiplier=calibrationRequest?.etaMultiplier??saved?.next?.etaMultiplier??saved?.etaMultiplier??5;
  let attempt=calibrationRequest?(saved?.calibration?.at(-1)?.attempt??-1)+1:saved?.nextAttempt??saved?.attempt??0;
  if(calibrationRequest){
    while(fs.existsSync(path.join(ROOT,'cache','loop-w4r4-s'+scene+(variant?'-'+variant:'')+'-a'+attempt+'-warmup.json')))attempt++;
    maxAttempts=Math.max(maxAttempts,attempt+1);
  }
  const calibration=saved?.calibration??[],interruptedProfiles=saved?.interruptedProfiles??[],startedAtUtc=saved?.startedAtUtc??new Date().toISOString();
  for(;attempt<maxAttempts;attempt++){
    const warmupFile='loop-w4r4-s'+scene+(variant?'-'+variant:'')+'-a'+attempt+'-warmup.json';
    const params={schema:3,phase:PHASE,scene,variant,attempt,warmupFile,background,recurrent,etaMultiplier,startedAtUtc,calibrationRequest,calibration,interruptedProfiles,done:false,ok:false};
    if(write)writeResult(stem+'-results',params);
    const S=buildForCalibration({scene,seed:43,background,e2Recurrent:recurrent}),t=Date.now();
    const prep=warmup(S,{etaMultiplier,checkpointPath:path.join(ROOT,'cache',warmupFile),maxChunks,onChunk:(row,info)=>{
      if(row.second%10===0)console.log('预热 '+JSON.stringify({phase:PHASE,scene,attempt,second:row.second,converged:info.converged,maxRatio:info.convergence?Math.max(...Object.values(info.convergence.changes).map(x=>x.ratio)):null,etaMultiplier,wallSeconds:(Date.now()-t)/1000}));
    }});
    if(!prep.done){const result={...params,warmup:prep,done:false};if(write)writeResult(stem+'-results',result);return {ok:false,result};}
    const rest=S.rest(2000),failures=restingFailures(rest.rates,S),finite=validState(S.net);
    const row={phase:PHASE,scene,variant,attempt,warmupFile,background:{...S.background},recurrent,etaMultiplier,warmup:prep,rest,failures,finite,etaRestored:S.net.istdpGroups.every(g=>g.istdp.eta===.02),wallSeconds:(Date.now()-t)/1000};
    calibration.push(row);
    console.log('静息 '+JSON.stringify({scene,attempt,converged:prep.converged,seconds:prep.seconds,failures,finite}));
    if(scene===1){
      row.postWarmupResume=resumeCheck(S);
      if(!row.postWarmupResume.ok){const result={...params,done:true,ok:false,reason:'bitwise-resume',calibration};if(write)writeResult(stem+'-results',result);return {ok:false,result};}
    }
    const passed=prep.converged&&finite&&row.etaRestored&&failures.length===0,next=nextCalibration(row);
    const repeated=next&&calibration.some(r=>JSON.stringify({background:r.background,recurrent:r.recurrent,etaMultiplier:r.etaMultiplier})===JSON.stringify(next));
    if(passed||!finite||!next||repeated||attempt+1>=maxAttempts){
      const result={...params,createdAtUtc:new Date().toISOString(),done:true,ok:passed,calibration,reason:passed?'passed':!finite?'non-finite':repeated?'calibration-cycle':!next?'no-next-allowed-calibration':'calibration-attempt-limit'};
      if(write)writeResult(stem+'-results',result);return {ok:passed,result};
    }
    const result={...params,next,nextAttempt:attempt+1,calibration};
    if(write)writeResult(stem+'-results',result);
    ({background,recurrent,etaMultiplier}=next);
  }
}
export function verifyShallow({write=true}={}){
  const prepared=JSON.parse(fs.readFileSync(path.join(DIR,'scene-3-results.json'),'utf8'));
  if(!prepared.ok)throw new Error('场景三预热必须先通过');
  const last=prepared.calibration.at(-1),S=buildForCalibration({scene:3,seed:43,background:last.background,e2Recurrent:last.recurrent});
  const checkpointStem=(last.warmupFile??'loop-w4r4-s3-a'+last.attempt+'-warmup.json').replace(/-warmup\.json$/,'');
  const cp=readCheckpoint(path.join(ROOT,'cache',checkpointStem+'-warmup.json'));
  const file=path.join(ROOT,'cache',checkpointStem+'-shallow.json'),identity=stateHash({protocol:'shallow',version:3,warmup:cp.identity});
  const saved=loadProbe(file,identity);S.loadState(saved?.state??cp.state);
  const rest=saved?.rest??S.rest(2000),trials=saved?.trials??[],unitChecks=saved?.unitChecks??S.e2.labels.map((label,k)=>({label,k,restHz:rest.rates['E2:'+k],baseline:S.baseline['E2:'+k]})).filter(r=>r.label);
  const schedule=unitChecks.flatMap(row=>Array.from({length:10},(_,trial)=>({row,trial})));
  for(let index=trials.length;index<schedule.length;index++){
    const {row,trial}=schedule[index];
    const start=S.net.t;S.run(1000,{features:S.spec.prototypes[row.label],learn:false});
    const duringHz=groupRate(S.net,S.e2.units[row.k],start+50,start+1000),onBins=binGroups(S.net,{unit:S.e2.units[row.k]},start,1000).unit,withdraw=S.net.t;
    S.run(1000,{learn:false});const bins=binGroups(S.net,{unit:S.e2.units[row.k]},withdraw,1000).unit,binMs=binMsFor(S.e2.units[row.k]);
    // 回到阈值以下的时刻取峰值之后的首次下穿；比例指标逐试次计算。
    const peakIndex=bins.indexOf(Math.max(...bins.slice(0,Math.ceil(150/binMs)))),returnedIndex=bins.findIndex((v,k)=>k>=Math.max(0,peakIndex)&&v<=row.baseline.theta);
    const returnMs=returnedIndex<0?null:(returnedIndex+.5)*binMs;
    trials.push({label:row.label,k:row.k,trial,duringHz,returnMs,returned150:returnMs!==null&&returnMs<=150,binMs,onBins,bins});
    saveProbe(file,{schema:3,identity,rest,unitChecks,trials,done:trials.length===schedule.length,state:S.saveState()});
    console.log('E2 浅 '+JSON.stringify({label:row.label,trial,duringHz,returnMs}));
  }
  const checks=unitChecks.map(row=>{
    const own=trials.filter(t=>t.k===row.k),meanHz=own.reduce((s,t)=>s+t.duringHz,0)/own.length,returnFraction=own.filter(t=>t.returned150).length/own.length,psth=meanPsth(own.map(t=>t.bins)),onPsth=meanPsth(own.map(t=>t.onBins));
    return {...row,meanHz,returnFraction,psth,onPsth,ok:meanHz>=8&&meanHz>=3*row.restHz&&returnFraction>=.9&&row.restHz>=.5&&row.restHz<=2};
  });
  const result={schema:3,createdAtUtc:new Date().toISOString(),ok:checks.every(row=>row.ok),statistics:'20-ms bins labelled at their centres; ten-trial PSTH; return fraction measured per trial',attempt:last.attempt,background:last.background,recurrent:last.recurrent,checks,trials};
  if(write){writeResult('shallow-'+(last.variant?last.variant+'-':'')+'a'+last.attempt+'-results',result);writeResult('shallow-results',result);}return {ok:result.ok,result};
}
function heldRecord(S,feature,condition,attempt){
  const file=path.join(ROOT,'cache','loop-w4r4-held-s1-a'+attempt+'-'+condition+'-'+S.spec.features.indexOf(feature)+'.json');
  const identity=stateHash({protocol:'held',version:3,feature,condition,initial:S.saveState()}),saved=loadProbe(file,identity);
  if(saved)S.loadState(saved.state);
  const rest=saved?.rest??S.rest(2000),baseline=saved?.baseline??structuredClone(S.baseline),trials=saved?.trials??[];
  for(let trial=trials.length;trial<5;trial++){
    const start=S.net.t;S.run(10000,{features:[feature],learn:false});
    const rates=Object.fromEntries(['N','M'].map(type=>[type,groupRate(S.net,S.errors[type][feature],start+150,start+10000)]));
    const bins=binGroups(S.net,{N:S.errors.N[feature],M:S.errors.M[feature]},start,10000);
    trials.push({trial,start,rates,bins});S.run(2000,{learn:false});
    saveProbe(file,{schema:3,identity,rest,baseline,trials,done:trials.length===5,state:S.saveState()});
    console.log('持续呈现试次 '+JSON.stringify({feature,condition,trial:trial+1,total:5}));
  }
  const groups=Object.fromEntries(['N','M'].map(type=>{
    const base=baseline[type+':'+feature],rate=trials.reduce((s,r)=>s+r.rates[type],0)/5,comparison=poissonComparison(rate,rest.rates[type+':'+feature],S.errors[type][feature].length,5*9.85,2,1),psth=meanPsth(trials.map(r=>r.bins[type]));
    const binMs=binMsFor(S.errors[type][feature]),after150=psth.filter((_,k)=>k*binMs>=150);
    return [type,{rate,baseline:rest.rates[type+':'+feature],signedDifference:rate-rest.rates[type+':'+feature],...comparison,binMs,psth,excess:excessFraction(after150,base)}];
  }));
  return {feature,condition,groups,trials,baseline};
}
export function verifyPredictionTool({write=true}={}){
  const prepared=JSON.parse(fs.readFileSync(path.join(DIR,'scene-1-results.json'),'utf8')),last=prepared.calibration.at(-1);
  if(!prepared.ok)throw new Error('场景一预热必须先通过');
  const make=()=>buildForCalibration({scene:1,seed:43,background:last.background,e2Recurrent:last.recurrent}),cp=readCheckpoint(path.join(ROOT,'cache','loop-w4r4-s1-a'+last.attempt+'-warmup.json')),S=make();
  S.loadState(cp.state);
  const before=S.spec.features.map(feature=>{const probe=make();probe.loadState(rt(cp.state));const row=heldRecord(probe,feature,'before',last.attempt);console.log('持续呈现 '+JSON.stringify({feature,condition:'before',groups:Object.fromEntries(Object.entries(row.groups).map(([k,r])=>[k,{rate:r.rate,baseline:r.baseline,difference:r.signedDifference,threeSE:r.threeSE}]))}));return row;});
  if(write)writeResult('prediction-tool-before-results',{schema:3,attempt:last.attempt,before});
  const trained=trainPredictions(S,1,{seed:20261011,etaMultiplier:last.etaMultiplier,checkpointPath:path.join(ROOT,'cache','loop-w4r4-s1-a'+last.attempt+'-predictions.json'),checkpointEvery:1,onChunk:(row,info)=>{if(info.index%5===0)console.log('预测抵消训练 '+info.index+'/'+info.total);}});
  const state=S.saveState(),after=S.spec.features.map(feature=>{const probe=make();probe.loadState(rt(state));const row=heldRecord(probe,feature,'after',last.attempt);console.log('持续呈现 '+JSON.stringify({feature,condition:'after',groups:Object.fromEntries(Object.entries(row.groups).map(([k,r])=>[k,{rate:r.rate,baseline:r.baseline,difference:r.signedDifference,threeSE:r.threeSE}]))}));return row;});
  const result={schema:3,createdAtUtc:new Date().toISOString(),ok:trained.done,before,after,training:trained,statistics:'five 10-second trials per feature and condition; N/M after 150 ms; signed baseline difference and 3SE are record-only'};
  if(write)writeResult('prediction-tool-results',result);return {ok:result.ok,result};
}

export function verifyLoopBuild({write=!process.env.FAST}={}){
  const units=verifyUnits({write,freshRest:!process.env.FAST});if(!units.ok)return units;
  const c=checker('W4-r3 预热、静息与 E2 门槛'),metrics={units:units.result,scenes:[]};
  const protocolFile=path.join(DIR,'protocol-resume-results.json'),protocol=fs.existsSync(protocolFile)?JSON.parse(fs.readFileSync(protocolFile,'utf8')):verifyProtocolResume({write}).result;
  metrics.protocolResume=protocol;if(!c.check('预热与预测抵消协议检查点逐位续算',protocol.ok))return {ok:c.done(),result:metrics};
  for(const scene of [1,2,3,4]){
    const cached=path.join(DIR,'scene-'+scene+'-results.json');
    const outcome=process.env.FAST&&fs.existsSync(cached)?{result:JSON.parse(fs.readFileSync(cached,'utf8'))}:verifyScene(scene,{write});
    const result=outcome.result;metrics.scenes.push(result);
    if(!c.check('场景 '+scene+'：300 s 内收敛、冻结后所有频率达标',result.done&&result.ok,JSON.stringify(result.calibration.at(-1)?.failures)))return {ok:c.done(),result:metrics};
  }
  const shallow=process.env.FAST?JSON.parse(fs.readFileSync(path.join(DIR,'shallow-results.json'),'utf8')):verifyShallow({write}).result;
  metrics.shallow=shallow;c.check('场景三 E2：有证据活跃、撤除快、静息无自燃',shallow.ok);
  if(!shallow.ok)return {ok:c.done(),result:metrics};
  const tool=process.env.FAST?JSON.parse(fs.readFileSync(path.join(DIR,'prediction-tool-results.json'),'utf8')):verifyPredictionTool({write}).result;
  metrics.predictionTool=tool;c.check('预测抵消训练工具完成（持续误差只记录）',tool.ok);
  const result={schema:3,package:'w4-build-r3',createdAtUtc:new Date().toISOString(),ok:c.done(),metrics};if(write)writeResult('build-results',result);return {ok:result.ok,result};
}
if(process.argv[1]&&import.meta.url===pathToFileURL(path.resolve(process.argv[1])).href){
  const mode=process.env.MODE??'all';
    const outcome=mode==='protocol'?verifyProtocolResume():mode==='unit'?verifyUnits():mode==='scene'?verifyScene(Number(process.env.SCENE??1),{maxAttempts:Number(process.env.MAX_ATTEMPTS??12),maxChunks:Number(process.env.MAX_CHUNKS??Infinity),variant:process.env.VARIANT??null,calibrationRequest:process.env.CALIBRATION_FILE?JSON.parse(fs.readFileSync(process.env.CALIBRATION_FILE,'utf8')):null}):mode==='shallow'?verifyShallow():mode==='predictions'?verifyPredictionTool():verifyLoopBuild();
  process.exit(outcome.ok?0:outcome.result?.done===false?2:1);
}
