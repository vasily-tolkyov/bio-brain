/** 第 4.7 节：仅开抑制性可塑性的确定性准备协议。 */
import fs from 'node:fs';
import path from 'node:path';
import {deflateSync,inflateSync} from 'node:zlib';
import {resumableRng} from './e1.mjs';
import {groupRate,stateHash,poissonComparison,baselines,binGroups} from './readout.mjs';
const ONLY_INHIBITION=Object.freeze({learn:true,e1Learn:false,prototypeLearn:false,tdLearn:false});
const CHECKPOINT_CODEC='deflate-xor-shuffle-spikes-v2';
const LEGACY_CHECKPOINT_CODEC='deflate-spikes-v1';
function shuffleSpikeBytes(bytes,count){
  const shuffled=Buffer.allocUnsafe(bytes.length);
  for(let i=0;i<count;i++)for(let b=0;b<8;b++)shuffled[b*count+i]=bytes[i*8+b]^(i?bytes[(i-1)*8+b]:0);
  return shuffled;
}
function restoreSpikeBytes(shuffled,count){
  const bytes=Buffer.allocUnsafe(shuffled.length);
  for(let i=0;i<count;i++)for(let b=0;b<8;b++)bytes[i*8+b]=shuffled[b*count+i]^(i?bytes[(i-1)*8+b]:0);
  return bytes;
}
/** 无损压缩完整的脉冲向量，保持网络导出/导入接口及运算不变。 */
export function encodeCheckpoint(data){
  if(!data?.state?.net?.spikes)return data;
  const spikes=data.state.net.spikes.map(row=>{
    if(!row.length)return row;
    const bytes=Buffer.from(row.data,'base64');
    if(bytes.length!==row.length*8)throw new Error('脉冲向量编码长度不匹配');
    const packed=deflateSync(shuffleSpikeBytes(bytes,row.length),{level:1}).toString('base64');
    return packed.length<row.data.length?{...row,data:packed,compression:'deflate-xor-shuffle'}:row;
  });
  return {...data,state:{...data.state,net:{...data.state.net,spikes}},checkpointCodec:CHECKPOINT_CODEC};
}
export function decodeCheckpoint(data){
  if(!data?.checkpointCodec)return data;
  if(![CHECKPOINT_CODEC,LEGACY_CHECKPOINT_CODEC].includes(data.checkpointCodec)||!Array.isArray(data.state?.net?.spikes))throw new Error('检查点压缩格式不匹配');
  const spikes=data.state.net.spikes.map(row=>{
    if(!row.compression)return row;
    if(!['deflate','deflate-xor-shuffle'].includes(row.compression))throw new Error('脉冲向量压缩格式不匹配');
    const inflated=inflateSync(Buffer.from(row.data,'base64'));
    if(inflated.length!==row.length*8)throw new Error('脉冲向量解压长度不匹配');
    const bytes=row.compression==='deflate-xor-shuffle'?restoreSpikeBytes(inflated,row.length):inflated;
    const plain={...row,data:bytes.toString('base64')};delete plain.compression;return plain;
  });
  const plain={...data,state:{...data.state,net:{...data.state.net,spikes}}};delete plain.checkpointCodec;return plain;
}
/** 回退文件也参与恢复；之后正常写回时，按修改时间选择最新的完整 JSON。 */
export function checkpointFile(file){
  const dir=path.dirname(file),base=path.basename(file,'.json'),candidates=[];
  if(fs.existsSync(file))candidates.push(file);
  if(fs.existsSync(dir))for(const entry of fs.readdirSync(dir)){
    if(entry.startsWith(base+'.')&&/^\d+(?:-\d+)?\.json$/.test(entry.slice(base.length+1)))candidates.push(path.join(dir,entry));
  }
  candidates.sort((a,b)=>fs.statSync(b).mtimeMs-fs.statSync(a).mtimeMs||(a===file?1:b===file?-1:b.localeCompare(a)));
  return candidates[0]??null;
}
export const checkpointExists=file=>checkpointFile(file)!==null;
export function readCheckpoint(file){return decodeCheckpoint(JSON.parse(fs.readFileSync(checkpointFile(file)??file,'utf8').replace(/^\uFEFF/,'')));}
export function writeCheckpoint(file,data,{rename=fs.renameSync,sleep=ms=>Atomics.wait(new Int32Array(new SharedArrayBuffer(4)),0,0,ms),onFallback=message=>console.warn(JSON.stringify(message))}={}){
  if(!file)return;
  if(path.basename(path.dirname(file))==='cache'&&!path.basename(file).startsWith('loop-'))throw new Error('新缓存必须以 loop- 开头');
  fs.mkdirSync(path.dirname(file),{recursive:true});
  const serialized=JSON.stringify(encodeCheckpoint(data))+'\n';fs.writeFileSync(file+'.tmp',serialized);
  for(let retry=0;retry<=5;retry++){
    try{rename(file+'.tmp',file);return file;}catch(error){
      if(!['EPERM','EBUSY','EACCES'].includes(error.code))throw error;
      if(retry<5){sleep(1000);continue;}
      const stamp=Date.now();let fallback=file.replace(/\.json$/,'')+'.'+stamp+'.json',suffix=0;
      while(fs.existsSync(fallback))fallback=file.replace(/\.json$/,'')+'.'+stamp+'-'+(++suffix)+'.json';
      fs.writeFileSync(fallback,serialized,{flag:'wx'});
      onFallback({event:'json-rename-fallback',requested:file,written:fallback,code:error.code,retries:5});return fallback;
    }
  }
}
export function restLevels(S,rates,e2Rates=rates){
  const failures=[],wideFailures=[];
  for(const [name,value] of Object.entries(rates)){
    const type=name.split(':')[0],committed=type==='E2'&&S.e2.labels[Number(name.split(':')[1])]!==null;
    const rate=type==='E2'?e2Rates[name]:value;
    const range=type==='E1'?[2,5]:type==='I1'?[5,20]:['N','M'].includes(type)?[2,5]:type==='E2'?(committed?[.5,2]:[0,1]):type==='L'?[0,.5]:null;
    const wide=type==='E1'?[1.5,6]:['N','M'].includes(type)?[1,8]:type==='S'?[0,40]:null;
    if(range&&!(Number.isFinite(rate)&&rate>=range[0]&&rate<=range[1]))failures.push({name,rate,range});
    if(wide&&!(Number.isFinite(value)&&value>=wide[0]&&value<=wide[1]))wideFailures.push({name,rate:value,range:wide});
  }
  return {restInRange:failures.length===0,restWideRange:wideFailures.length===0,selectionRestingFailures:failures,wideFailures};
}
/** W4 用完整 3 s 基线；E2 单独以 10 s 静息测量（含基线和 θ）。 */
export function measureRest(S){
  const rest=S.rest(3000),baseline=structuredClone(S.baseline),start=S.net.t;
  const bins=binGroups(S.net,S.e1.featureIds,rest.start,3000,20),ignited=Array.from({length:150},(_,i)=>Object.values(bins).some(row=>row[i]>10));
  const ignition={fraction:ignited.filter(Boolean).length/150,bins:ignited,binMs:20,thresholdHz:10,rule:'any E1 feature group >10 Hz; v2 restMetrics'};
  S.run(10000,{learn:false,e1Learn:false,prototypeLearn:false,tdLearn:false});
  const groups=Object.fromEntries(S.e2.units.map((ids,k)=>['E2:'+k,ids]));
  const rates=Object.fromEntries(Object.entries(groups).map(([name,ids])=>[name,groupRate(S.net,ids,start,S.net.t)]));
  const e2Baseline=baselines(S.net,groups,start,10000);Object.assign(S.baseline,e2Baseline);
  const e2Rest={start,end:S.net.t,durationSeconds:10,rates,baseline:e2Baseline,units:S.e2.labels.map((label,k)=>({k,label,committed:label!==null,hz:rates['E2:'+k]}))};
  return {rest:{...rest,baseline},e2Rest,ignition,...restLevels(S,rest.rates,rates)};
}
function inhibitionScope(S,multiplier){
  if(!Number.isFinite(multiplier)||multiplier<1||multiplier>5)throw new Error('ISTDP η 倍数超出 [1,5]');
  const groups=S.net.istdpGroups??[],original=groups.map(g=>g.istdp.eta);
  return {original,apply:()=>groups.forEach((g,i)=>{g.istdp.eta=original[i]*multiplier;}),restore:()=>groups.forEach((g,i)=>{g.istdp.eta=original[i];})};
}
function enableInhibition(S){
  for(const g of S.net.groups)if(g.plastic)g.frozen=true;else if(g.istdp)g.frozen=!!g.e2Idle;
  S.e2.learner.enabled=false;
  for(const learner of Object.values(S.expect.learners))learner.enabled=false;
}
function groupRates(S,start,ms){
  return Object.fromEntries(Object.entries(S.groups).map(([name,ids])=>[name,groupRate(S.net,ids,start,start+ms)]));
}
export function convergence(curve,groupSizes){
  const tail=curve.slice(-40),names=Object.keys(tail[0]?.rates??{}).filter(name=>['E1','N','M','S'].includes(name.split(':')[0]));
  if(tail.length<40)return {rule:'adjacent-20s-means<=max(0.3Hz,3SE)',ok:false,changes:{}};
  const changes=Object.fromEntries(names.map(name=>{
    const first=tail.slice(0,20).reduce((s,r)=>s+r.rates[name],0)/20,second=tail.slice(20).reduce((s,r)=>s+r.rates[name],0)/20;
    return [name,poissonComparison(first,second,groupSizes[name],20,20)];
  }));
  return {rule:'adjacent-20s-means<=max(0.3Hz,3SE)',ok:Object.values(changes).every(r=>r.ok),changes};
}
function identity(S,seed,etaMultiplier,features=null){
  return stateHash({version:5,scene:S.spec.scene,seed,config:S.config,features,background:S.background,labels:S.e2.labels,recurrent:S.e2.handles.recurrent.map(g=>g.scale[0]),e1:S.cfg,inhibition:S.net.istdpGroups.map(g=>g.istdp),etaMultiplier});
}
function restore(S,file,resume,protocol,key){
  if(!file||!resume||!checkpointExists(file))return null;
  const checkpoint=readCheckpoint(file);
  if(checkpoint.schema!==3||checkpoint.protocol!==protocol||checkpoint.identity!==key)throw new Error('协议检查点参数不匹配');
  S.loadState(checkpoint.state);return checkpoint;
}
/** maxChunks 仅暂停以便精确续算，不改变 60–300 s 的判据。 */
export function warmup(S,{seed=S.seed,etaMultiplier=1,checkpointPath=null,resume=true,checkpointEvery=5,maxChunks=Infinity,onChunk=null}={}){
  if(seed!==S.seed)throw new Error('预热种子须与构建器一致；输入噪声沿用完整网络 RNG 状态');
  const scope=inhibitionScope(S,etaMultiplier),key=identity(S,seed,etaMultiplier),checkpoint=restore(S,checkpointPath,resume,'warmup',key);
  const start=checkpoint?.start??S.net.t,curve=checkpoint?.curve??[],checks=checkpoint?.checks??[],sizes=Object.fromEntries(Object.entries(S.groups).map(([name,ids])=>[name,ids.length]));
  let done=!!checkpoint?.done,converged=!!checkpoint?.converged,last=checkpoint?.convergence??null,chunks=0;
  enableInhibition(S);scope.apply();
  const save=()=>{
    scope.restore();const state=S.saveState();scope.apply();
    writeCheckpoint(checkpointPath,{schema:3,protocol:'warmup',identity:key,seed,etaMultiplier,originalEtas:scope.original,start,curve,checks,done,converged,convergence:last,state});
  };
  try{
    while(!done&&chunks<maxChunks){
      const from=S.net.t;S.run(1000,ONLY_INHIBITION);chunks++;
      curve.push({second:curve.length+1,start:from,end:S.net.t,rates:groupRates(S,from,1000)});
      if(curve.length>=60&&curve.length%10===0){last=convergence(curve,sizes);checks.push({second:curve.length,...last});converged=last.ok;}
      done=converged||curve.length>=300;
      if(done||curve.length%checkpointEvery===0)save();
      onChunk?.(curve.at(-1),{done,converged,convergence:last,etaMultiplier});
    }
    if(!done)save();
  }finally{scope.restore();S.run(0,{learn:false,prototypeLearn:false,tdLearn:false});}
  return {protocol:'warmup',seed,etaMultiplier,originalEtas:scope.original,start,end:S.net.t,seconds:curve.length,done,converged,convergence:last,checks,curve};
}
/** 每特征 30 次，顺序和休息间隔由可恢复的独立 RNG 决定。 */
export function trainPredictions(S,scene=S.spec.scene,{seed=S.seed,etaMultiplier=1,features=S.spec.features,checkpointPath=null,resume=true,checkpointEvery=1,maxChunks=Infinity,onChunk=null}={}){
  if(scene!==S.spec.scene)throw new Error('预测抵消场景不匹配');
  if(!Array.isArray(features)||features.length===0||new Set(features).size!==features.length||features.some(g=>!S.spec.features.includes(g)))throw new Error('预测抵消测试组无效');
  const scope=inhibitionScope(S,etaMultiplier),key=identity(S,seed,etaMultiplier,features),checkpoint=restore(S,checkpointPath,resume,'trainPredictions',key);
  const rng=resumableRng(seed),stimuli=[];
  for(const feature of features)for(const ms of [1000,3000,10000])for(let trial=0;trial<10;trial++)stimuli.push({features:[feature],trial,ms});
  rng.shuffle(stimuli);
  const schedule=checkpoint?.schedule??stimuli.map(row=>({...row,restMs:1000+1000*rng.uniform()}));
  const start=checkpoint?.start??S.net.t,curve=checkpoint?.curve??[];
  let index=checkpoint?.index??0,chunks=0;
  enableInhibition(S);scope.apply();
  const save=()=>{
    scope.restore();const state=S.saveState();scope.apply();
    writeCheckpoint(checkpointPath,{schema:3,protocol:'trainPredictions',identity:key,seed,etaMultiplier,originalEtas:scope.original,start,index,schedule,curve,done:index===schedule.length,state});
  };
  try{
    while(index<schedule.length&&chunks<maxChunks){
      const stimulus=schedule[index],from=S.net.t;
      S.run(stimulus.ms,{...ONLY_INHIBITION,features:stimulus.features});
      const during=groupRates(S,from,stimulus.ms),restStart=S.net.t;
      S.run(stimulus.restMs,ONLY_INHIBITION);
      curve.push({index,stimulus,start:from,end:S.net.t,during,rest:groupRates(S,restStart,S.net.t-restStart)});
      index++;chunks++;
      if(index%checkpointEvery===0||index===schedule.length)save();
      onChunk?.(curve.at(-1),{done:index===schedule.length,index,total:schedule.length,etaMultiplier});
    }
    if(index<schedule.length)save();
  }finally{scope.restore();S.run(0,{learn:false,prototypeLearn:false,tdLearn:false});}
  return {protocol:'trainPredictions',seed,scene,etaMultiplier,originalEtas:scope.original,start,end:S.net.t,index,total:schedule.length,done:index===schedule.length,schedule,curve};
}
