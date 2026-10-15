/** 统一的群读出与模块接口；读出只分析真实放电记录。 */
import {createHash} from 'node:crypto';
export const channelKey=({g,c='+'})=>g+':'+c;
export function pairs(net,pre,post,p=1,{autapses=false}={}){
  const out=[];for(const a of pre)for(const b of post)if((autapses||a!==b)&&net.rng.uniform()<p)out.push([a,b]);return out;
}
export function groupRate(net,ids,start,end){
  let count=0;for(const i of ids)for(let q=net.spikes[i].length-1;q>=0&&net.spikes[i][q]>=start-1e-8;q--)if(net.spikes[i][q]<end-1e-8)count++;
  return ids.length&&end>start?count*1000/(ids.length*(end-start)):0;
}
export const binMsFor=ids=>ids.length<40?20:10;
export function binGroups(net,groups,start,ms,binMs=null){
  const out={};
  for(const [name,ids] of Object.entries(groups)){
    const width=binMs??binMsFor(ids),n=Math.round(ms/width);
    const bins=new Float64Array(n);
    for(const i of ids)for(let q=net.spikes[i].length-1;q>=0&&net.spikes[i][q]>=start-1e-8;q--){
      const t=net.spikes[i][q],b=Math.floor((t-start+1e-8)/width);
      if(t<start+ms-1e-8&&b>=0&&b<n)bins[b]+=1000/(ids.length*width);
    }
    out[name]=[...bins];
  }
  return out;
}
export function baselines(net,groups,start,ms=2000){
  return Object.fromEntries(Object.entries(binGroups(net,groups,start,ms)).map(([name,bins])=>{
    const mu=bins.reduce((s,x)=>s+x,0)/(bins.length||1),sigma=Math.sqrt(bins.reduce((s,x)=>s+(x-mu)**2,0)/(bins.length||1));
    const theta=Math.max(mu+4*sigma,mu+5);
    return [name,{mu,sigma,theta,binMs:binMsFor(groups[name]),neurons:groups[name].length,durationSeconds:ms/1000,restExceedFraction:bins.filter(v=>v>theta).length/(bins.length||1)}];
  }));
}
export function moduleInterface(net,{ids,handles={},saveState=()=>({}),loadState=()=>{},selfTest=()=>true,...other}){
  return {ids,handles,readout:(start,ms,binMs=null)=>binGroups(net,ids,start,ms,binMs),saveState,loadState,selfTest,...other};
}
export function poissonComparison(first,second,n,firstSeconds,secondSeconds,tolerance=.3){
  if(!(n>0&&firstSeconds>0&&secondSeconds>0)||![first,second].every(v=>Number.isFinite(v)&&v>=0))throw new Error('泊松比较参数无效');
  const se=Math.sqrt(first/(n*firstSeconds)+second/(n*secondSeconds)),difference=Math.abs(first-second),threeSE=3*se,limit=Math.max(tolerance,threeSE);
  return {first,second,n,firstSeconds,secondSeconds,difference,se,threeSE,limit,ratio:difference/limit,ok:difference<=limit};
}
export function meanPsth(trials){
  if(!trials.length||trials.some(row=>row.length!==trials[0].length))throw new Error('PSTH 试次长度不一致');
  return trials[0].map((_,k)=>trials.reduce((s,row)=>s+row[k],0)/trials.length);
}
export function excessFraction(bins,baseline,allowance=.01){
  const fraction=bins.filter(v=>v>baseline.theta).length/(bins.length||1),limit=2*baseline.restExceedFraction+allowance;
  return {fraction,restFraction:baseline.restExceedFraction,limit,ok:fraction<=limit};
}
export function connectionHash(net){
  const h=createHash('sha256');
  h.update(JSON.stringify(Object.entries(net.pops).map(([name,p])=>[name,p.start,p.n])));
  for(const g of net.groups){h.update(JSON.stringify([g.kind,g.target,g.contacts,g.pRelease,g.tauRec,g.slow?.tauR??null,g.slow?.tauD??null]));for(const a of [g.pre,g.post,g.scale])h.update(Buffer.from(a.buffer,a.byteOffset,a.byteLength));}
  return h.digest('hex');
}
export const stateHash=state=>createHash('sha256').update(JSON.stringify(state)).digest('hex');

/** 半开窗口；等频或下降都读为消失。只读完整脉冲历史。 */
export function boundaryDirection(net,e1Ids,t){return groupRate(net,e1Ids,t,t+50)>groupRate(net,e1Ids,t-100,t)?'+':'−';}
/** 任意 50 ms 内第 5 次脉冲为装入时刻，下一窗口从本窗口末端+20 ms起。 */
export function loadEvents(spikes,{windowMs=50,minSpikes=5}={}){
  if(!(windowMs>0)||!Number.isInteger(minSpikes)||minSpikes<1)throw new Error('装入窗口参数无效');
  const times=spikes.flat().slice().sort((a,b)=>a-b),events=[];
  if(times.some(t=>!Number.isFinite(t)))throw new Error('装入脉冲含非有限值');
  for(let i=0;i<times.length;){
    const start=times[i],end=start+windowMs;
    if(i+minSpikes>times.length||times[i+minSpikes-1]>=end-1e-8){i++;continue;}
    let j=i+minSpikes;while(j<times.length&&times[j]<end-1e-8)j++;
    events.push({time:times[i+minSpikes-1],start,end,strength:j-i});
    i=j;while(i<times.length&&times[i]<end+20-1e-8)i++;
  }
  return events;
}
