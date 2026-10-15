/** 架构第 13 节的确定性测试世界。所有事件时刻与时长以 ms 计。 */
import {makeRng} from '../core/rng.mjs';

export const WORLD_RATE = 58;
export const FEATURES = Object.freeze({
  1:['闪电','雷声','鸟叫'],
  2:['硬壳','皱纹','光滑','带帽','圆','大','小','褐色','绿色','灰色','树下','河边'],
  3:['硬','重','轻','长条','灰','红','硬壳','皱','光','软果','硬地','软地','裂开','烂了'],
  4:['红','圆','灌木','锯齿叶','光滑叶','裂叶','甜','苦'],
});
export const NUT_PROTOTYPES = Object.freeze({
  核桃:['硬壳','皱纹','大','褐色','树下'],
  榛子:['硬壳','光滑','小','褐色','树下'],
  橡子:['硬壳','带帽','小','绿色','树下'],
  卵石:['光滑','圆','小','灰色','河边'],
});
export const PRESET_PROTOTYPES = Object.freeze({
  3:{石头:['硬','重','灰'],木棍:['轻','长条'],核桃:['硬壳','皱'],浆果:['软果','红'],硬地面:['硬地'],软地面:['软地']},
  4:{红果子:['红','圆','灌木']},
});
const channel=(g,c='+')=>({g,c});
export function sceneSpec(scene) {
  scene=Number(scene);
  if(!FEATURES[scene])throw new Error('未知场景 '+scene);
  const actions=scene===3?['砸','摔']:scene===4?['吃']:[];
  const context=scene===1?FEATURES[1].map(g=>channel(g,'±')):actions.map(g=>channel(g));
  const outputs=scene===1?FEATURES[1].map(g=>channel(g,'±')):scene===3?['裂开','烂了'].map(g=>channel(g,'±')):scene===4?['甜','苦'].map(g=>channel(g,'±')):[];
  return {scene,features:FEATURES[scene].slice(),actions,context,outputs,e2Units:scene===2?16:12,prototypes:PRESET_PROTOTYPES[scene]??{}};
}
const event=(g,start,duration,extra={})=>({g,start,duration,...extra});
const key=features=>features.slice().sort().join('|');

export function eventScript(scene,seed,trials,extra={}) {
  const spec=sceneSpec(scene),events=trials.flatMap((trial,index)=>trial.events.map(e=>({...e,trial:index}))).sort((a,b)=>a.start-b.start||a.g.localeCompare(b.g,'zh'));
  for(const e of events) {
    if(!Number.isFinite(e.start)||e.start<0||!Number.isFinite(e.duration)||e.duration<=0)throw new Error('事件时刻无效');
    if(e.action ? !spec.actions.includes(e.action) : !spec.features.includes(e.g))throw new Error('事件通道不属于场景');
  }
  return {schema:1,scene,seed,rateHz:WORLD_RATE,spec,trials,events,duration:trials.at(-1)?.end??0,...extra};
}
/** 返回正在呈现的特征与动作。没有对网络状态或学习量的直接写入。 */
export function activeEvents(script,t) {
  const features=new Set(),actions=new Set();
  for(const e of script.events)if(t>=e.start&&t<e.start+e.duration)(e.action?actions:features).add(e.action??e.g);
  return {features,actions};
}
export function scriptedRate(script,featureIds,{background=()=>0,actionIds={}}={}) {
  const featureById=new Map(),actionById=new Map();
  for(const [g,ids] of Object.entries(featureIds))for(const id of ids)featureById.set(id,g);
  for(const [a,ids] of Object.entries(actionIds))for(const id of ids)actionById.set(id,a);
  let lastT=NaN,active;
  return (i,t)=>{
    if(t!==lastT){active=activeEvents(script,t);lastT=t;}
    if(active.features.has(featureById.get(i))||active.actions.has(actionById.get(i)))return script.rateHz;
    return typeof background==='function'?background(i,t):background;
  };
}

export function stormWorld({seed=1,set='A',probe=null,count=probe?10:set==='change'?30:40}={}) {
  if(!['A','B','change'].includes(set))throw new Error('未知雷雨训练集');
  const rng=makeRng(seed),trials=[];
  let time=0;
  const add=(index,kind,{delay,hasBird=false,frozen=false}={})=>{
    const start=time,events=[];
    if(kind!=='bird')events.push(event('闪电',start,100));
    if(kind==='storm'||kind==='early')events.push(event('雷声',start+delay,kind==='early'?1000:2000));
    if(hasBird||kind==='bird')events.push(event('鸟叫',start+(kind==='bird'?0:rng.uniform()*10000),200));
    const last=Math.max(...events.map(e=>e.start+e.duration)),rest=20000+5000*rng.uniform();
    time=last+rest;
    trials.push({index,kind,start,end:time,delay,hasBird:hasBird||kind==='bird',frozen,events});
  };
  if(probe){
    if(!['lightning','bird','early'].includes(probe))throw new Error('未知探测');
    for(let i=0;i<count;i++)add(i,probe==='lightning'?'lightning':probe==='bird'?'bird':'early',{delay:3000,frozen:true});
  }else for(let i=0;i<count;i++){
    const delay=set==='B'?1000+9000*rng.uniform():set==='change'?6000:3000;
    const hasBird=rng.uniform()<.5;
    add(i,'storm',{delay,hasBird});
    if(set==='change'&&i%2===1)add(i,'lightning',{frozen:true});
  }
  return eventScript(1,seed,trials,{set,probe});
}

export function nutWorld({seed=1,perKind=60}={}) {
  const rng=makeRng(seed),pools={},reserved={};
  for(const [kind,prototype] of Object.entries(NUT_PROTOTYPES)){
    const outside=FEATURES[2].filter(g=>!prototype.includes(g)),combinations=[];
    for(let omitted=0;omitted<5;omitted++)for(const g of outside)combinations.push([...prototype.filter((_,i)=>i!==omitted),g]);
    pools[kind]=rng.shuffle(combinations);reserved[kind]=pools[kind].slice(0,10).map(row=>row.slice());
  }
  // 跨种类排除相同组合，保证任何预留刺激都没有在训练阶段出现。
  const reservedKeys=new Set(Object.values(reserved).flat().map(key));
  const trainPools=Object.fromEntries(Object.entries(pools).map(([kind,rows])=>[kind,rows.filter(row=>!reservedKeys.has(key(row)))]));
  if(Object.values(trainPools).some(rows=>!rows.length))throw new Error('预留组合耗尽训练池');
  const order=rng.shuffle(Object.keys(NUT_PROTOTYPES).flatMap(kind=>Array(perKind).fill(kind)));
  let time=0;
  const trials=order.map((kind,index)=>{
    const available=trainPools[kind],features=available[rng.pick(available.length)].slice(),start=time;time+=3000;
    return {index,kind,features,start,end:time,frozen:false,events:features.map(g=>event(g,start,1000))};
  });
  const training=eventScript(2,seed,trials);
  const makeTest=partial=>{
    let t=0,index=0;const rows=[];
    for(const [kind,prototype] of Object.entries(NUT_PROTOTYPES))for(let i=0;i<10;i++){
      const features=partial?rng.shuffle(prototype.slice()).slice(0,3):reserved[kind][i].slice(),start=t;t+=3000;
      rows.push({index:index++,kind,features,start,end:t,frozen:true,events:features.map(g=>event(g,start,1000))});
    }
    return eventScript(2,seed,rows,{partial});
  };
  return {schema:1,scene:2,seed,prototypes:NUT_PROTOTYPES,reserved,training,test:makeTest(false),completion:makeTest(true)};
}

const STONE_KINDS=[
  {name:'石头砸核桃硬地',features:['硬','重','灰','硬壳','皱','硬地'],action:'砸',outcome:'裂开'},
  {name:'木棍砸核桃硬地',features:['轻','长条','硬壳','皱','硬地'],action:'砸',outcome:null},
  {name:'石头砸浆果硬地',features:['硬','重','灰','软果','红','硬地'],action:'砸',outcome:'烂了'},
  {name:'石头砸核桃软地',features:['硬','重','灰','硬壳','皱','软地'],action:'砸',outcome:null},
  {name:'核桃摔到硬地',features:['硬壳','皱','硬地'],action:'摔',outcome:'裂开'},
];
function actionTrial(kind,index,start,{frozen=false,lookOnly=false}={}) {
  const duration=lookOnly?1000:2200,events=kind.features.map(g=>event(g,start,duration));
  if(!lookOnly){
    events.push(event(kind.action,start+1000,100,{action:kind.action}));
    if(kind.outcome)events.push(event(kind.outcome,start+1200,1000));
  }
  return {index,kind:kind.name,features:kind.features.slice(),action:lookOnly?null:kind.action,outcome:lookOnly?null:kind.outcome,start,end:start+duration+4000,frozen,events};
}
export function stoneWorld({seed=1,perKind=35,test=false}={}) {
  const rng=makeRng(seed);
  const kinds=test?[
    {name:'新石头砸核桃硬地',features:['硬','重','红','硬壳','皱','硬地'],action:'砸',outcome:'裂开'},
    {name:'石头砸新坚果硬地',features:['硬','重','灰','硬壳','光','硬地'],action:'砸',outcome:'裂开'},
  ]:STONE_KINDS;
  const order=rng.shuffle(kinds.flatMap((kind)=>Array(test?10:perKind).fill(kind)));
  let time=0;
  const trials=order.map((kind,index)=>{const trial=actionTrial(kind,index,time,{frozen:test});time=trial.end;return trial;});
  return eventScript(3,seed,trials,{test});
}
export function fruitWorld({seed=1,phase=0,count=phase===0?60:150,control=false,test=false,lookOnly=phase===0}={}) {
  if(![0,1].includes(phase))throw new Error('未知果子阶段');
  const rng=makeRng(seed),leaves=['锯齿叶','光滑叶','裂叶'];
  const order=test?leaves.flatMap(leaf=>Array(10).fill(leaf)):Array.from({length:count},()=>leaves[rng.pick(3)]);
  let time=0;
  const trials=order.map((leaf,index)=>{
    const outcome=control?(rng.uniform()<1/3?'苦':'甜'):leaf==='光滑叶'?'苦':'甜';
    const kind={name:leaf,features:['红','圆','灌木',leaf],action:'吃',outcome};
    const trial=actionTrial(kind,index,time,{lookOnly,frozen:test});trial.leaf=leaf;time=trial.end;return trial;
  });
  return eventScript(4,seed,trials,{phase,control,test,lookOnly});
}
export function generateWorld(scene,options={}) {
  return ({1:stormWorld,2:nutWorld,3:stoneWorld,4:fruitWorld})[Number(scene)]?.(options)??(()=>{throw new Error('未知场景')})();
}
