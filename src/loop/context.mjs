/** 每个特征一个累加型 L；C 仅由真实突触输入激活。 */
import {channelKey,pairs,moduleInterface} from './readout.mjs';
import {slowCharge,NMDA_REST_CHARGE} from './errors.mjs';
export const DELAYS=Object.freeze(Array.from({length:8},(_,k)=>100*2**k));
// 架构 7.2 的电荷表：AMPA=0.78，静息附近 NMDA=0.56。
export const AMPA_CHARGE=.78;
export const LOAD_BASE_STRENGTH=3*(20/7)*(AMPA_CHARGE+NMDA_REST_CHARGE)/slowCharge(5,30);
const loadParameters=Object.freeze({C:100,b:200,tauW:50});
const loadChannels=(context,outputs)=>new Map([...context,...outputs].map(ch=>[channelKey(ch),ch]));
export function shadowLPopulations(context,outputs,gains=[]){
  return gains.flatMap(gain=>[...loadChannels(context,outputs).keys()].map(key=>({name:'ShadowL:'+gain+':'+key,n:10,p:loadParameters})));
}
export function contextPopulations(context,outputs){
  const out=[...loadChannels(context,outputs).keys()].map(key=>({name:'L:'+key,n:10,p:loadParameters}));
  for(const ch of context)for(let k=0;k<8;k++)out.push({name:'C:'+channelKey(ch)+':'+k,n:6,p:{C:200,tRef:2,b:0}});
  return out;
}
function configureLoad(net,ids){
  ids.forEach((id,q)=>{const p=net.par[id];net.par[id]={...p,...loadParameters,Vth:p.EL+30*(1+.1*(2*q/9-1))};});
}
export function buildContext(net,{channels,outputs,errors,byIndegree,lToC=.2,loadGain=1}){
  const all=loadChannels(channels,outputs),L={},C={},handles={load:[],slow:[],strengthen:[]},descriptors=[];
  const errorInputs=(key,ch,target,gain,shadow,list)=>{
    for(const [type,pre] of [['N',errors.N[ch.g]],['M',errors.M[ch.g]]]){
      const group=byIndegree({pre,post:target,route:'load',kind:'slow',tauR:5,tauD:30,strength:LOAD_BASE_STRENGTH*gain});
      group.loopLoad={key,input:type,gain,baseStrength:LOAD_BASE_STRENGTH,k_ref:7,shadow,method:'charge-equivalent',referenceCharge:AMPA_CHARGE+NMDA_REST_CHARGE,charge:slowCharge(5,30)};list.push(group);
    }
  };
  for(const [key,ch] of all){
    L[key]=net.idx('L:'+key);configureLoad(net,L[key]);
    if(errors.N[ch.g])errorInputs(key,ch,L[key],loadGain,false,handles.load);
  }
  for(const ch of channels){
    const key=channelKey(ch);C[key]=[];
    for(let k=0;k<8;k++){
      const ids=net.idx('C:'+key+':'+k);C[key].push(ids);
      for(let q=0;q<6;q++){const id=ids[q],p={...net.par[id]};p.Vth=p.EL+(p.Vth-p.EL)*(1+.1*(2*q/5-1));net.par[id]=p;}
      const tauR=DELAYS[k]/1.386,tauD=2*tauR;
      handles.slow.push(net.connect({pairs:pairs(net,L[key],ids,1),contacts:10,pRelease:.3,tauRec:100,kind:'slow',scale:lToC,tauR,tauD}));
    }
    descriptors.push({...ch,slots:C[key]});
  }
  const ids=Object.fromEntries([...Object.entries(L).map(([key,v])=>['L:'+key,v]),...Object.entries(C).flatMap(([key,slots])=>slots.map((v,k)=>['C:'+key+':'+k,v]))]);
  const shadows={},shadowHandles=[];
  const module=moduleInterface(net,{ids,handles,L,C,shadows,shadowHandles,channels:descriptors,
    saveState:()=>({slowScale:handles.slow.map(g=>[...g.scale])}),
    loadState:state=>state.slowScale.forEach((a,i)=>handles.slow[i].scale.set(a)),
    selfTest:()=>Object.values(L).every(ids=>ids.length===10)&&Object.values(C).every(slots=>slots.length===8&&slots.every(ids=>ids.length===6))&&handles.slow.every(g=>g.kind==='slow'&&!g.plastic)&&[...handles.load,...shadowHandles.filter(g=>g.loopLoad)].every(g=>g.kind==='slow'&&g.slow.tauR===5&&g.slow.tauD===30&&g.scale.every(x=>Math.abs(x-LOAD_BASE_STRENGTH*g.loopLoad.gain*g.loopRoute.multiplier)<=1e-12)&&!g.plastic&&!g.istdp),
  });
  // 正式拓扑构建完后独立抽样；每个影子有独立背景且没有任何输出。
  module.addShadows=(gains,actions,expect)=>{
    for(const gain of gains){
      const copy=shadows[gain]={};
      for(const [key,ch] of all){
        const target=net.idx('ShadowL:'+gain+':'+key);copy[key]=target;ids['ShadowL:'+gain+':'+key]=target;configureLoad(net,target);
        if(errors.N[ch.g])errorInputs(key,ch,target,gain,true,shadowHandles);
        else {
          const group=net.connect({pairs:pairs(net,actions.ids[ch.g],target,.5),contacts:10,pRelease:.3,tauRec:100,scale:3});
          actions.registerInput(ch.g,group);shadowHandles.push(group);
        }
        if(expect.Up[key])shadowHandles.push(net.connect({pairs:pairs(net,expect.Up[key],target,.25),contacts:10,pRelease:.3,tauRec:100,scale:1}));
      }
    }
  };
  return module;
}
