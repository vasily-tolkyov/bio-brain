/** 自己的动作也是证据，且有独立触发通道。 */
import {pairs,moduleInterface} from './readout.mjs';
export const actionPopulations=actions=>actions.map(g=>({name:'A:'+g,n:20,p:{b:20,tauW:300}}));
export function buildActions(net,{actions}){
  const ids=Object.fromEntries(actions.map(g=>[g,net.idx('A:'+g)]));
  const inputs=Object.fromEntries(actions.map(g=>[g,[]])),state=Object.fromEntries(actions.map(g=>[g,{active:false,onset:null}]));
  let drive=()=>0,background=()=>0;
  const module=moduleInterface(net,{ids,handles:{},onStepWritesResources:false,
    setDrive:(rate,base)=>{drive=rate;background=base;},registerInput:(g,group)=>inputs[g].push(group),
    saveState:()=>structuredClone(state),loadState:saved=>{for(const g of actions)Object.assign(state[g],saved[g]);},
    selfTest:()=>actions.every(g=>ids[g].length===20),
  });
  // 动作传出副本只传递开始的 50 ms；持续动作和结束不再触发 L。
  module.updateGate=t=>{for(const g of actions){const row=state[g],active=drive(ids[g][0],t)>background(ids[g][0],t);if(active&&!row.active)row.onset=t;row.active=active;const open=active&&t-row.onset<50-1e-8;for(const group of inputs[g])group.scale.fill(open?3:0);}};
  module.onStep=t=>module.updateGate(t);
  return module;
}
export function connectActions(net,actions,context){
  for(const [g,ids] of Object.entries(actions.ids)){
    const target=context.L[g+':+'];if(!target)throw new Error('动作没有装入群');
    actions.handles[g]=net.connect({pairs:pairs(net,ids,target,.5),contacts:10,pRelease:.3,tauRec:100,scale:3});actions.registerInput(g,actions.handles[g]);
  }
}
