/** 第一级误差：静态预测的慢中继和证据中继；P 不接这些群。 */
import {PC_DEFAULTS} from '../L4-network/layered-network.mjs';
import {moduleInterface} from './readout.mjs';
// 架构 7.2：NMDA 的积分电导 7.025 × −60 mV 镁阻断 0.080，取设计值 0.56。
export const NMDA_REST_CHARGE=0.56;
export function slowCharge(tauR,tauD){
  if(!Number.isFinite(tauR)||!Number.isFinite(tauD)||tauR<=0||tauD<=tauR)throw new Error('慢受体时间常数必须满足 0 < τR < τD');
  const tp=tauR*tauD/(tauD-tauR)*Math.log(tauD/tauR),k=1/(Math.exp(-tp/tauD)-Math.exp(-tp/tauR));
  return .3*(tauD-tauR)*k;
}
export const slowStrength=(sEq,tauR,tauD)=>sEq*NMDA_REST_CHARGE/slowCharge(tauR,tauD);
export function errorPopulations(features,relaySize=40){
  return features.flatMap(g=>[
    {name:'N:'+g,n:20,p:PC_DEFAULTS.nP},{name:'S:'+g,n:relaySize,p:PC_DEFAULTS.sP},
    {name:'M:'+g,n:20,p:PC_DEFAULTS.nP},{name:'V:'+g,n:relaySize,p:PC_DEFAULTS.sP},
  ]);
}
export function buildErrors(net,{features,e1,byIndegree,localSEq=2,localMEq=localSEq,localTauR=25,relaySize=40}){
  const N={},S={},M={},V={},handles={evidence:[],relay:[],negativeRelay:[],localPrediction:[],localNegative:[]};
  for(const g of features){N[g]=net.idx('N:'+g);S[g]=net.idx('S:'+g);M[g]=net.idx('M:'+g);V[g]=net.idx('V:'+g);}
  for(const g of features){
    handles.evidence.push(byIndegree({pre:e1.featureIds[g],post:N[g],route:'evidence',strength:3}));
    handles.evidence.push(byIndegree({pre:e1.featureIds[g],post:V[g],route:'negativeEvidence',strength:3}));
    for(const [route,post,sEq,list] of [['localPrediction',S[g],localSEq,handles.localPrediction],['localNegative',M[g],localMEq,handles.localNegative]]){
      const tauR=localTauR,tauD=3*tauR,strength=slowStrength(sEq,tauR,tauD);
      const group=byIndegree({pre:e1.featureIds[g],post,route,kind:'slow',tauR,tauD,strength});
      group.loopSlow={method:'charge-equivalent',feature:g,tauR,tauD,sEq,sSlow:strength,referenceCharge:NMDA_REST_CHARGE,charge:slowCharge(tauR,tauD)};
      list.push(group);
    }
  }
  for(const g of features)handles.relay.push(byIndegree({pre:S[g],post:N[g],route:'predictionInhibition',kind:'inh',strength:.5,istdp:{eta:.02,target:3,wMax:30,tauPre:0}}));
  for(const g of features)handles.negativeRelay.push(byIndegree({pre:V[g],post:M[g],route:'negativeInhibition',kind:'inh',strength:.5,istdp:{eta:.02,target:3,wMax:30,tauPre:0}}));
  const ids=Object.fromEntries(features.flatMap(g=>[['N:'+g,N[g]],['S:'+g,S[g]],['M:'+g,M[g]],['V:'+g,V[g]]]));
  return moduleInterface(net,{ids,handles,N,S,M,V,
    saveState:()=>({relay:handles.relay.map(g=>[...g.scale]),negativeRelay:handles.negativeRelay.map(g=>[...g.scale])}),
    loadState:state=>{state.relay.forEach((a,i)=>handles.relay[i].scale.set(a));state.negativeRelay.forEach((a,i)=>handles.negativeRelay[i].scale.set(a));},
    selfTest:()=>features.every(g=>N[g].length===20&&M[g].length===20&&S[g].length===relaySize&&V[g].length===relaySize)&&[...handles.relay,...handles.negativeRelay].every(g=>g.loopRoute.designTauPre===0&&g.istdp.tauPre==null&&g.istdp.target===3&&g.istdp.tau===20)&&[...handles.localPrediction,...handles.localNegative].every(g=>g.kind==='slow'&&g.slow.tauR===localTauR&&g.slow.tauD===3*localTauR&&!g.plastic&&!g.istdp&&g.loopRoute.reference===24&&g.scale.every(x=>Math.abs(x-slowStrength(g.loopSlow.sEq,localTauR,3*localTauR)*g.loopRoute.multiplier)<=1e-12)),
  });
}
