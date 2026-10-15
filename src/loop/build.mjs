/** 按场景在一个网络里构建闭环各群；不扩展核心的群 API。 */
import {SpikingNetwork} from '../L4-network/spiking-network.mjs';
import {StepHooks} from '../L4-network/step-hooks.mjs';
import {mammalianCosts} from '../L1-neuron/hh.mjs';
import {E1_DEFAULTS,RULES,groupOf,setGain,e1Configuration} from './e1.mjs';
import {sceneSpec,scriptedRate} from './world.mjs';
import {e2Populations,buildE2} from './e2.mjs';
import {errorPopulations,buildErrors} from './errors.mjs';
export {slowCharge,slowStrength,NMDA_REST_CHARGE} from './errors.mjs';
import {contextPopulations,shadowLPopulations,buildContext} from './context.mjs';
import {expectPopulations,buildExpect} from './expect.mjs';
import {actionPopulations,buildActions,connectActions} from './actions.mjs';
import {moduleInterface,baselines,groupRate,connectionHash,pairs} from './readout.mjs';
export const BACKGROUNDS=Object.freeze({E1:30,I1:12,E2:18,I2:12,J:9,N:14,S:9,M:12,V:9,C:0,L:2,P:0,Q:0,Z:0,D:0,Up:0,Down:0,IUp:0,IDown:0,A:0});
export const BACKGROUND_RANGES=Object.freeze({E1:[20,32],I1:[8,16],N:[10,18],M:[8,16],S:[6,20],V:[6,12],C:[0,20],L:[0,6],P:[0,20],Up:[0,20],Down:[0,20],E2:[10,20]});
export const INDEGREES=Object.freeze({evidence:16,predictionInhibition:50,prediction:24,up:8,negativeEvidence:16,negativeInhibition:10,negative:24,recognition:8,feedforwardInhibition:40,recurrent:16,localPrediction:24,localNegative:24,load:7});
export const SHARED_ROUTES=Object.freeze(['evidence','negativeEvidence','predictionInhibition','negativeInhibition','prediction','negative','localPrediction','localNegative','load']);
/** 第 4.6 节：受限通路 k/n_pre≤0.35，按 k_ref/k 补偿，逐边随机抽样。 */
export function connectByIndegree(net,{pre,post,route,strength,...rest}){
  const reference=INDEGREES[route];
  if(!reference||!pre.length||!post.length)throw new Error('入度通路或群无效');
  const sharingLimited=SHARED_ROUTES.includes(route),k=Math.min(reference,pre.length,sharingLimited?Math.floor(.35*pre.length):Infinity);
  if(k<1)throw new Error('共享上限下没有可用入度');
  const p=k/pre.length,multiplier=reference/k;
  const designTauPre=rest.istdp?.tauPre??null;
  // 核心的缺省值才表示两条 20 ms 短迹；显式 0 会把 alpha 少算一半。
  // 第 6 节的“关闭长迹”在接口边界转为缺省短窗，target/eta/tau 均不变。
  if(rest.istdp?.tauPre===0){rest.istdp={...rest.istdp};delete rest.istdp.tauPre;}
  const group=net.connect({pairs:pairs(net,pre,post,p),contacts:10,pRelease:.3,tauRec:100,...rest,scale:strength*multiplier});
  group.loopRoute={route,reference,k_ref:reference,k,n_pre:pre.length,nPre:pre.length,nPost:post.length,p,sharingRatio:p,sharingLimited,multiplier,compensation:multiplier,strength,designTauPre};
  return group;
}
export function indegreeTable(net){
  return Object.entries(INDEGREES).map(([route,reference])=>{
    const groups=net.groups.filter(g=>g.loopRoute?.route===route),posts=groups.reduce((n,g)=>n+g.loopRoute.nPost,0);
    const details=groups.map(g=>({...g.loopRoute,meanIndegree:g.pre.length/g.loopRoute.nPost,scaleMin:Math.min(...g.scale),scaleMax:Math.max(...g.scale)}));
    const meanIndegree=posts?groups.reduce((n,g)=>n+g.pre.length,0)/posts:null;
    const expectedIndegree=posts?groups.reduce((n,g)=>n+g.loopRoute.k*g.loopRoute.nPost,0)/posts:null;
    return {route,reference,meanIndegree,expectedIndegree,groups:groups.length,details};
  });
}
function evidence(net,cfg,features){
  const E=net.idx('E1'),I=net.idx('I1'),pairs=[];
  for(let a=0;a<cfg.NE;a++)for(let b=0;b<cfg.NE;b++){
    if(a===b)continue;const ga=groupOf(a,cfg),gb=groupOf(b,cfg);
    if(net.rng.uniform()<(ga>=0&&ga===gb?cfg.pIn:cfg.pOut))pairs.push([a,b]);
  }
  const EE=net.connect({pairs,contacts:10,pRelease:.3,tauRec:100,plastic:{sMin:.1,sMax:9,rho0:.1,accel:5,budget:.2}});
  const EI=net.connect({from:'E1',to:'I1',p:cfg.pEI,contacts:10,pRelease:.3,tauRec:100,scale:1.5});
  const IE=net.connect({from:'I1',to:'E1',p:cfg.pIE,contacts:10,pRelease:.5,tauRec:300,kind:'inhFast',scale:4,istdp:{eta:.02,target:3,wMax:40}});
  const II=net.connect({from:'I1',to:'I1',p:cfg.pII,contacts:10,pRelease:.5,tauRec:300,kind:'inhFast',scale:4});
  const featureIds=Object.fromEntries(features.map((g,k)=>[g,E.slice(k*80,(k+1)*80)])),S={net,cfg,E,I,EE,IE,G:.6,b:20};
  setGain(S,.6);EE.frozen=true;
  return Object.assign(S,moduleInterface(net,{ids:{E1:E,I1:I,...Object.fromEntries(features.map(g=>['E1:'+g,featureIds[g]]))},handles:{EE,EI,IE,II},featureIds,
    saveState:()=>({G:S.G,b:S.b,rho:[...EE.rho],scale:[...EE.scale],IE:[...IE.scale],plastic:{...EE.plastic}}),
    loadState:state=>{S.G=state.G;S.b=state.b;EE.rho.set(state.rho);EE.scale.set(state.scale);IE.scale.set(state.IE);Object.assign(EE.plastic,state.plastic);},
    selfTest:()=>E.length===Math.max(720,80*(features.length+1))&&I.length===E.length/4&&features.every(g=>featureIds[g].length===80),
  }));
}
export function buildLoop({scene=1,seed=1,background={},presets=null,e2Recurrent=4.5,localSEq=2,localMEq=localSEq,localTauR=25,relaySize=40,e1Adaptation=20,loadGain=1,shadowL=[],ablateE2Prediction=false,lToC=.2,tdEta=1e-5,tdMax=6,cancelScale=3,extensions=[],includeExpect=true}={}){
  if(!Number.isFinite(e2Recurrent)||e2Recurrent<2||e2Recurrent>6)throw new Error('E2 递归超出 [2,6]');
  if(!Number.isFinite(localSEq)||localSEq<.5||localSEq>4)throw new Error('S 局部保持等效强度超出 [0.5,4]');
  if(!Number.isFinite(localMEq)||localMEq<localSEq||localMEq>2*localSEq)throw new Error('M/S 局部保持等效强度比超出 [1,2]');
  if(!Number.isFinite(localTauR)||localTauR<15||localTauR>40)throw new Error('局部保持 τR 超出 [15,40] ms');
  if(!Number.isInteger(relaySize)||relaySize<40||relaySize>80)throw new Error('中继规模超出 [40,80]');
  if(!Number.isFinite(e1Adaptation)||e1Adaptation<5||e1Adaptation>20)throw new Error('E1 适应超出 [5,20] pA');
  if(!Number.isFinite(loadGain)||loadGain<.5||loadGain>2)throw new Error('L 增益超出 [0.5,2]');
  if(!Array.isArray(shadowL)||new Set(shadowL).size!==shadowL.length||shadowL.some(v=>!Number.isFinite(v)||v<.5||v>2||v===loadGain))throw new Error('影子 L 增益无效或重复正式增益');
  if(typeof ablateE2Prediction!=='boolean')throw new Error('E2 消融开关须为布尔值');
  const spec=sceneSpec(scene),K=spec.features.length,cfg=e1Configuration({...E1_DEFAULTS,K,seed},'v2'),{NE,NI}=cfg;
  const pops=[{name:'E1',n:NE,p:{b:e1Adaptation,tauW:300}},{name:'I1',n:NI,p:{C:100,tRef:1}},
    ...e2Populations(spec.e2Units),...errorPopulations(spec.features,relaySize),...contextPopulations(spec.context,spec.outputs),...(includeExpect?expectPopulations(spec.outputs):[]),...actionPopulations(spec.actions),...extensions.flatMap(extension=>extension.populations({spec,cfg})),...shadowLPopulations(spec.context,spec.outputs,shadowL)];
  const net=new SpikingNetwork({pops,costs:mammalianCosts(),seed,dt:.1,plasticity:RULES.GB_RATE});
  const byIndegree=opts=>connectByIndegree(net,opts);
  const e1=evidence(net,cfg,spec.features);e1.b=e1Adaptation;
  const errors=buildErrors(net,{features:spec.features,e1,byIndegree,localSEq,localMEq,localTauR,relaySize});
  const e2=buildE2(net,{features:spec.features,units:spec.e2Units,e1,errors,presets:presets??spec.prototypes,byIndegree,recurrentStrength:e2Recurrent,ablatePrediction:ablateE2Prediction});
  const context=buildContext(net,{channels:spec.context,outputs:spec.outputs,errors,byIndegree,lToC,loadGain}),actions=buildActions(net,{actions:spec.actions});connectActions(net,actions,context);
  const expect=buildExpect(net,{outputs:includeExpect?spec.outputs:[],context,eta:tdEta,wMax:tdMax,cancelScale});
  const hooks=new StepHooks();hooks.add('prototype',e2.learner);hooks.add('actions',actions);
  for(const [key,learner] of Object.entries(expect.learners))hooks.add('td:'+key,learner);
  const modules={e1,e2,errors,context,actions,expect};
  const config={localSEq,localMEq,localTauR,relaySize,e1Adaptation,loadGain,shadowL:shadowL.slice(),ablateE2Prediction,e2Recurrent,includeExpect,lToC,tdEta,tdMax,cancelScale};
  const S={net,cfg,spec,seed,config,modules,e1,e2,errors,context,actions,expect,hooks,background:{...BACKGROUNDS,...Object.assign({},...extensions.map(extension=>extension.background??{}))},baseline:{}};
  for(const extension of extensions){
    if(!extension.name||modules[extension.name])throw new Error('扩展模块名称重复或为空');
    const module=extension.build(net,S);modules[extension.name]=module;
    if(typeof module.onStep==='function')hooks.add(extension.name,module);
  }
  if(shadowL.length)context.addShadows(shadowL,actions,expect);
  const groups=S.groups=Object.assign({},...Object.values(modules).map(module=>module.ids));
  S.setBackground=values=>{
    for(const [name,value] of Object.entries(values)){
      const bounds=BACKGROUND_RANGES[name];
      if(!bounds||!Number.isFinite(value)||value<bounds[0]||value>bounds[1])throw new Error('背景超出标定范围：'+name);
      S.background[name]=value;
    }
    return S;
  };
  S.setBackground(background);
  S.backgroundRate=()=>{
    const rates=new Float64Array(net.N);
    for(const [name,pop] of Object.entries(net.pops))rates.fill(S.background[name.startsWith('ShadowL:')?'L':name.split(':')[0]]??0,pop.start,pop.start+pop.n);
    return i=>rates[i];
  };
  S.run=(ms,{features=[],actions:activeActions=[],hz=58,drives=null,script=null,learn=false,e1Learn=false,prototypeLearn=learn,tdLearn=learn,extra=null}={})=>{
    e1.EE.frozen=!learn||!e1Learn;if(e1.EE.frozen)e1.EE.z.fill(0);
    e2.learner.enabled=prototypeLearn;for(const learner of Object.values(expect.learners))learner.enabled=tdLearn;
    for(const extension of extensions)modules[extension.name].setLearning?.(learn);
    const backgroundRate=S.backgroundRate();let rate;
    if(script)rate=scriptedRate(script,e1.featureIds,{background:backgroundRate,actionIds:actions.ids});
    else{const rates=Float64Array.from({length:net.N},(_,i)=>backgroundRate(i));for(const g of features){if(!e1.featureIds[g])throw new Error('未知特征 '+g);for(const i of e1.featureIds[g])rates[i]=hz;}for(const g of activeActions){if(!actions.ids[g])throw new Error('未知动作 '+g);for(const i of actions.ids[g])rates[i]=hz;}if(drives)for(const [i,hz] of drives)rates[i]=hz;rate=i=>rates[i];}
    actions.setDrive(rate,backgroundRate);if(ms>0)actions.updateGate(net.t);
    net.run(ms,{rate,learn,extra,...hooks.runOptions()});return S;
  };
  S.measureBaseline=(start,ms=2000)=>{
    S.baseline=baselines(net,groups,start,ms);
    spec.features.forEach((g,k)=>e2.learner.featureTheta[k]=S.baseline['E1:'+g].theta);
    e2.units.forEach((_,k)=>e2.learner.unitTheta[k]=S.baseline['E2:'+k].theta);
    for(const [key,learner] of Object.entries(expect.learners)){learner.baselineUp=S.baseline['Up:'+key].mu;learner.baselineDown=S.baseline['Down:'+key].mu;}
    return S.baseline;
  };
  S.rest=(ms=2000)=>{const start=net.t;S.run(ms,{learn:false,prototypeLearn:false,tdLearn:false});S.measureBaseline(start,ms);return {start,end:net.t,rates:Object.fromEntries(Object.entries(groups).map(([name,ids])=>[name,groupRate(net,ids,start,start+ms)]))};};
  S.saveState=()=>({schema:1,scene:spec.scene,seed,config:{...config},net:net.exportState(),modules:Object.fromEntries(Object.entries(modules).map(([name,module])=>[name,module.saveState()])),hooks:hooks.exportState(),background:{...S.background},baseline:structuredClone(S.baseline)});
  S.loadState=state=>{
    if(state.schema!==1||state.scene!==spec.scene)throw new Error('场景状态不匹配');
    if(state.config&&JSON.stringify(state.config)!==JSON.stringify(config))throw new Error('局部保持或消融配置不匹配');
    net.importState(state.net);for(const [name,saved] of Object.entries(state.modules))modules[name].loadState(saved);hooks.importState(state.hooks);Object.assign(S.background,state.background);S.baseline=structuredClone(state.baseline);return S;
  };
  S.selfTest=()=>Object.values(modules).every(module=>module.selfTest());
  S.table=()=>({scene:spec.scene,config:{...config},neurons:net.N,populations:Object.entries(net.pops).map(([name,pop])=>({name,n:pop.n,start:pop.start})),connections:net.groups.map((g,index)=>({index,kind:g.kind,target:g.target,n:g.pre.length,plastic:!!g.plastic,istdp:!!g.istdp,istdpTarget:g.istdp?.target??null,tauR:g.slow?.tauR??null,tauD:g.slow?.tauD??null})),indegrees:indegreeTable(net),e1:{NE,NI,K,probabilities:{pIn:cfg.pIn,pOut:cfg.pOut,pEI:cfg.pEI,pIE:cfg.pIE,pII:cfg.pII},indegrees:e1Indegrees(e1)},hash:connectionHash(net)});
  return S;
}

function e1Indegrees(e1){
  const {EE,EI,IE,II}=e1.handles,{cfg}=e1,grouped=cfg.K*cfg.M;
  let within=0,outside=0;
  for(let k=0;k<EE.pre.length;k++)if(EE.post[k]<grouped){
    if(groupOf(EE.pre[k],cfg)===groupOf(EE.post[k],cfg))within++;else outside++;
  }
  return [{route:'within',reference:20,meanIndegree:within/grouped},
    {route:'between',reference:53,meanIndegree:outside/grouped},
    {route:'E1-I1',reference:144,meanIndegree:EI.pre.length/cfg.NI},
    {route:'I1-E1',reference:72,meanIndegree:IE.pre.length/cfg.NE},
    {route:'I1-I1',reference:72,meanIndegree:II.pre.length/cfg.NI}];
}
