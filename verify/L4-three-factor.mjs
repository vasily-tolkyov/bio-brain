/** L4j：三因素规则的手算、关闭与完整 JSON 续算。 */
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';
import {SpikingNetwork} from '../src/L4-network/spiking-network.mjs';
import {GB_RATE} from '../src/L3-plasticity/calcium-rule.mjs';
import {ThreeFactor} from '../src/L4-network/three-factor.mjs';
import {StepHooks} from '../src/L4-network/step-hooks.mjs';
import {checker} from '../src/core/check.mjs';
const ROOT=fileURLToPath(new URL('../',import.meta.url));
const digest=x=>createHash('sha256').update(JSON.stringify(x)).digest('hex');
const roundTrip=x=>JSON.parse(JSON.stringify(x));
function stub(options={}){
  const net={N:3,dt:.1,t:0},positive={pre:Int32Array.of(0),scale:Float64Array.of(.1)},negative={pre:Int32Array.of(0),scale:Float64Array.of(.1)};
  const td=new ThreeFactor({net,positive,negative,wMax:1,...options});return {net,positive,negative,td};
}
function network(enabled=null){
  const net=new SpikingNetwork({seed:19,costs:{atpPerSpike:1000,atpRestPerMs:100},plasticity:GB_RATE,
    pops:[{name:'C',n:2,p:{C:100,b:2}},{name:'P',n:2},{name:'Q',n:1,p:{C:100,tRef:1}},{name:'Up',n:1},{name:'Down',n:1}]});
  const positive=net.connect({from:'C',to:'P',p:1,scale:.6}),negative=net.connect({from:'C',to:'Q',p:1,scale:.6});
  net.connect({from:'Q',to:'P',p:1,kind:'inhFast',scale:2});
  net.connect({from:'C',to:'C',p:1,plastic:{sMin:.1,sMax:5,rho0:.2,accel:5,budget:.2}});
  const hooks=new StepHooks();
  const td=enabled===null?null:hooks.add('td',new ThreeFactor({net,positive,negative,up:net.idx('Up'),down:net.idx('Down'),eta:1e-6,wMax:6,enabled,initialize:false}));
  return {net,hooks,td};
}
const rate=(i,t)=>i<2?25:i===5?(t%200<100?25:8):i===6?(t%200<100?8:25):12;
function advance(S,ms){S.net.run(ms,{rate,learn:true,...(S.td?S.hooks.runOptions():{})});}
export function verifyL4ThreeFactor({write=true}={}){
  const c=checker('第 4 层 三因素时间差学习'),metrics={};
  const S=stub({eta:.01,factor:t=>t<2.5?2:-1}),pulses=[],updates=[];
  let expectedPlus=.1,expectedMinus=.1;
  for(let tick=1;tick<=40;tick++){
    S.net.t+=S.net.dt;const fired=new Uint8Array(3);
    if(tick===1||tick===17){fired[0]=1;pulses.push(S.net.t);}
    S.td.onStep(S.net.t,fired);
    if(tick%10===0){
      const e=pulses.reduce((sum,t)=>sum+Math.exp(-(S.net.t-t)/200),0),delta=S.net.t<2.5?2:-1,dw=.01*e*delta;
      expectedPlus=Math.min(1,Math.max(0,expectedPlus+dw));expectedMinus=Math.min(1,Math.max(0,expectedMinus-dw));
      updates.push({t:S.net.t,e,delta,dw});
    }
  }
  const maxError=Math.max(Math.abs(S.positive.scale[0]-expectedPlus),Math.abs(S.negative.scale[0]-expectedMinus));
  metrics.handCalculation={maxError,actualPlus:S.positive.scale[0],actualMinus:S.negative.scale[0],expectedPlus,expectedMinus,updates};
  c.check('确定脉冲与第三因子：每 1 ms 的 Δw 与解析资格迹手算相差 ≤ 1e-9',maxError<=1e-9,JSON.stringify(metrics.handCalculation));
  const U=stub({eta:.001,up:[1],down:[2],baselineUp:2,baselineDown:1});
  for(let tick=1;tick<=100;tick++){U.net.t+=U.net.dt;const fired=new Uint8Array(3);if(tick===1){fired[0]=1;fired[1]=1;}U.td.onStep(U.net.t,fired);}
  const expectedDelta=99*(1-Math.exp(-1)),expectedWeight=.1+.001*Math.exp(-(U.net.t-.1)/200)*expectedDelta;
  metrics.thirdFactor={delta:U.td.delta,expectedDelta,weight:U.positive.scale[0],expectedWeight,maxError:Math.max(Math.abs(U.td.delta-expectedDelta),Math.abs(U.positive.scale[0]-expectedWeight))};
  c.check('U 的组平均频率减基线并按 10 ms 平滑，给出正确第三因子',metrics.thirdFactor.maxError<=1e-9,JSON.stringify(metrics.thirdFactor));
  const B=stub({eta:2,factor:t=>t<1.5?100:-100});
  for(let tick=1;tick<=10;tick++){B.net.t+=B.net.dt;B.td.onStep(B.net.t,Uint8Array.of(tick===1?1:0,0,0));}
  const positiveBound=B.positive.scale[0]===1&&B.negative.scale[0]===0;
  for(let tick=11;tick<=20;tick++){B.net.t+=B.net.dt;B.td.onStep(B.net.t,new Uint8Array(3));}
  metrics.bounds={positiveBound,negativeBound:B.positive.scale[0]===0&&B.negative.scale[0]===1,pairSum:S.positive.scale[0]+S.negative.scale[0]};
  c.check('δ 为正时正权重增加、负权重等量减少，正负边界均生效',positiveBound&&metrics.bounds.negativeBound&&Math.abs(metrics.bounds.pairSum-.2)<1e-12,JSON.stringify(metrics.bounds));
  const bare=network(),off=network(false);advance(bare,500);advance(off,500);
  metrics.disabled={bare:digest(bare.net.exportState()),withModule:digest(off.net.exportState())};
  c.check('关闭学习的模块继续维护资格迹，但网络放电、全部状态与 ATP 逐位不变',metrics.disabled.bare===metrics.disabled.withModule,JSON.stringify(metrics.disabled));
  const continuous=network(true),first=network(true);advance(continuous,2000);advance(first,1000);
  const saved={net:first.net.exportState(),hooks:first.hooks.exportState()},resumed=network(true);
  resumed.net.importState(roundTrip(saved.net));resumed.hooks.importState(roundTrip(saved.hooks));advance(resumed,1000);
  metrics.resume={continuous:digest({net:continuous.net.exportState(),hooks:continuous.hooks.exportState()}),resumed:digest({net:resumed.net.exportState(),hooks:resumed.hooks.exportState()}),bytes:Buffer.byteLength(JSON.stringify(saved)),changedWeights:[...continuous.td.positive[0].scale].some(w=>w!==.6)};
  c.check('网络、资格迹、平滑状态与成对权重：JSON 导出/导入后精确续算',metrics.resume.continuous===metrics.resume.resumed&&metrics.resume.changedWeights,JSON.stringify(metrics.resume));
  const hooks=new StepHooks(),order=[],makeModule=name=>({onStepWritesResources:false,count:0,onStep(){order.push(name);this.count++;},exportState(){return {count:this.count};},importState(state){this.count=state.count;}});
  const prototype=hooks.add('prototype',makeModule('prototype')),delayed=hooks.add('delayed',makeModule('delayed'));
  hooks.onStep(.1,new Uint8Array(3));hooks.enable('delayed',false);const snapshot=roundTrip(hooks.exportState());hooks.onStep(.2,new Uint8Array(3));hooks.importState(snapshot);
  metrics.dispatch={order,prototypeCount:prototype.count,delayedCount:delayed.count,enabled:hooks.modules.get('delayed').enabled,sparse:hooks.runOptions().onStepWritesResources===false};
  c.check('按注册顺序分发，并保存扩展规则的状态和启用开关',order.join(',')==='prototype,delayed,prototype'&&prototype.count===1&&delayed.count===1&&!metrics.dispatch.enabled&&metrics.dispatch.sparse,JSON.stringify(metrics.dispatch));
  const ok=c.done(),result={schema:1,package:'w3-td',createdAtUtc:new Date().toISOString(),ok,metrics};
  if(write){const dir=path.join(ROOT,'experiments/logs/2026-10-10-loop-w3-td');fs.mkdirSync(dir,{recursive:true});fs.writeFileSync(path.join(dir,'three-factor-results.json'),JSON.stringify(result,null,2)+'\n');}
  return {ok,result};
}
if(process.argv[1]&&import.meta.url===pathToFileURL(path.resolve(process.argv[1])).href)process.exit(verifyL4ThreeFactor().ok?0:1);
