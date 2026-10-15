/** 测试世界与 C 放电解码的独立验证；FAST 包含本组。 */
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {checker,fmt} from '../src/core/check.mjs';
import {FEATURES,NUT_PROTOTYPES,stormWorld,nutWorld,stoneWorld,fruitWorld,activeEvents,scriptedRate,sceneSpec} from '../src/loop/world.mjs';
import {CONTEXT_DELAYS,synthesizeContext,decodeContext,integrateU,detectBoundaries,median} from '../src/loop/decode.mjs';

export function verifyLoopWorld({write=true}={}) {
  const check=checker('闭环 测试世界与解码'),seed=20261010,metrics={seed},scripts={};
  scripts.A=stormWorld({seed});scripts.B=stormWorld({seed,set:'B'});
  const birdFraction=script=>script.trials.filter(row=>row.hasBird).length/script.trials.length;
  const birds={A:birdFraction(scripts.A),B:birdFraction(scripts.B)};
  metrics.birdFractions=birds;
  check.check('两套 40 场雷雨的鸟叫比例均在 50% ± 10%',Object.values(birds).every(x=>Math.abs(x-.5)<=.1),JSON.stringify(birds));
  const validStorm=script=>script.trials.every(row=>{
    const flash=row.events.find(e=>e.g==='闪电'),thunder=row.events.find(e=>e.g==='雷声'),bird=row.events.find(e=>e.g==='鸟叫');
    const last=Math.max(...row.events.map(e=>e.start+e.duration));
    return flash.duration===100&&thunder.duration===2000&&thunder.start-flash.start===row.delay&&(!bird||(bird.duration===200&&bird.start-flash.start>=0&&bird.start-flash.start<10000))&&row.end-last>=20000&&row.end-last<=25000;
  });
  check.check('雷雨事件时长、固定 / 均匀延迟与 20–25 s 静息符合脚本',validStorm(scripts.A)&&validStorm(scripts.B)&&scripts.A.trials.every(row=>row.delay===3000)&&scripts.B.trials.every(row=>row.delay>=1000&&row.delay<=10000));
  scripts.change=stormWorld({seed,set:'change'});
  scripts.probes=Object.fromEntries(['lightning','bird','early'].map(probe=>[probe,stormWorld({seed,probe})]));
  check.check('延迟变更最多 30 场，每两场一次冻结探测；三种探测各 10 次',scripts.change.trials.filter(t=>!t.frozen).length===30&&scripts.change.trials.filter(t=>t.frozen).length===15&&scripts.change.trials.filter(t=>!t.frozen).every(t=>t.delay===6000)&&Object.values(scripts.probes).every(s=>s.trials.length===10&&s.trials.every(t=>t.frozen))&&scripts.probes.early.trials.every(t=>t.events.find(e=>e.g==='雷声').duration===1000));

  scripts.nuts=nutWorld({seed});
  const nuts=scripts.nuts,combination=xs=>xs.slice().sort().join('|'),reserved=new Set(Object.values(nuts.reserved).flat().map(combination));
  const validCombination=row=>{const p=NUT_PROTOTYPES[row.kind];return row.features.length===5&&new Set(row.features).size===5&&row.features.filter(g=>p.includes(g)).length===4;};
  metrics.nuts={training:nuts.training.trials.length,reserved:Object.fromEntries(Object.entries(nuts.reserved).map(([k,v])=>[k,v.length])),heldOutLeakage:nuts.training.trials.filter(t=>reserved.has(combination(t.features))).length};
  check.check('坚果训练 4×60 次，每次原型内 4 个加原型外 1 个；预留组合从未训练',nuts.training.trials.length===240&&nuts.training.trials.every(validCombination)&&metrics.nuts.heldOutLeakage===0&&Object.values(nuts.reserved).every(rows=>rows.length===10),JSON.stringify(metrics.nuts));
  check.check('坚果测试 4×10 个预留组合、补全测试 4×10 个三特征线索，全部冻结',nuts.test.trials.length===40&&nuts.test.trials.every(t=>validCombination(t)&&reserved.has(combination(t.features))&&t.frozen)&&nuts.completion.trials.length===40&&nuts.completion.trials.every(t=>t.features.length===3&&t.features.every(g=>NUT_PROTOTYPES[t.kind].includes(g))&&t.frozen));

  scripts.stone=stoneWorld({seed});scripts.stoneTest=stoneWorld({seed,test:true});
  const stoneCounts=Object.fromEntries([...new Set(scripts.stone.trials.map(t=>t.kind))].map(k=>[k,scripts.stone.trials.filter(t=>t.kind===k).length]));
  metrics.stoneCounts=stoneCounts;
  const validAction=t=>{
    const a=t.events.find(e=>e.action),o=t.events.find(e=>['裂开','烂了','甜','苦'].includes(e.g));
    return a.start-t.start===1000&&a.duration===100&&(!o||(o.start-a.start-a.duration===100&&o.duration===1000))&&t.end-Math.max(...t.events.map(e=>e.start+e.duration))===4000;
  };
  check.check('石头五类经历各 35 次；动作、后果与静息时间正确，迁移测试 2×10 次',scripts.stone.trials.length===175&&Object.values(stoneCounts).every(n=>n===35)&&scripts.stone.trials.every(validAction)&&scripts.stoneTest.trials.length===20&&scripts.stoneTest.trials.every(t=>t.frozen&&validAction(t)));

  scripts.fruit0=fruitWorld({seed});scripts.fruit1=fruitWorld({seed,phase:1});scripts.fruitControl=fruitWorld({seed,phase:1,control:true});
  const shares=script=>Object.fromEntries(['锯齿叶','光滑叶','裂叶'].map(leaf=>[leaf,script.trials.filter(t=>t.leaf===leaf).length/script.trials.length]));
  metrics.leafFractions={phase0:shares(scripts.fruit0),phase1:shares(scripts.fruit1)};
  check.check('果子阶段 0 / 1 分别 60 / 150 次，三种叶子各占 1/3 ± 0.08',scripts.fruit0.trials.length===60&&scripts.fruit1.trials.length===150&&Object.values(metrics.leafFractions).every(shares=>Object.values(shares).every(x=>Math.abs(x-1/3)<=.08)),JSON.stringify(metrics.leafFractions));
  check.check('只看没有动作或味道；吃后光滑叶苦、另两种甜；独立随机苦味对照',scripts.fruit0.trials.every(t=>t.action===null&&t.outcome===null)&&scripts.fruit1.trials.every(t=>t.outcome===(t.leaf==='光滑叶'?'苦':'甜')&&validAction(t))&&scripts.fruitControl.trials.some(t=>t.leaf!=='光滑叶'&&t.outcome==='苦')&&scripts.fruitControl.trials.some(t=>t.leaf==='光滑叶'&&t.outcome==='甜'));

  let next=0;const channels=['闪电','雷声','鸟叫'].flatMap(g=>['+','−'].map(c=>({g,c,slots:CONTEXT_DELAYS.map(()=>Array.from({length:6},()=>next++))})));
  const errors=[];let correct=0,total=0;
  for(const channel of channels)for(let k=0;k<40;k++){
    const age=100*(240**(k/39)),now=30000+age;
    const spikes=synthesizeContext(channels,[{g:channel.g,c:channel.c,start:30000}],{now});
    const decoded=decodeContext({spikes,channels,now});
    const match=decoded.find(row=>row.g===channel.g&&row.c===channel.c);
    total++;if(match&&decoded.length===1){correct++;errors.push(Math.abs(match.age-age)/age);}else errors.push(Infinity);
  }
  metrics.decode={samples:total,medianRelativeError:median(errors),identityAccuracy:correct/total,maxRelativeError:Math.max(...errors)};
  check.check('窗口合成放电：时间中位相对误差 ≤ 0.1，身份 / 方向正确率 ≥ 99%',metrics.decode.medianRelativeError<=.1&&metrics.decode.identityAccuracy>=.99,JSON.stringify(metrics.decode));
  const repeats=[{g:'闪电',c:'+',start:1000},{g:'闪电',c:'+',start:6000},{g:'鸟叫',c:'−',start:7500}],now=16000;
  const decoded=decodeContext({spikes:synthesizeContext(channels,repeats,{now}),channels,now});
  metrics.repeated=decoded;
  check.check('同通道重复装入与不同通道并存可以分别解码，25 s 外不保留候选',decoded.length===3&&repeats.every(e=>decoded.some(row=>row.g===e.g&&row.c===e.c&&Math.abs(row.onset-e.start)<10))&&decodeContext({spikes:synthesizeContext(channels,[{g:'闪电',c:'+',start:1000}],{now:40000}),channels,now:40000}).length===0);
  const stronger=decodeContext({spikes:synthesizeContext(channels,[{g:'闪电',c:'+',start:1000,strength:2}],{now:3000}),channels,now:3000})[0];
  check.check('两倍装入不改变时间解码，误差小于 15%',Math.abs(stronger.age-2000)/2000<=.15,fmt(stronger.age));
  const input=[[5,15,25],[5,15,25],[45],[45]];
  const u=integrateU({spikes:input,up:[0,1],down:[2,3],start:0,end:50,baselineUp:0,baselineDown:0});
  const boundaries=detectBoundaries({spikes:[[5,15,45],[5,15,45]],ids:[0,1],start:0,end:60});
  metrics.tools={u,boundaries};
  check.check('U 积分按组平均计数；连续阈上分箱形成两个变化边界',Math.abs(u-4)<1e-12&&boundaries.bursts.length===2&&boundaries.bursts[0].start===0&&boundaries.bursts[0].end===20&&boundaries.bursts[1].start===40);
  const rate=scriptedRate(scripts.A,{闪电:[0,1],雷声:[2],鸟叫:[3]},{background:3});
  check.check('脚本只在事件期间以 58 Hz 驱动，背景及通道结构正确',rate(0,0)===58&&rate(0,100)===3&&activeEvents(scripts.A,3000).features.has('雷声')&&sceneSpec(2).context.length===0&&sceneSpec(3).context.length===2&&sceneSpec(4).outputs.length===2&&FEATURES[3].length===14);
  const ok=check.done();
  const result={schema:1,package:'w5-world',createdAtUtc:new Date().toISOString(),ok,metrics,scripts};
  if(write){const out=fileURLToPath(new URL('../experiments/logs/2026-10-10-loop-w5-world/world-results.json',import.meta.url));fs.mkdirSync(path.dirname(out),{recursive:true});fs.writeFileSync(out,JSON.stringify(result,null,2)+'\n');}
  return {ok,result};
}
if(process.argv[1]&&import.meta.url===pathToFileURL(path.resolve(process.argv[1])).href)process.exit(verifyLoopWorld().ok?0:1);
