/** 时间期待与第二级误差的神经回路；没有 E→P/Q 或 P→N/M。 */
import {ThreeFactor} from '../L4-network/three-factor.mjs';
import {channelKey,pairs,moduleInterface} from './readout.mjs';
export const GAMMA=Math.exp(-30/3000);
export function expectPopulations(outputs){
  return outputs.flatMap(ch=>{
    const key=channelKey(ch);
    return [['P',20,false],['Q',10,true],['Z',10,true],['D',20,false],['Up',10,false],['Down',10,false],['IUp',10,true],['IDown',10,true]]
      .map(([type,n,inh])=>({name:type+':'+key,n,p:inh?{C:100,tRef:1,b:0}:{b:5,tauW:200}}));
  });
}
export function buildExpect(net,{outputs,context,eta=1e-5,wMax=6,cancelScale=3}){
  const P={},Q={},Z={},D={},Up={},Down={},IUp={},IDown={},learners={},handles={paired:[],delay:[],reset:[],error:[],strengthen:[]},allC=Object.values(context.C).flat(2);
  const connect=(pre,post,p,rest={})=>net.connect({pairs:pairs(net,pre,post,p),contacts:10,pRelease:.3,tauRec:100,...rest});
  for(const ch of outputs){
    const key=channelKey(ch);for(const [name,table] of [['P',P],['Q',Q],['Z',Z],['D',D],['Up',Up],['Down',Down],['IUp',IUp],['IDown',IDown]])table[key]=net.idx(name+':'+key);
    const plus=connect(allC,P[key],.2),minus=connect(allC,Q[key],.2);
    handles.paired.push({key,plus,minus});
    handles.reset.push(connect(Q[key],P[key],.5,{kind:'inhFast',scale:4}));
    handles.reset.push(connect(context.L[key],Z[key],.5,{scale:3}));
    handles.reset.push(connect(Z[key],P[key],.5,{kind:'inhFast',scale:8,tauRec:100}));
    handles.delay.push(connect(P[key],D[key],.5,{kind:'slow',tauR:10,tauD:30,scale:1}));
    // U↑ 的观察项由本通道 L 提供；D 只经抑制池到 U↑。
    handles.error.push(connect(context.L[key],Up[key],.5,{scale:3}));
    handles.error.push(connect(P[key],Up[key],.5,{scale:cancelScale*GAMMA}));
    handles.error.push(connect(D[key],IUp[key],.5,{scale:cancelScale}));
    handles.error.push(connect(IUp[key],Up[key],.5,{kind:'inhFast',scale:4}));
    // U↓ 接 D 的兴奋，P 与观察项均经另一抑制池抵消。
    handles.error.push(connect(D[key],Down[key],.5,{scale:cancelScale}));
    handles.error.push(connect(P[key],IDown[key],.5,{scale:cancelScale*GAMMA}));
    handles.error.push(connect(context.L[key],IDown[key],.5,{scale:3}));
    handles.error.push(connect(IDown[key],Down[key],.5,{kind:'inhFast',scale:4}));
    handles.strengthen.push(connect(Up[key],context.L[key],.25,{scale:1}));
    learners[key]=new ThreeFactor({net,positive:plus,negative:minus,up:Up[key],down:Down[key],eta,wMax,enabled:false});
  }
  const ids=Object.fromEntries([['P',P],['Q',Q],['Z',Z],['D',D],['Up',Up],['Down',Down],['IUp',IUp],['IDown',IDown]].flatMap(([name,table])=>Object.entries(table).map(([key,v])=>[name+':'+key,v])));
  return moduleInterface(net,{ids,handles,P,Q,Z,D,Up,Down,IUp,IDown,learners,
    saveState:()=>({learners:Object.fromEntries(Object.entries(learners).map(([key,module])=>[key,module.exportState()]))}),
    loadState:saved=>Object.entries(saved.learners).forEach(([key,state])=>learners[key].importState(state)),
    selfTest:()=>Object.values(P).every(v=>v.length===20)&&Object.values(D).every(v=>v.length===20)&&Object.values(Up).every(v=>v.length===10)&&Object.values(Down).every(v=>v.length===10)&&handles.delay.every(g=>g.slow.tauR===10&&g.slow.tauD===30),
  });
}
