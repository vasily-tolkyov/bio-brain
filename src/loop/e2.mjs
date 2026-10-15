/** 范畴池与共用原型：四条通路和余弦侧抑制一起更新。 */
import {pairs,moduleInterface} from './readout.mjs';
export const cosine=(a,b)=>{
  let dot=0,aa=0,bb=0;for(let i=0;i<a.length;i++){dot+=a[i]*b[i];aa+=a[i]*a[i];bb+=b[i]*b[i];}
  return aa>0&&bb>0?dot/Math.sqrt(aa*bb):0;
};
export function e2Populations(units){
  return [...Array.from({length:units},(_,k)=>({name:'E2:'+k,n:20,p:{b:20,tauW:300}})),
    {name:'I2',n:units*20/4,p:{C:100,tRef:1}},...Array.from({length:units},(_,k)=>({name:'J:'+k,n:5,p:{C:100,tRef:2,b:5,tauW:200}}))];
}
export class PrototypeLearning{
  constructor({net,features,featureIds,units,W,paths,lateral,eta=.01,referenceHz=25,maxHz=40}){
    Object.assign(this,{net,features,featureIds,units,W,paths,lateral,eta,referenceHz,maxHz});
    this.referenceHz=Float64Array.from(typeof referenceHz==='number'?new Array(features.length).fill(referenceHz):referenceHz);
    this.maxHz=Float64Array.from(typeof maxHz==='number'?new Array(units.length).fill(maxHz):maxHz);
    if(this.referenceHz.length!==features.length||this.maxHz.length!==units.length||![...this.referenceHz,...this.maxHz].every(x=>Number.isFinite(x)&&x>0))throw new Error('原型归一化频率无效');
    this.onStepWritesResources=false;this.enabled=false;this.gain=new Float64Array(units.length).fill(1);
    this.featureCounts=new Float64Array(features.length);this.unitCounts=new Float64Array(units.length);
    this.featureTheta=new Float64Array(features.length).fill(5);this.unitTheta=new Float64Array(units.length).fill(5);
    this.onStep=this.onStep.bind(this);this.sync();
  }
  sync(){
    for(const {k,g,kind,group,strength} of this.paths)group.scale.fill(strength*this.W[k][g]);
    for(const {k,l,group,strength} of this.lateral)group.scale.fill(k===l?0:strength*cosine(this.W[k],this.W[l]));
  }
  update(featureRates,unitRates){
    if(!this.enabled||!featureRates.some((rate,g)=>rate>this.featureTheta[g]))return;
    for(let k=0;k<this.units.length;k++){
      if(!(unitRates[k]>this.unitTheta[k]))continue;
      const a=Math.min(1,Math.max(0,unitRates[k]/this.maxHz[k])),learning=this.eta*this.gain[k]*a;
      for(let g=0;g<this.features.length;g++){const x=Math.min(1,Math.max(0,featureRates[g]/this.referenceHz[g]));this.W[k][g]=Math.min(1,Math.max(0,this.W[k][g]+learning*(x-this.W[k][g])));}
    }
    this.sync();
  }
  onStep(t,fired){
    for(let g=0;g<this.features.length;g++)for(const i of this.featureIds[this.features[g]])this.featureCounts[g]+=fired[i]?1:0;
    for(let k=0;k<this.units.length;k++)for(const i of this.units[k])this.unitCounts[k]+=fired[i]?1:0;
    if(Math.round(t/this.net.dt)%Math.round(10/this.net.dt)!==0)return;
    this.update([...this.featureCounts].map((n,g)=>n*1000/(this.featureIds[this.features[g]].length*10)),[...this.unitCounts].map((n,k)=>n*1000/(this.units[k].length*10)));
    this.featureCounts.fill(0);this.unitCounts.fill(0);
  }
  exportState(){return {schema:1,W:this.W.map(row=>[...row]),gain:[...this.gain],featureCounts:[...this.featureCounts],unitCounts:[...this.unitCounts],featureTheta:[...this.featureTheta],unitTheta:[...this.unitTheta],eta:this.eta,referenceHz:[...this.referenceHz],maxHz:[...this.maxHz],enabled:this.enabled};}
  importState(saved){
    if(saved.schema!==1||saved.W.length!==this.W.length||saved.W.some(row=>row.length!==this.features.length))throw new Error('原型状态不匹配');
    saved.W.forEach((row,k)=>this.W[k].set(row));
    for(const name of ['gain','featureCounts','unitCounts','featureTheta','unitTheta','referenceHz','maxHz'])this[name].set(saved[name]);
    for(const name of ['eta','enabled'])this[name]=saved[name];this.sync();return this;
  }
}
export function buildE2(net,{features,units:count,e1,errors,presets={},byIndegree,recurrentStrength=4.5,ablatePrediction=false}){
  const units=Array.from({length:count},(_,k)=>net.idx('E2:'+k)),J=Array.from({length:count},(_,k)=>net.idx('J:'+k)),I=net.idx('I2'),W=Array.from({length:count},()=>new Float64Array(features.length)),labels=new Array(count).fill(null);
  Object.entries(presets).forEach(([label,parts],k)=>{if(k>=count)throw new Error('预提交数超过池容量');labels[k]=label;for(const g of parts){const index=features.indexOf(g);if(index<0)throw new Error('原型包含未知特征');W[k][index]=1;}});
  const connect=(pre,post,p,rest={})=>net.connect({pairs:pairs(net,pre,post,p),contacts:10,pRelease:.3,tauRec:100,...rest});
  const E=units.flat(),paths=[],lateral=[],handles={recurrent:[],inhibition:[],balance:[],paths,lateral};
  for(let k=0;k<count;k++)handles.recurrent.push(byIndegree({pre:units[k],post:units[k],route:'recurrent',strength:labels[k]===null?0:recurrentStrength}));
  handles.balance.push(connect(E,I,.2,{scale:1.5}));
  // 按单元分组，才能在空闲期间固定抑制；提交后再打开 Vogels 调节。
  for(let k=0;k<count;k++){
    const group=connect(I,units[k],.4,{kind:'inhFast',pRelease:.5,tauRec:300,scale:4,istdp:{eta:.02,target:1,wMax:40}});
    group.e2Unit=k;group.e2Idle=labels[k]===null;group.frozen=group.e2Idle;handles.inhibition.push(group);handles.balance.push(group);
  }
  handles.balance.push(connect(I,I,.4,{kind:'inhFast',pRelease:.5,tauRec:300,scale:4}));
  handles.balance.push(byIndegree({pre:e1.E,post:I,route:'feedforwardInhibition',strength:1.5}));
  for(let k=0;k<count;k++){
    connect(units[k],J[k],.3,{scale:3});
    for(let l=0;l<count;l++)if(k!==l)lateral.push({k,l,strength:3,group:connect(J[k],units[l],.3,{kind:'inh',scale:0})});
    for(let g=0;g<features.length;g++){
      const feature=features[g];
      for(const [kind,pre,post,strength,recep] of [
        ['recognition',e1.featureIds[feature],units[k],6,'exc'],
        ['up',errors.N[feature],units[k],3,'exc'],
        ['prediction',units[k],errors.S[feature],2,'nmda'],
        ['negative',units[k],errors.M[feature],2,'exc'],
      ]){
        const effective=ablatePrediction&&['prediction','negative'].includes(kind)?0:strength;
        const group=byIndegree({pre,post,route:kind,strength:effective,kind:recep});
        paths.push({k,g,kind,strength:effective*group.loopRoute.multiplier,group});
      }
    }
  }
  const learner=new PrototypeLearning({net,features,featureIds:e1.featureIds,units,W,paths,lateral});
  const ids=Object.fromEntries([...units.map((v,k)=>['E2:'+k,v]),['I2',I],...J.map((v,k)=>['J:'+k,v])]);
  const module=moduleInterface(net,{ids,handles,units,I,J,W,labels,learner,
    saveState:()=>({learner:learner.exportState(),labels:labels.slice()}),
    loadState:saved=>{learner.importState(saved.learner);saved.labels.forEach((label,k)=>labels[k]=label);syncCommitment();},
    selfTest:()=>I.length===count*20/4&&units.every(ids=>ids.length===20)&&J.every(ids=>ids.length===5)&&paths.length===4*count*features.length&&lateral.every(row=>row.k!==row.l)&&handles.inhibition.every((g,k)=>g.istdp.target===1&&g.e2Idle===(labels[k]===null)&&(labels[k]!==null||g.frozen&&handles.recurrent[k].scale.every(x=>x===0))),
  });
  function syncCommitment(){
    for(let k=0;k<count;k++){
      const idle=labels[k]===null,group=handles.recurrent[k];group.loopRoute.strength=idle?0:recurrentStrength;group.scale.fill(group.loopRoute.strength*group.loopRoute.multiplier);
      handles.inhibition[k].e2Idle=idle;if(idle)handles.inhibition[k].frozen=true;
    }
  }
  module.commit=(k,label,prototype)=>{
    if(!Number.isInteger(k)||k<0||k>=count||typeof label!=='string'||!label||prototype.length!==features.length||!Array.from(prototype).every(x=>Number.isFinite(x)&&x>=0&&x<=1))throw new Error('提交原型无效');
    W[k].set(prototype);labels[k]=label;syncCommitment();handles.inhibition[k].frozen=false;learner.sync();
  };
  module.recycle=k=>{if(!Number.isInteger(k)||k<0||k>=count)throw new Error('回收单元无效');W[k].fill(0);labels[k]=null;handles.inhibition[k].scale.fill(4);syncCommitment();learner.sync();};
  module.match=x=>W.map(row=>cosine(x,row));return module;
}
