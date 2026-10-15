/** 架构 8.4：突触前资格迹 × 平滑的 U 差，成对修改 C→P / C→Q 的 scale。 */
const pack=a=>({length:a.length,data:Buffer.from(a.buffer,a.byteOffset,a.byteLength).toString('base64')});
function load(a,state){const b=Buffer.from(state.data,'base64');if(a.length!==state.length||b.length!==a.byteLength)throw new Error('三因素状态长度不匹配');Buffer.from(a.buffer,a.byteOffset,a.byteLength).set(b);}
export class ThreeFactor {
  constructor({net,positive,negative,up=[],down=[],baselineUp=0,baselineDown=0,eta=1e-5,wMax=6,tauE=200,smoothMs=10,updateMs=1,enabled=true,factor=null,initialize=true}){
    if(!(eta>=0&&wMax>0&&tauE>0&&smoothMs>0&&updateMs>0))throw new Error('无效三因素参数');
    this.onStepWritesResources=false;
    this.net=net;this.positive=Array.isArray(positive)?positive:[positive];this.negative=Array.isArray(negative)?negative:[negative];
    if([...this.positive,...this.negative].some(g=>!g||g.pre.length!==g.scale.length))throw new Error('三因素连接组无效');
    Object.assign(this,{up:[...up],down:[...down],baselineUp,baselineDown,eta,wMax,tauE,smoothMs,updateMs,enabled,factor,gain:1});
    this.e=new Float64Array(net.N);this.pre=Int32Array.from([...new Set([...this.positive,...this.negative].flatMap(g=>[...g.pre]))].sort((a,b)=>a-b));
    this.binUp=0;this.binDown=0;this.smoothUp=0;this.smoothDown=0;this.delta=0;this.lastTime=net.t;
    this.onStep=this.onStep.bind(this);
    if(initialize)for(const g of [...this.positive,...this.negative])g.scale.fill(.1*wMax);
  }
  onStep(t,fired){
    const elapsed=t-this.lastTime;this.lastTime=t;
    const decay=Math.exp(-elapsed/this.tauE);
    for(const i of this.pre){this.e[i]*=decay;if(fired[i])this.e[i]+=1;}
    for(const i of this.up)this.binUp+=fired[i]?1:0;
    for(const i of this.down)this.binDown+=fired[i]?1:0;
    const tick=Math.round(t/this.net.dt),binTicks=Math.max(1,Math.round(10/this.net.dt)),updateTicks=Math.max(1,Math.round(this.updateMs/this.net.dt));
    if(tick%binTicks===0){
      const weight=1-Math.exp(-10/this.smoothMs);
      const up=this.up.length?this.binUp*1000/(this.up.length*10)-this.baselineUp:0;
      const down=this.down.length?this.binDown*1000/(this.down.length*10)-this.baselineDown:0;
      this.smoothUp+=weight*(up-this.smoothUp);this.smoothDown+=weight*(down-this.smoothDown);
      this.delta=this.smoothUp-this.smoothDown;this.binUp=0;this.binDown=0;
    }
    if(tick%updateTicks!==0||!this.enabled)return;
    const delta=this.factor?this.factor(t,this):this.delta;
    const rate=this.eta*this.gain;
    for(const [groups,sign] of [[this.positive,1],[this.negative,-1]])for(const g of groups)for(let k=0;k<g.pre.length;k++)g.scale[k]=Math.min(this.wMax,Math.max(0,g.scale[k]+sign*rate*this.e[g.pre[k]]*delta));
  }
  exportState(){
    return {schema:1,parameters:{baselineUp:this.baselineUp,baselineDown:this.baselineDown,eta:this.eta,wMax:this.wMax,tauE:this.tauE,smoothMs:this.smoothMs,updateMs:this.updateMs,gain:this.gain,enabled:this.enabled},
      e:pack(this.e),binUp:this.binUp,binDown:this.binDown,smoothUp:this.smoothUp,smoothDown:this.smoothDown,delta:this.delta,lastTime:this.lastTime,
      positive:this.positive.map(g=>pack(g.scale)),negative:this.negative.map(g=>pack(g.scale))};
  }
  importState(saved){
    if(saved.schema!==1||saved.positive.length!==this.positive.length||saved.negative.length!==this.negative.length)throw new Error('三因素状态不匹配');
    load(this.e,saved.e);Object.assign(this,saved.parameters);
    for(const name of ['binUp','binDown','smoothUp','smoothDown','delta','lastTime'])this[name]=saved[name];
    saved.positive.forEach((row,i)=>load(this.positive[i].scale,row));saved.negative.forEach((row,i)=>load(this.negative[i].scale,row));
    return this;
  }
}
export const createThreeFactor=options=>new ThreeFactor(options);
