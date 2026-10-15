/** 只读神经元放电的分析工具，时刻以 ms 计；不接受世界事件或真实事件起点。 */
export const CONTEXT_DELAYS = Object.freeze(Array.from({length:8},(_,k)=>100*2**k));
export const median = xs => {const a=xs.slice().sort((x,y)=>x-y);return a.length?(a[Math.floor((a.length-1)/2)]+a[Math.ceil((a.length-1)/2)])/2:NaN;};

export function binRates(spikes,ids,start,end,binMs=10) {
  if(!(binMs>0)||!(end>=start)||!ids.length)throw new Error('无效分箱');
  const out=new Float64Array(Math.ceil((end-start)/binMs));
  for(const id of ids)for(const t of spikes[id]??[])if(t>=start&&t<end){
    const k=Math.floor((t-start)/binMs),duration=Math.min(binMs,end-(start+k*binMs));
    out[k]+=1000/(ids.length*duration);
  }
  return [...out];
}
/** 合成窗口只供解码检查。细胞放电来自 [0.6,1.55]d 窗口，不传给解码器事件标签。 */
export function synthesizeContext(channels,events,{hz=30,now=Infinity,neurons=null}={}) {
  const N=neurons??1+Math.max(...channels.flatMap(ch=>ch.slots.flat()));
  const spikes=Array.from({length:N},()=>[]);
  for(const e of events){
    const ch=channels.find(ch=>ch.g===e.g&&ch.c===e.c);
    if(!ch)throw new Error('未知合成通道');
    for(let k=0;k<ch.slots.length;k++){
      const d=CONTEXT_DELAYS[k],ids=ch.slots[k],period=1000/(hz*(e.strength??1));
      for(let j=0;j<ids.length;j++)for(let t=e.start+.6*d+period*(j+.5)/ids.length;t<=Math.min(now,e.start+1.55*d);t+=period)spikes[ids[j]].push(t);
    }
  }
  for(const row of spikes)row.sort((a,b)=>a-b);
  return spikes;
}
/** 从最短档的群体窗口前沿估计装入时刻，并以其它档的窗口核对。
 * 使用已记录的 C 放电历史，可分开同通道的多轮装入；超过 25 s 的候选自然过期。
 * 神经元编号给出身份，估计的装入时刻完全来自放电，不读取世界脚本。 */
export function decodeContext({spikes,channels,now,windowMs=25000,minCells=3}) {
  const decoded=[];
  for(const ch of channels){
    const first=ch.slots[0],d=CONTEXT_DELAYS[0],hits=[];
    for(const id of first)for(const t of spikes[id]??[])if(t<=now&&t>=now-windowMs-.6*d)hits.push({t,id});
    hits.sort((a,b)=>a.t-b.t);
    const clusters=[];let cluster=[];
    for(const hit of hits){if(cluster.length&&hit.t-cluster.at(-1).t>.4*d){clusters.push(cluster);cluster=[];}cluster.push(hit);}
    if(cluster.length)clusters.push(cluster);
    for(const row of clusters){
      if(new Set(row.map(h=>h.id)).size<Math.min(minCells,first.length)||row.length<minCells)continue;
      const onset=row[0].t-.6*d,age=now-onset;
      if(age<0||age>windowMs)continue;
      let supportedSlots=0;
      for(let k=0;k<ch.slots.length;k++){
        const dk=CONTEXT_DELAYS[k],start=onset+.6*dk-10,end=Math.min(now,onset+1.55*dk+10);
        if(start<=end&&ch.slots[k].some(id=>(spikes[id]??[]).some(t=>t>=start&&t<=end)))supportedSlots++;
      }
      decoded.push({g:ch.g,c:ch.c,age,onset,supportedSlots,spikes:row.length,cells:new Set(row.map(h=>h.id)).size});
    }
  }
  return decoded.sort((a,b)=>a.onset-b.onset);
}
export function integrateU({spikes,up,down,start,end,baselineUp=0,baselineDown=0,binMs=10}) {
  const a=binRates(spikes,up,start,end,binMs),b=binRates(spikes,down,start,end,binMs);
  return a.reduce((sum,hz,k)=>sum+(Math.max(0,hz-baselineUp)+Math.max(0,b[k]-baselineDown))*Math.min(binMs,end-start-k*binMs)/1000,0);
}
export function detectBoundaries({spikes,ids,start,end,mu=0,sigma=0,binMs=10}) {
  const theta=Math.max(mu+4*sigma,mu+5),rates=binRates(spikes,ids,start,end,binMs),bursts=[];
  let active=null;
  rates.forEach((hz,k)=>{
    const t=start+k*binMs;
    if(hz>theta){
      active??={start:t,end:t,peak:0,integral:0};
      active.end=Math.min(end,t+binMs);active.peak=Math.max(active.peak,hz);
      active.integral+=(hz-mu)*(active.end-t)/1000;
    }else if(active){bursts.push(active);active=null;}
  });
  if(active)bursts.push(active);
  return {theta,rates,bursts};
}
