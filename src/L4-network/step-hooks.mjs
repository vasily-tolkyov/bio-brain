/** 一个 onStep 回调中按注册顺序分发多个独立学习模块。 */
export class StepHooks {
  constructor(){this.modules=new Map();this.onStep=this.onStep.bind(this);}
  add(name,module,{enabled=true}={}){
    if(this.modules.has(name)||typeof module.onStep!=='function')throw new Error('步进模块名称重复或没有 onStep');
    this.modules.set(name,{module,enabled});return module;
  }
  remove(name){return this.modules.delete(name);}
  enable(name,enabled=true){const entry=this.modules.get(name);if(!entry)throw new Error('未知步进模块');entry.enabled=enabled;}
  onStep(t,fired){for(const {module,enabled} of this.modules.values())if(enabled)module.onStep(t,fired);}
  runOptions(){return {onStep:this.onStep,onStepWritesResources:[...this.modules.values()].some(({module,enabled})=>enabled&&module.onStepWritesResources!==false)};}
  exportState(){return {schema:1,modules:[...this.modules].map(([name,{module,enabled}])=>({name,enabled,state:module.exportState?.()??null}))};}
  importState(saved){
    if(saved.schema!==1||saved.modules.length!==this.modules.size||saved.modules.some(row=>!this.modules.has(row.name)))throw new Error('步进模块集合不匹配');
    for(const row of saved.modules){const entry=this.modules.get(row.name);entry.enabled=row.enabled;if(row.state!==null){if(!entry.module.importState)throw new Error('步进模块不支持续算');entry.module.importState(row.state);}}
    return this;
  }
}
/** 外部原型或延迟写入规则通过 add 注册，其状态由同一分发器保存。 */
export const createStepHooks=()=>new StepHooks();
