// 并行先训练对照网络（第二层没经历物体序列），主实验跑到这里时直接读缓存
import { trainedState } from '../demo/lib-hierarchy.mjs';
trainedState(2, { variant: 'noseq' });
console.log('对照网络训练完成');
