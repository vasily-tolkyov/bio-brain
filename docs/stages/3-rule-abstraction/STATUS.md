# 第 3 段：规则抽象

状态：部分完成。快照日期：2026-10-06。集成分支：`stage/3-rule-abstraction`。

## 目标

从经验中抽象出规则：按后果把情境归类，关注能预测后果的特征，并在新情境中按规则预测。

## 完成判据

- 已达成：`verify/L4-uat.mjs`、`verify/L4-two-layer.mjs`、`verify/L4-hierarchy.mjs` 的验证。
- 草案，未达成：前馈可学，脉冲版复现规则提取的陷阱测试。

## 现状

### 速率版

- 一层吸引子加三因素读出满足万能近似：f_sin 误差在 64 个吸引子时为 0.241，512 个时为 0.111。
- 分开的模块经第二层组合后交互可读（0.489），前提是按强度分级传递。
- 无监督叠层不增加信息。
- 误差引导的抽象对齐类别：象限正确率 88%，无监督约 50%（c372e7c）。
- 规则提取（b56b1a7）：结果逐层改写下层解释，注意力移向声音；陷阱测试 98.5%，只有前馈时 73.0%，经验缺少变化时 56.1%。

### 脉冲版

- 两层网络长出物体级吸引子，能跨层补全（1172c39）。
- 多层网络（5de0c7f）：第一层不变性 0.55 → 0.92（2 层），4 层时最上层为 1.00；想象时第一层重建 X 为 16.3 Hz。
- 层数到 2–3 层就饱和；每多一层，250 ms 内多耗约 580×10⁹ ATP。

脉冲版的抽象目前只是共现绑定。前馈为固定的按组拓扑投射，不学习；没有结果通道和动作通道；中间层没有误差驱动的学习，见[技术说明](../../../README.zh-CN.md)“已知缺口”第 6 条。

## 尝试与里程碑

- 2026-10-01，`milestone/uat-two-layer`（1172c39）：三因素读出与两层物体吸引子。
- 2026-10-01，c372e7c：误差引导的对齐抽象，象限正确率 88%。
- 2026-10-01，`milestone/rule-extraction-rate`（b56b1a7）：速率版按结果学习规则，陷阱测试 98.5%。
- 2026-10-01，`milestone/multilayer-spiking`（5de0c7f）：多层共现绑定、反馈补全与想象。

## 未解决的问题与下一步

在脉冲网络里加入结果通道和动作通道；中间层按误差通过三因素规则学习；前馈改为可学。误差学习的瓶颈与[第 4 段](../4-error-correction/STATUS.md)相连。

## 复现

在仓库根目录执行：

```powershell
node demo/uat.mjs
node demo/rule-extraction.mjs
node demo/fig8-two-layer.mjs
node verify/L4-uat.mjs
node verify/L4-two-layer.mjs
node verify/L4-hierarchy.mjs
```

`demo/uat.mjs` 约 3.5 分钟，规则提取几秒；多层验证约 10 分钟，没有缓存时先训练约 20 分钟。缓存见 [CACHE.md](../../CACHE.md)，早期诊断存于 [experiments](../../../experiments/README.md)。

来源：公开发布计划附录 A3 第 3 段、[README.zh-CN.md](../../../README.zh-CN.md) 与相关提交说明。返回[路线图](../../ROADMAP.md)。
