# 第 6 段：行动层

状态：未开始，只有速率版 PFL3 转向读出。快照日期：2026-10-06。集成分支：`stage/6-action`。

## 目标

动作改变后续输入；动作的副本告诉前向模型该期待哪个转移、在什么时候；与环境形成闭环。

## 完成判据

草案：在动作触发的小世界里，网络按结果学会选动作，并靠动作副本及时给出预测。

## 现状

动作闭环未开始。已有速率版 PFL3 转向读出，朝向目标时为零且稳定。

## 尝试与里程碑

- 2026-10-01，`milestone/L1-L5-base`（f8be8ee）：包含 PFL3 速率转向读出；尚无行动闭环里程碑。

## 未解决的问题与下一步

- 把规则提取的挖方块世界改成序列式：看到方块 → 选工具 → 得到结果。
- 动作选择用多巴胺式第三因子；脉冲版 actor-critic 可参考 Frémaux, Sprekeler & Gerstner 2013。
- 与[规则抽象](../3-rule-abstraction/STATUS.md)、[目标规划](../5-planning/STATUS.md)和[误差校对](../4-error-correction/STATUS.md)衔接。

## 复现

`node verify/L5-brain.mjs` 的 PFL3 部分。动作闭环暂无复现脚本。

来源：公开发布计划附录 A3 第 6 段、[README.zh-CN.md](../../../README.zh-CN.md) 与 f8be8ee 提交说明。返回[路线图](../../ROADMAP.md)。
