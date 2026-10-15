# bio-brain：逐层生物脑模型与闭环架构

从神经元、突触和可塑性机制，构建多层脉冲网络与闭环模型。每层记录代谢 ATP 与动力学能量。

本分支收录截至 **2026-10-14（America/Los_Angeles）** 的最新架构设计、完整实现源码和实验报告。它是研究代码快照。**W4-diag4 已完成，但未通过；W4-r4 与 W6–W8 尚未开始。**

## 阅读入口

- [最新架构设计](docs/ARCHITECTURE.md)
- [实现范围、代码来源与验收状态](docs/PUBLICATION.md)
- [闭环工作包状态](docs/loop/STATUS.md)
- [实验报告目录](docs/REPORTS.md)
- [最新 W4 边界检测诊断报告](docs/loop/attempts/w4-diag4.md)
- [原工程计划](docs/plans/loop-v1-engineering.md)与[边界检测重构计划](docs/plans/w4-boundary-reconstruction.md)
- [基础模型技术说明](README.zh-CN.md)与[路线图](docs/ROADMAP.md)

## 代码与实验状态

`src/`、`demo/`、`verify/` 共 68 个文件，内容来自最新 W4-diag4 归档，其中包含此前已集成的基础模型、E1、慢受体、三因素学习、世界与解码实现，以及最新边界检测构建器和诊断工具。验证器所需的基准工具已从 `experiments/` 移到 `verify/`，并同步一处导入路径。[源码清单](docs/SOURCE-MANIFEST.md)列出逐文件 SHA-256。

W0、W1、W2、W3、W5 的历史验收已通过。W4-diag4 的 14 个固定点完整结束，场景一上游配置合格 **0/12**，没有选定配置或 L 增益。最新实现的收录不表示 W4 已通过正式验收。

## 运行与验证

需要 Node.js ≥ 20；运行源码不需要安装第三方依赖。

```powershell
node verify/L1-neuron.mjs
node verify/L2-synapse.mjs
node verify/L4-slow.mjs
node verify/L4-three-factor.mjs
node verify/loop-world.mjs
```

以上命令是分项检查入口。历史运行结果见各报告。本次发布只检查源码一致性、语法、导入路径与四场景结构，没有重跑科学实验。

`npm run verify`、`FAST=1 npm run verify` 与 `verify/loop-build.mjs` 仍含旧 W4 正式验收流程；该流程尚未按新计划第 5 节完善，不能用它宣称本轮 W4-r4 通过。诊断脚本 `demo/loop-diag.mjs` 保留固定网格实现；本次发布没有重新启动网格。

训练缓存、完整脉冲、电导序列、逐点 JSON、运行日志和检查点保留在本地。本分支的实验材料只收录报告正文、汇总表和报告插图。`data/malecns-epg-kernel.json` 是模型使用的输入数据。

## 目录

| 路径 | 内容 |
| --- | --- |
| `src/` | 神经元、突触、可塑性、网络、脑结构和闭环实现 |
| `demo/` | 演示与最新固定诊断工具 |
| `verify/` | 分项验证与现有验收入口 |
| `docs/ARCHITECTURE.md` | 最新架构设计 |
| `docs/loop/`、`docs/stages/` | 执行状态与实验报告 |
| `docs/loop/reports/` | 报告附表和插图 |
| `docs/plans/` | 设计对应的工程计划，作为历史规范保存 |
| `figures/` | 基础模型技术说明中的插图 |
| `experiments/README.md` | 实验报告索引与收录规则 |

## 许可

本仓库没有附开源许可证：代码公开可见，但保留所有权利。如需复用，请先联系作者。
