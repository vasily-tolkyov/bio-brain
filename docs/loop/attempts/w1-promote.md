# W1：提升与收线

状态：本轮通过（用户已明确验收扫描范围；原范围预检保留于旧标签）。日期：2026-10-10（America/Los_Angeles）。

分支：`exp/loop/w1-promote`；工作区：`D:\kimi_kairos\bio-brain-wt\e-loop-w1`；基于 `stage/loop`。

目标：提升 v2 的学习规则与可复用 E1，保持历史数值，并按计划收线旧选项。首先核对修改范围和验收是否可以同时满足；发现停止条件时不继续实施。

依据：工程计划第 4 节 W1；设计基线为 `76b5179:docs/ARCHITECTURE.md`。下面原样记录本包的改动范围与验收条目。通用记录、日志和收尾依计划第 1、2、5 节执行。

## 计划原文

### W1 提升与收线（`exp/loop/w1-promote`）

**改动范围：**

- 从标签取出四个文件（它们与 master 的差别只有 v2 的改动）：

  ```powershell
  git checkout attempt/2/attractor-chain-v2 -- src/L3-plasticity/calcium-rule.mjs src/L4-network/spiking-network.mjs verify/L3-hebbian.mjs verify/L4-emergence.mjs
  ```

- 新建 `src/loop/e1.mjs`：从标签上的 `demo/attractor-chain.mjs` 只抽出可复用部分，其余不带：
  - 抽出：`CONFIG`（改名为 `E1_DEFAULTS`）、`RULES`、组与原型的辅助函数、`buildChain`（改名为 `buildE1`，随机数的消耗顺序不变）、`setGain`、`run`（含刺激期学习的门控）、`variant`、`resumableRng`、`captureRuntime` / `restoreRuntime`、`groupBins`、`mean`；
  - 不带：序列、标定、试次、对照与作图。
- 从标签移植 `demo/rate-rule.mjs` 与 `verify/L4-rate-rule.mjs`：导入改为 `src/loop/e1.mjs`；缓存前缀改为 `loop-rate-rule-`，让它重新计算而不是读旧缓存。
- `verify/all.mjs` 只加入 L4h（FAST 模式下跳过），不加入 L4g。
- 删除 `demo/handover.mjs`；删除 `src/L4-network/layered-network.mjs` 中的 `upperGain`、`upperTauN`、`upperAdapt`（代码与注释）。
- 新增 `figures/rate-rule-results.json`（由移植后的 demo 生成）。

**验收：**

1. `node verify/L3-hebbian.mjs` 退出码为 0。
   - GB_RATE 的 60 次配对，与参考值相差均 ≤ 0.005：Δt = +10 ms 时，1 Hz −0.074、5 Hz −0.113、10 Hz −0.122、20 Hz +0.040、40 Hz +0.112；20 Hz、Δt = −10 ms 时 −0.119。
   - GB_NMDA 在 1 Hz、+10 ms 时为 +0.053，相差 ≤ 0.001。
2. `node verify/L4-emergence.mjs` 退出码为 0，其中包括 GB_RATE 的逐拍对拍。
3. `node verify/L4-rate-rule.mjs` 退出码为 0。重新生成的 `figures/rate-rule-results.json` 与标签上的同名文件逐项比较，每个数值相差 ≤ 1e-12，以此证明 `buildE1` 复现了 `buildChain`。
4. `git grep -n -E "upperGain|upperTauN|upperAdapt"` 没有输出；`demo/handover.mjs` 已删除。
5. 完整的 `npm run verify` 退出码为 0。

**预计耗时**：约 3 小时（含完整回归约 50 分钟、规则复算约 30 分钟）。

## 执行结果

### 范围冲突的实测

在任何代码修改之前，对设计基线执行以下只读检查：

```powershell
git grep -n -E "upperGain|upperTauN|upperAdapt" 76b5179 --
```

命令退出码为 0，共 13 行匹配。其中以下 3 行位于 W1 允许修改范围以外，无法通过范围内的删除消除：

| 文件 | 基线行号 | 内容 |
| --- | --- | --- |
| `docs/ARCHITECTURE.md` | 623 | 架构第 14 节明确列出要删除的三个旧选项名 |
| `experiments/handover-early.mjs` | 1 | 历史实验的注释引用旧选项 |
| `experiments/handover-early.mjs` | 12 | 历史实验实际向构建器传入三个旧选项 |

W1 验收第 4 条要求全仓搜索没有输出；即使将本包范围内的 10 行引用全部移除，上述 3 行仍然存在。证据取自 `76b5179`，与此次记录中按要求抄录的选项名无关。抄录验收条目的记录也会命中全仓搜索，进一步说明检查需要明确扫描范围。

详细结果见 w1-results.json（本地证据：`w1-results.json`，未上传），原始输出见 scope-preflight.log（本地证据：`scope-preflight.log`，未上传）。没有做参数标定。

### 停止依据与验收状态

工程计划第 1.3 节规定：“需要改范围以外的文件时，停止并报告。”第 6 节将“需要改本包改动范围以外的文件”列为停止条件。

因此本包按未通过路径收尾，未提升或删除任何源代码，未运行 L3、L4、规则复算和完整 verify；这些检查均为“未运行”，不记为通过或动力学失败。W0、W2–W8 均未开始。

归档标签为本地 `attempt/loop/w1-promote`。初始记录与结果记录挑拣至 `stage/loop`，按计划删除实验工作区和分支。缓存核对统一记录在闭环 STATUS 和收尾清单中。

### 恢复所需的计划调整

继续执行前，需要明确旧选项检查的扫描范围，并决定如何处理 `experiments/handover-early.mjs` 中仍在使用旧选项的历史脚本。例如，将该脚本加入 W1 的收线范围，将检查限定为运行代码并排除设计与记录文档。本次未自行改写计划、缩小验收范围或扩展修改清单。

## 2026-10-10 继续执行

用户已明确选择“仅检查 src/、demo/、verify/，保留设计文档与归档实验中的历史引用”。本轮 W1 第 4 条据此使用 `git grep -n -E "upperGain|upperTauN|upperAdapt" -- src/ demo/ verify/`，并检查旧 demo 已删除。其余设计、修改范围、数值验收、缓存与停止条件不变。此前未实现的尝试及原始证据继续保存在 `attempt/loop/w1-promote`，不改写标签。

本轮复用已通过的 S0、D0，从 `stage/loop` 提取 v2 学习规则与 E1 构建，重新计算全部九种规则实验，不读取旧规则缓存。数值比较覆盖科学结果和配置；实验墙钟 `seconds` 独立记录，用于实际用时报告。

### 本轮验收

1. L3 通过，GB_RATE 六项配对及 GB_NMDA 参考值在计划容差内。
2. L4-emergence 通过，含 GB_RATE 逐拍对拍。
3. L4-rate-rule 通过，科学结果与 v2 标签逐项数值差不超过 1e-12；全部新缓存使用 `loop-rate-rule-`。
4. `src/、demo/、verify/` 中没有已收线选项，旧 demo 删除。
5. 完整 `npm run verify` 通过。既有主缓存全部保持原 SHA-256。

### 本轮结果：通过

- L3：15 项全部通过，六项 GB_RATE 配对均在 0.005 容差内，GB_NMDA 为 +0.052954。
- L4-emergence：11 项全部通过；三种规则逐拍对拍的最大 ATP 相对误差为 2.975e-16。
- L4-rate-rule：4 项全部通过，九个实验全部重新计算，独立进程总墙钟 677.449 s；全部缓存使用 `loop-rate-rule-` 前缀。
- 与 v2 标签的结果逐项比较：740 个科学数值，最大差异 0；九个墙钟用时单独记录，没有混作科学结果。
- E1 构建器直接对拍：默认配置与规则诊断配置下，构建、背景运行与一次变体学习后的放电、全部状态向量和连接状态均逐位相同。
- 限定目录扫描没有旧选项名，旧 demo 已删除。四个提升文件与 v2 标签相同。
- 完整 `npm run verify` 退出 0，11 组检查全通过，墙钟 2765 s。检查组边界的检查点与完整日志已保存。
- 既有 114 个缓存的 SHA-256 全部不变；没有参数标定。实现提交 `7447995`，初始记录 `aaaf4b6`。

实测数据见 本轮结果（本地证据：`resume-w1-results.json`，未上传）、逐项比较（本地证据：`resume-rate-equivalence.json`，未上传）、E1 对拍（本地证据：`resume-e1-equivalence.json`，未上传）与完整回归日志（本地证据：`resume-full-verify.log`，未上传）。本轮按通过路径合并；旧范围预检标签保持不变。
