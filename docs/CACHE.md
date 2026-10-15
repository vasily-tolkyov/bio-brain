# 训练缓存：cache-2026-10-06

从 [Release cache-2026-10-06](https://github.com/vasily-tolkyov/bio-brain/releases/tag/cache-2026-10-06) 下载 `bio-brain-cache-2026-10-06.zip`，把压缩包放在仓库根目录，然后执行：

```powershell
Expand-Archive bio-brain-cache-2026-10-06.zip -DestinationPath cache
(Get-ChildItem cache\*.json).Count
```

压缩包内文件直接位于根目录；解压后应得到 48 个 `cache/*.json`。`cache/` 不进入 Git。没有缓存时，能重新训练的脚本会自行训练，耗时见下文。

缓存必须与代码版本对应。网络连接按固定种子依次生成；修改网络构造后再载入旧缓存会出错。不同工作区共享缓存时也适用这一规则，见[工作区约定](ROADMAP.md)。早于最终代码的中间缓存仅供参考；v1 缺少网络代码，不能精确复现。

## 脉冲涌现：1 个

- 文件：`trained-seed1.json`。
- 产生脚本：[demo/lib-trained.mjs](../demo/lib-trained.mjs)，由 `demo/fig5-decode.mjs` 使用。
- 对应版本：`milestone/spiking-emergence` 之后的 master。
- 重新训练：约 4 分钟。

## 多层网络：9 个

点神经元版 5 个：`hier-1.json`、`hier-2.json`、`hier-2-noseq.json`、`hier-3.json`、`hier-4.json`。

- 产生脚本：[demo/lib-hierarchy.mjs](../demo/lib-hierarchy.mjs)。
- 对应版本：`milestone/multilayer-spiking` 之后。
- 首次训练：约 20 分钟。

顶树突版 4 个：`hier-2-ap.json`、`hier-2-ap-noseq.json`、`hier-3-ap.json`、`hier-4-ap.json`。

- 产生脚本：同上，入口为 `node demo/fig10-hierarchy.mjs apical`。
- 对应版本：`attempt/4/apical-feedback` 之后。
- 计划未单独给出顶树突版的重新训练耗时。

## 平衡层：1 个

- 文件：`bal-1-s9-e0.02.json`。
- 产生脚本：[demo/balanced-layer.mjs](../demo/balanced-layer.mjs)，参数 `BAL_SMAX=9`、`BAL_ETA=0.02`。
- 对应版本：发布时的 master（网络代码与 5134960 相同）。
- 重新训练：约 9 分钟。

## 预测编码默认参数：2 个

- 文件：`pc-2.json`、`pc-2-seq.json`。
- 产生脚本：[demo/predictive-hierarchy.mjs](../demo/predictive-hierarchy.mjs)，默认参数。
- 对应版本：`milestone/sequence-expectation` 与发布时的 master。
- 重新训练：约 12 + 15 分钟。

## 预测编码 2×2 对照变体：12 个

产生脚本同为 `demo/predictive-hierarchy.mjs`，使用 `PCO` 参数。文件后缀是 PCO 的 JSON 去掉符号后的形式，例如 `tdKindexcsTD1predTauPre0` 对应 `'{"tdKind":"exc","sTD":1,"predTauPre":0}'`。

| 参数后缀 | 物体训练缓存 | 序列训练缓存 |
| --- | --- | --- |
| `tdKindexcsTD1` | `pc-2-tdKindexcsTD1.json` | `pc-2-seq-tdKindexcsTD1.json` |
| `tdKindexcsTD1predTauPre0` | `pc-2-tdKindexcsTD1predTauPre0.json` | `pc-2-seq-tdKindexcsTD1predTauPre0.json` |
| `tdKindnmdasTD2predTauPre0` | `pc-2-tdKindnmdasTD2predTauPre0.json` | `pc-2-seq-tdKindnmdasTD2predTauPre0.json` |
| `tdKindnmdasTD2predTauPre200` | `pc-2-tdKindnmdasTD2predTauPre200.json` | `pc-2-seq-tdKindnmdasTD2predTauPre200.json` |
| `tdKindnmdasTD3predTauPre0` | `pc-2-tdKindnmdasTD3predTauPre0.json` | `pc-2-seq-tdKindnmdasTD3predTauPre0.json` |
| `tdKindnmdasTD3predTauPre200` | `pc-2-tdKindnmdasTD3predTauPre200.json` | `pc-2-seq-tdKindnmdasTD3predTauPre200.json` |

它们是技术说明第四步的 2×2 对照产物。其中 2026-10-03 白天生成的 `predTauPre0` 一组早于 586753a 最终代码，仅供参考。计划未逐项给出变体的重新训练耗时。

## 序列痕迹中间结果：1 个

- 文件：`pc-2-seq-trace200.json`。
- 关联脚本：`demo/predictive-hierarchy.mjs` 的历史中间版本。
- 对应版本：586753a 之前的中间结果，仅供参考；没有指定可精确对应的提交与耗时。

## 负误差与加深上层：10 个

文件：`neg-G1.json`、`neg-G1-negKindnmda.json`、`neg-G1.3.json`、`neg-G1.3-negKindnmda.json`、`neg-G1.4.json`、`neg-G1.4-b40.json`、`neg-G1.4-b60.json`、`neg-G1.4-b100.json`、`neg-G1.4-negKindnmda.json`、`neg-G1.4-sJE5.json`。

- 产生与使用脚本：[demo/negative-error.mjs](../demo/negative-error.mjs)（`G`、`NEG`）与 demo/handover.mjs（历史文件：`handover.mjs`，未纳入本次发布）（`GAIN`、`ADAPT`）共用。
- 对应版本：`attempt/4/negative-error`、`attempt/4/handover`。
- 重新训练：每份约 7 分钟。

## 上层交接：7 个

文件：`ho-g1-t200.json`、`ho-g1.4-t0.json`、`ho-g1.4-t200.json`、`ho-g1.4-t200-cut.json`、`ho-g1.4-t200-b40.json`、`ho-g1.4-t200-b60.json`、`ho-g1.4-t200-b100.json`。

- 产生脚本：`demo/handover.mjs`，参数 `GAIN`、`TAUN`、`ADAPT`、`NEGCUT`。
- 对应版本：`attempt/4/handover`。
- 重新训练：每份约 40 分钟。

## 深层预测 v1：2 个

- 文件：`deep-nobias.json`、`deep-bias.json`。
- 早期脚本：标签 `attempt/4/deep-predictive` 上的 [experiments/deep-record.mjs](https://github.com/vasily-tolkyov/bio-brain/blob/attempt/4/deep-predictive/experiments/deep-record.mjs)。
- v1 的网络代码没有提交，不能精确复现；未给出可用于重训的准确代码版本与耗时。

## 深层预测 v2：3 个

- 文件：`deep2-nobias.json`、`deep2-bias.json`、`deep2-nobias-cueMax8sDS1.json`。
- 产生脚本：标签上的 [demo/deep-predictive.mjs](https://github.com/vasily-tolkyov/bio-brain/blob/attempt/4/deep-predictive/demo/deep-predictive.mjs)，参数 `BIAS`、`DEEP`。
- 对应版本：`attempt/4/deep-predictive`。
- 重新训练：每份约 30 分钟。
- 结论和局限见[尝试记录](stages/4-error-correction/attempts/deep-predictive.md)。

## 合计与来源

共 48 个文件：1 + 9 + 1 + 2 + 12 + 1 + 10 + 7 + 2 + 3。来源为公开发布计划附录 A5、现有 cache 文件名与关联脚本。实验结果的意义见[路线图](ROADMAP.md)和[完整技术说明](../README.zh-CN.md)。

## 有向通道演化链 v2：新增实验缓存

这一族不在上述 48 个发布缓存中。对应[实验标签 attempt/2/attractor-chain-v2](https://github.com/vasily-tolkyov/bio-brain/tree/attempt/2/attractor-chain-v2) 的 GB_RATE 实验，配置、来源权重和学习增益均有指纹校验；不能与第一版的 `chain-*` 混用。实验过程及停止条件见[尝试记录](stages/2-spiking-network/attempts/attractor-chain-v2.md)。本次通过规则、吸引子和学习增益门槛，S2 两次补救后仍失败，后续阶段未运行。

- `chain-v2-attractor-progress.json`、`chain-v2-attractors.json`：300 次变体训练的精确续算检查点与最终吸引子权重。
- `chain-v2-learn-g*.json`：六个学习态增益的静息、补全测量；只读权重，不重新训练。
- `chain-v2-sequence-*.json`、`chain-v2-main.json`：序列训练与两次允许的补救；`*-progress.json` 保存完整动力学和随机数状态。
- `chain-v2-calib-g*-b*.json`：24 个演化态工作点。
- `chain-v2-trials-*.json`、`chain-v2-control-*.json`、`chain-v2-shuffled*.json`：主测试、四种动力学对照与原规则比较；后缀 `-gbnmda` 为原规则对照的独立缓存。

只有实际运行的阶段才会生成相应文件。产生脚本为标签上的 [demo/attractor-chain.mjs](https://github.com/vasily-tolkyov/bio-brain/blob/attempt/2/attractor-chain-v2/demo/attractor-chain.mjs)，通过 `MODE=attractors/learn-gain/sequence/calib/test/controls` 分阶段运行；并行测量进程只写各自缓存，聚合结果由 `COLLECT_ONLY=1` 单独生成。首次完成全部阶段约需数小时，失败时按记录中的门槛停止。

规则验收另外保存九个 `rate-rule-{classes,direction,rest}-{gb_rate,gb_nmda,gb_nmda_eta0}.json`，由标签上的 [demo/rate-rule.mjs](https://github.com/vasily-tolkyov/bio-brain/blob/attempt/2/attractor-chain-v2/demo/rate-rule.mjs) 生成，包含配置和规则参数指纹。这些缓存同样不进入 Git；九项汇总数据保存在标签上的 `figures/rate-rule-results.json`。

v2 完成时实际新增 15 个链缓存与九个规则缓存；与 48 个发布缓存及两个第一版缓存合计 74 个。完整回归后的哈希核对确认原有 50 个缓存全部未变；新增文件清单与 SHA-256 见标签上的 `experiments/logs/2026-10-09-attractor-chain-v2/cache-snapshot.json`，核对记录见同目录的 `cache-integrity.json`。

## 有向通道演化链 v3：动力学需求与学习候选

代码与数据位于[实验标签 attempt/2/attractor-chain-v3](https://github.com/vasily-tolkyov/bio-brain/tree/attempt/2/attractor-chain-v3)。v3 从 v2 的 320×40 ms 权重开始，不改 `src` 或版本为 2 的 `CONFIG`。四种学习设置、来源权重和测量条件写入新缓存的独立指纹。开始前的 74 个缓存按 SHA-256 核对，均作为只读输入。设计与决策规则见[v3 尝试记录](stages/2-spiking-network/attempts/attractor-chain-v3.md)。

- `chain-v3-calib-g{G}-b{b}.json`：六个增益 × 四个适应强度，每点十次终点 F 线索和 3 s 静息。
- `chain-v3-dose-k{k}.json` 及其 `-progress.json`：k=1/2/3/5 的剂量诊断。人工放大只改七条正向连接，记录截断与实际组平均；k>1 的结果用于测需求，不能代替学到权重的成功证据。
- `chain-v3-seq-{L1a,L1b,L1c,L1d}-{160,320}-40.json` 及其 `-progress.json`：四个候选的序列状态；320 回合只在决策规则授权唯一一次补救时生成。检查点还保存候选突触前 NMDA 迹，用于精确续算。
- `chain-v3-rate-{classes,rest}-{候选}.json`：候选四类选择性与 30 s 静息复查，汇总为 `figures/rate-rule-v3.json`。
- `chain-v3-cand-*.json`：情形 B 的候选完整标定网格；`chain-v3-main-*.json`、`chain-v3-control-*.json`、`chain-v3-shuffled*.json`：学到权重的主测试、四项对照与倒序重训。

仅实际执行的阶段才生成缓存。产生脚本为 `demo/attractor-chain.mjs` 的 `MODE=v3-calib/v3-sequence/v3-dose/v3-decision/v3-remedy/v3-test/v3-controls/v3-all`，候选参数 `CANDIDATE` 和并行工作条件 `G/B/K/CONTROL` 均受固定列表限制。工作进程只写各自缓存，结果用 `COLLECT_ONLY=1` 或 `MODE=v3-learning-collect` 聚合；`v3-all` 可按相同决策规则依次重做。规则复查由 `demo/rate-rule.mjs` 的 `CANDIDATE` 参数传入，`MODE=all` 对该候选只运行四类和静息。

哈希基线与结束核对分别为 `experiments/logs/2026-10-09-attractor-chain-v3/cache-before.json`、`cache-integrity.json`。新增缓存同样不进入 Git；完整测量汇总在 `figures/chain-results.json` 的 `v3` 节，图为 `figures/fig14-chain-dose.png`。

本次实际新增 24 个标定缓存、四个候选状态及四个续算检查点、八个规则复查缓存，共 40 个；合计 114 个。24 点无工作点，按情形 C 停止，剂量、候选重标定、320 回合补救、主测试和对照缓存均未生成。原有 74 个文件的 SHA-256 全部不变。
