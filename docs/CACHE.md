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

- 产生与使用脚本：[demo/negative-error.mjs](../demo/negative-error.mjs)（`G`、`NEG`）与 [demo/handover.mjs](../demo/handover.mjs)（`GAIN`、`ADAPT`）共用。
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
