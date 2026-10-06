# 实验档案

这些脚本、日志和结果从开发时的 `scratch/` 目录复制归档。它们是存档，不保证能在当前代码上直接运行；原始目录保持原样。脚本平放在这一层，是为了让其中 `../demo/`、`../src/` 的相对导入仍然有效。

master 共保留 23 个脚本、45 个日志与结果，加上本页共 69 个文件。分组及历史版本对应关系如下；“约”表示开发时期的对应关系，不保证当时的每个中间版本都已提交。

## 脚本

### 第 2 段：脉冲涌现诊断

对应：milestone/spiking-emergence（09529fe）。

- [emerge.mjs](emerge.mjs)
- [equiv.mjs](equiv.mjs)
- [blocks.mjs](blocks.mjs)

### 第 3–4 段：多层网络诊断

对应：milestone/multilayer-spiking 至 attempt/4/apical-feedback（5de0c7f–9b1e505）。

- [probe-ff.mjs](probe-ff.mjs)
- [seq-item.mjs](seq-item.mjs)
- [train-noseq.mjs](train-noseq.mjs)
- [imagery.mjs](imagery.mjs)
- [lat-check.mjs](lat-check.mjs)
- [l1-blocks.mjs](l1-blocks.mjs)
- [diag-l2.mjs](diag-l2.mjs)
- [budget-l2.mjs](budget-l2.mjs)
- [check-l2.mjs](check-l2.mjs)

### 第 3 段：对齐抽象早期版本

对应：c372e7c；demo/aligned-abstraction.mjs 的早期版本。

- [aligned-abstraction-early.mjs](aligned-abstraction-early.mjs)

### 第 4 段：预测编码与序列期待诊断

对应：milestone/predictive-coding 至 milestone/sequence-expectation（b501089–586753a）。

- [pc-expect-cut.mjs](pc-expect-cut.mjs)
- [pc-unit.mjs](pc-unit.mjs)
- [pc-half.mjs](pc-half.mjs)
- [pc-timing.mjs](pc-timing.mjs)
- [pc-struct.mjs](pc-struct.mjs)
- [pc-ledger-trace.mjs](pc-ledger-trace.mjs)
- [pc-seq-trace.mjs](pc-seq-trace.mjs)

### 第 4 段：负误差与上层增益扫描

对应：attempt/4/negative-error（77a909c）。

- [neg-test.mjs](neg-test.mjs)
- [deep-scan.mjs](deep-scan.mjs)

### 第 4 段：交接早期版本

对应：attempt/4/handover（5134960）；demo/handover.mjs 的早期版本。

- [handover-early.mjs](handover-early.mjs)

## 日志与结果


### 第 3 段：多层网络结果

对应：milestone/multilayer-spiking（约 5de0c7f）。共 3 个文件。

- [seq-item-15.json](logs/3-multilayer/seq-item-15.json)
- [seq-item-20.json](logs/3-multilayer/seq-item-20.json)
- [seq-item-25.json](logs/3-multilayer/seq-item-25.json)

### 第 4 段：预测编码

对应：milestone/predictive-coding（约 b501089）。共 10 个文件。

- [pc2-a.log](logs/4-predictive-coding/pc2-a.log)
- [pc2-b.log](logs/4-predictive-coding/pc2-b.log)
- [pc2-seq.log](logs/4-predictive-coding/pc2-seq.log)
- [pc2-nmda2.log](logs/4-predictive-coding/pc2-nmda2.log)
- [pc2-nmda3.log](logs/4-predictive-coding/pc2-nmda3.log)
- [pc2-nmda2-seq.log](logs/4-predictive-coding/pc2-nmda2-seq.log)
- [pc2-nmda3-seq.log](logs/4-predictive-coding/pc2-nmda3-seq.log)
- [pc2-final.log](logs/4-predictive-coding/pc2-final.log)
- [verify-after-cleanup.log](logs/4-predictive-coding/verify-after-cleanup.log)
- [verify-final.log](logs/4-predictive-coding/verify-final.log)

### 第 4 段：序列期待

对应：milestone/sequence-expectation（约 586753a）；verify-step4.log 保留完整验证通过记录（2919 s）。共 25 个文件。

- [ledger-200.log](logs/4-sequence-expectation/ledger-200.log)
- [seq-trace-200.log](logs/4-sequence-expectation/seq-trace-200.log)
- [pc0.log](logs/4-sequence-expectation/pc0.log)
- [pc0r.log](logs/4-sequence-expectation/pc0r.log)
- [pcA.log](logs/4-sequence-expectation/pcA.log)
- [pcAr.log](logs/4-sequence-expectation/pcAr.log)
- [pcB.log](logs/4-sequence-expectation/pcB.log)
- [pcBr.log](logs/4-sequence-expectation/pcBr.log)
- [pcC.log](logs/4-sequence-expectation/pcC.log)
- [pcCr.log](logs/4-sequence-expectation/pcCr.log)
- [pcD.log](logs/4-sequence-expectation/pcD.log)
- [pcDr.log](logs/4-sequence-expectation/pcDr.log)
- [pcE.log](logs/4-sequence-expectation/pcE.log)
- [verify-step4.log](logs/4-sequence-expectation/verify-step4.log)
- [predictive-results-predTauPre200.json](logs/4-sequence-expectation/predictive-results-predTauPre200.json)
- [predictive-results-tdKindnmdasTD2predTauPre0.json](logs/4-sequence-expectation/predictive-results-tdKindnmdasTD2predTauPre0.json)
- [predictive-results-tdKindnmdasTD2predTauPre200.json](logs/4-sequence-expectation/predictive-results-tdKindnmdasTD2predTauPre200.json)
- [predictive-results-tdKindnmdasTD3.json](logs/4-sequence-expectation/predictive-results-tdKindnmdasTD3.json)
- [predictive-results-tdKindnmdasTD3predTauPre200.json](logs/4-sequence-expectation/predictive-results-tdKindnmdasTD3predTauPre200.json)
- [fig12-check.png](logs/4-sequence-expectation/fig12-check.png)
- [fig12-predictive-predTauPre200.png](logs/4-sequence-expectation/fig12-predictive-predTauPre200.png)
- [fig12-predictive-tdKindnmdasTD2predTauPre0.png](logs/4-sequence-expectation/fig12-predictive-tdKindnmdasTD2predTauPre0.png)
- [fig12-predictive-tdKindnmdasTD2predTauPre200.png](logs/4-sequence-expectation/fig12-predictive-tdKindnmdasTD2predTauPre200.png)
- [fig12-predictive-tdKindnmdasTD3.png](logs/4-sequence-expectation/fig12-predictive-tdKindnmdasTD3.png)
- [fig12-predictive-tdKindnmdasTD3predTauPre200.png](logs/4-sequence-expectation/fig12-predictive-tdKindnmdasTD3predTauPre200.png)

### 第 4 段：上层交接

对应：attempt/4/handover（约 5134960）。共 7 个文件。

- [ho-b100.log](logs/4-handover/ho-b100.log)
- [ho-b40.log](logs/4-handover/ho-b40.log)
- [ho-b60.log](logs/4-handover/ho-b60.log)
- [ho-cut.log](logs/4-handover/ho-cut.log)
- [ho-main.log](logs/4-handover/ho-main.log)
- [ho-notrace.log](logs/4-handover/ho-notrace.log)
- [ho-shallow.log](logs/4-handover/ho-shallow.log)

## 深层预测网络与未归档文件

深层预测网络 v1 / v2 的脚本 `experiments/deep-record.mjs` 和 26 个日志只在 [attempt/4/deep-predictive 标签](https://github.com/vasily-tolkyov/bio-brain/tree/attempt/4/deep-predictive/experiments)上；其中 v1 的网络代码没有提交，不能精确复现。v2 的 `demo/deep-predictive.mjs` 也在该标签上。

空文件 `scratch/pc2-seq.err` 不归档。全部 96 个 scratch 文件的对应关系为：23 个 master 脚本 + 1 个深层早期脚本 + 45 个 master 日志与结果 + 26 个深层日志 + 1 个空文件。

来源：公开发布计划附录 C；开发时的 scratch 文件。各段的结论与复现入口见 [路线图](../docs/ROADMAP.md)。
