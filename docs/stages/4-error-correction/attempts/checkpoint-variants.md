# 清理前的预测变体合集

- 状态：失败（变体合集）。
- 代码：`attempt/4/checkpoint-variants`（eaf9f75，2026-10-03）。

## 目标与假设

尝试通过中间神经元、突触和树突学习机制、节律与状态切换改善预测的表达与学习。

## 做法

检查点包含：PV / SOM / VIP 中间神经元、快速 GABA_A、电压依赖钙规则、AMPA 拷贝、易化突触、deep1 / deep1a / cx / ampa / ampah、θ–γ、压缩重放、平台门控（BTSP）与树突预测（Urbanczik–Senn）规则、相位性神经调质与编码 / 提取切换。

平衡态也在此引入；它保留了下来，后来成为第一层。

## 结果

- 这些变体未成功表达所需预测。
- 平台规则第一版学成非特异反馈：X→X 14.5、X→Y 13.8、X→不成组 12.6；改成协方差形式后仍未成功。

## 结论与原因

共同根源是把预测加到承载证据的同一群神经元上，形成正反馈；学习、稳态、推断共用同一状态，也没有“预测对了”的信号。后续改成表征、误差与预测中继分群的[预测编码](predictive-coding.md)。

## 遗留问题

需要分开证据与预测，并以误差驱动学习。该检查点保留供历史检查；大部分变体已从 master 删除。

来源：[README.zh-CN.md](../../../../README.zh-CN.md)“走过、已删除的路线”、eaf9f75 提交说明，以及历史版本的 [dendritic-prediction.mjs 文件头](https://github.com/vasily-tolkyov/bio-brain/blob/attempt/4/checkpoint-variants/demo/dendritic-prediction.mjs)。
