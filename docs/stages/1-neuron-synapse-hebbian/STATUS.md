# 第 1 段：神经元、突触、赫布学习

状态：完成。快照日期：2026-10-06。已完成的本段不单独建立 stage 分支。

## 目标

生物物理可信的神经元、突触和赫布学习，能按真实离子流记 ATP。

## 完成判据

以下验证全部通过：`verify/L1-neuron.mjs`、`verify/L2-synapse.mjs`、`verify/L3-hebbian.mjs`。

## 现状

- HH 模型静息 −65.00 mV。6.3 °C 一次放电 Na⁺ 1377 nC/cm²、重叠系数 13.0（文献 1098、11.2）；18.5 °C 时 360 nC/cm²、3.76（文献 331、3.85）。
- LIF 的适应让放电间隔从 10 ms 变成 38 ms。
- 突触：EPSP 与接触数 r = 1.000；−20 mV 时 NMDA 电流是 −70 mV 时的 3.26 倍；突触后 1.65×10⁵ ATP/囊泡。
- 钙规则 STDP：+5 ms 增强 0.059，−20 ms 减弱 0.063。
- 一起激活之后，所需上游驱动 41 → 30 Hz，系统 ATP −27%。

## 尝试与里程碑

- 2026-10-01，`milestone/L1-L5-base`（f8be8ee）：L1 神经元、L2 突触、L3 钙规则赫布学习，以及 L4/L5 基础模型全部验证。

## 未解决的问题与下一步

群体实验把钙规则的时间常数缩短了 5 倍，见 [技术说明](../../../README.zh-CN.md)“已知缺口”第 4 条。更大的加速会破坏慢变量假设。

## 复现

在仓库根目录执行：

```powershell
node verify/L1-neuron.mjs
node verify/L2-synapse.mjs
node verify/L3-hebbian.mjs
node demo/fig1-L1-L3.mjs
```

来源：公开发布计划附录 A3 第 1 段、[README.zh-CN.md](../../../README.zh-CN.md) 与 f8be8ee 提交说明。返回[路线图](../../ROADMAP.md)。
