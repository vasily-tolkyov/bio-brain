# 深层预测网络：v1 / v2

- 状态：失败。
- 代码：[attempt/4/deep-predictive](https://github.com/vasily-tolkyov/bio-brain/tree/attempt/4/deep-predictive)。v2 的网络选项为 `pcDeep`；v1 的网络代码没有提交。
- 记录日期：2026-10-06。

## 目标与假设

把“现在的证据”和“接下来是什么”分给浅层与深层，让深层学习 X→Y，并在 X 期间通过中继表达 Y 的预测。v1 在输出侧学习；v2 把学习移到带痕迹的输入线索，检查能否获得有方向的预测。

## 做法

- v1：深层到中继使用钙规则学习。仅保留了[早期脚本](https://github.com/vasily-tolkyov/bio-brain/blob/attempt/4/deep-predictive/experiments/deep-record.mjs)和日志，不能精确复现。
- v2：学习只在输入侧。线索只连别组，带 200 ms 痕迹；老师 `ED` 同组、强度 6；深层到中继 `DS` 固定。代码见[deep-predictive.mjs](https://github.com/vasily-tolkyov/bio-brain/blob/attempt/4/deep-predictive/demo/deep-predictive.mjs)与[layered-network.mjs](https://github.com/vasily-tolkyov/bio-brain/blob/attempt/4/deep-predictive/src/L4-network/layered-network.mjs)。
- 比较无偏向、有偏向和 `cueMax 8 + sDS 1` 条件。脚本与 26 个 v1 / v2 日志保留在上述标签上。

## 结果

### v1

- 四个方向都学成 0.27–0.33。
- 中继自发冲到 37–46 Hz；正误差被压到 0.0–0.2 Hz。
- 期待为 −0.9 Hz。

### v2

- 线索效能：X→Y 0.224、Y→X 0.154、同物体交叉 0.336 / 0.339、无关连接 0.217。
- 前几遍每遍 Δρ（×10⁻³）：X→Y 11–28，同物体交叉 33–85，无关连接 14–18。
- 训练后 X 的最后 50 ms，深层 Y 只有 2.0 Hz，低于深层自发的 4.3 Hz；同一段时间 Y 的中继 2.3 Hz，而 Y 真来之后是 30–40 Hz。
- 学到的期待：无偏向 −1.1 Hz，有偏向 +1.6 Hz，`cueMax 8 + sDS 1` 时 +1.3 Hz，均在约 ±1.5 Hz 的测量噪声内。

## 结论与原因

带痕迹的赫布学习主要学到共现（同物体交叉、非特异连接），而不是先后；异突触预算把特异信号拉平。老师作为驱动，又让深层成为“现在”的拷贝。尚未得到超出噪声的序列期待。

## 遗留问题

下一步是 `three-factor-cue`：线索改用误差驱动的三因素规则，减弱 `ED`，处理负误差的接线，并在平衡后冻结或放慢表征细胞的抑制性可塑性。具体方案见 master 上的 `docs/stages/4-error-correction/STATUS.md`。

来源：公开发布计划附录 A4 第 9 条，以及标签上的[实验日志](https://github.com/vasily-tolkyov/bio-brain/tree/attempt/4/deep-predictive/experiments/logs/4-deep-predictive)。
