# 第 4 段：规则误差校对

状态：部分完成，当前主攻。快照日期：2026-10-06。集成分支：`stage/4-error-correction`。

## 目标

上层的预测与下层的证据比较；误差修正上层的状态，并驱动学习，让预测（包括“接下来是什么”）变对。

## 完成判据

- 已达成：`verify/L4-predictive.mjs` 的 10 项检查。
- 草案，未达成：产生预测的通路用误差驱动的规则学成，至少 3 个种子上 X→Y 明显高于无关连接，学到的期待超出噪声；深层在 X 期间表达“接下来是 Y”。

## 现状

### 预测编码两层

b501089 的每层含表征群 E、误差群 N 与预测中继群 S：

- 熟悉的 X：误差在头 25 ms 为 14.5 Hz，之后 3.4 Hz（自发 3.5）。
- 切断预测抑制 / 切断预测下行：39.3 / 33.5 Hz。
- 只给 A 时，缺失的 D 为 1.2 Hz。
- 没见过的 A + C：A、C 9.4 Hz，B、D 0.9 Hz。
- 误差群四组合计 12.0 Hz，切断预测时为 106.5。
- X 换成 Y 后头 50 ms，Y 的误差为 17.0 Hz；切断误差上行时为 31.4。

### 序列期待

586753a 中，中继到误差的抑制性可塑性使用 200 ms 突触前痕迹，预测下行只用 NMDA。学到的期待为 8.3 Hz；2×2 对照依次为 −0.1 / 0.3 / 2.0 / 8.3 Hz。X 的中继到 Y 的误差连接为 0.42 → 1.13；X 之后 Y 不来时，Y 的误差为 0.3 Hz（自发 3.0）。

这在误差层面成功，但上层仍不会在 X 之后提前转到 Y。77a909c 的 `pcNeg` 负误差细胞是实验选项，能纠正加深上层的部分失稳，见[记录](attempts/negative-error.md)。

### 产生预测的通路尚未学对

2026-10-05 的诊断指出两个瓶颈：

- 上层递归、深层线索、深层到中继使用共现式钙规则，会学习同时出现、反向和非特异相关；预测对了也不会停，只能靠异突触预算兜住。唯一成功的中继到误差那组使用 Vogels 抑制性可塑性，平均下来为“突触前 ×（误差细胞频率 − 目标）”，即误差驱动的 delta 规则。
- 表达预测需要深、能自持的动力学，正确学习却需要浅、由输入驱动的动力学：加深时方向学反，浅时方向对但表达不出。“现在”与“想象的下一步”争用同一群神经元；表征细胞上的快速稳态又抵消学到的吸引子深度。

测量噪声：几乎相同的两个网络，训练前测出的“反向 − 正向”分别为 2.8 与 1.4 Hz。“学到的期待”是两次测量之差，±1.5 Hz 以内的差别不能当作效果。深层 v2 的期待 −1.1 / +1.6 / +1.3 Hz 尚不能视为成功。

## 尝试与里程碑

- 2026-10-02，`attempt/4/apical-feedback`（9b1e505）：部分成功，顶树突反馈使上传证据干净，预测仍到不了正在感知的第一层。[记录](attempts/apical-feedback.md)
- 2026-10-02，`attempt/4/deep-population-v0`（9830f18；构造见 2380c32）：失败，预测抵达第一层顶树突，但深层不生成下一个物体。[记录](attempts/deep-population-v0.md)
- 2026-10-02，`attempt/4/theta-adp`（d8ebf94）：失败，学习后深层同步，权重变均匀。[记录](attempts/theta-adp.md)
- 2026-10-03，`attempt/4/checkpoint-variants`（eaf9f75）：失败变体合集，预测与证据共用状态形成正反馈。[记录](attempts/checkpoint-variants.md)
- 2026-10-03，`milestone/predictive-coding`（b501089）：成功，表征、误差和预测中继分群，物体级误差被解释掉。[记录](attempts/predictive-coding.md)
- 2026-10-04，`milestone/sequence-expectation`（586753a）：误差层面成功，200 ms 痕迹与 NMDA-only 中继带来 8.3 Hz 期待。[记录](attempts/sequence-expectation.md)
- 2026-10-04，`attempt/4/negative-error`（77a909c）：部分成功，负误差稳定加深上层，但静息自燃仍为 17–24%。[记录](attempts/negative-error.md)
- 2026-10-04，`attempt/4/handover`（5134960）：失败，加深后学到反向连接，上层不提前切换到 Y。[记录](attempts/handover.md)
- 2026-10-06 归档，`attempt/4/deep-predictive`：失败，v1 非特异抑制误差，v2 主要学习共现，期待仍在噪声内。[记录](attempts/deep-predictive.md)

## 未解决的问题与下一步

第 2 段发现原钙规则在有自发活动的网络里不加选择地增强；深层 v2 的非特异线索和 handover 的反向学习值得用频率依赖变体 `GB_RATE` 复查。

下一步仅为计划，尚未开始：从 `attempt/4/deep-predictive` 建立 `exp/4/three-factor-cue`。

- 将线索 `L.CUE` 从钙规则改成三因素规则：Δw =（突触前 200 ms 资格痕迹减去它的滑动平均）×（第一层同组误差：正误差 N 增强，负误差 M 削弱）。
- 去掉或大幅减弱浅层到深层的驱动 `ED`，让深层只靠学到的线索放电。
- 现有负误差 M 的预测来自上层浅层 `NEGP`，不来自深层；要么给深层加一条到 M 的连接，要么改用 N 低于基线的部分。但后者在 X 期间会因竞争一直为负。
- 表征细胞上的抑制性可塑性在平衡后冻结或放慢；快速 Vogels 规则只留在误差细胞上。

## 复现

在对应版本的仓库根目录执行：

```powershell
node verify/L4-predictive.mjs
node demo/negative-error.mjs
node demo/handover.mjs
```

预测编码验证约 25 分钟，需要 `pc-2*.json`，没有时先训练约 30 分钟；负误差与交接 demo 各约 40 分钟。深层 v2 只在 [attempt/4/deep-predictive](https://github.com/vasily-tolkyov/bio-brain/tree/attempt/4/deep-predictive) 上，执行 `node demo/deep-predictive.mjs` 约 30 分钟。v1 的网络代码未提交，不能精确复现。

缓存版本见 [CACHE.md](../../CACHE.md)，档案见 [experiments](../../../experiments/README.md)。来源：公开发布计划附录 A3/A4、[README.zh-CN.md](../../../README.zh-CN.md)、提交说明及 demo 文件头。返回[路线图](../../ROADMAP.md)。
