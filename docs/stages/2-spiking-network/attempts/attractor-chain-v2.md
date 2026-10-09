# 有向通道演化链 v2

- 状态：进行中。
- 分支：exp/2/attractor-chain-v2。
- 起点：attempt/2/attractor-chain（272e83d）。
- 开始日期：2026-10-09。

本记录在正式模拟前保留 v2 的诊断依据、设计改动、规则实现、全部验收与停止条件。未特别说明的定义和条件沿用第一版尝试记录。仅允许改动 src/L3-plasticity/calcium-rule.mjs 与 src/L4-network/spiking-network.mjs，新增预设不改变默认规则。

诊断归档说明：源目录实际含 21 个 JSON；20 个主要结果加 README 放在指定目录，额外 Cpre=0.55 探索结果存入相邻 extra 目录，全部保留。e0 文件按计划更名并更正参数描述。
### 0.2 2026-10-09 的诊断

诊断为单一种子，在仓库外的临时目录 `C:\Users\86139\AppData\Local\Temp\chain-diag\` 中进行，网络与随机种子照抄第一版。步骤 R0 会把它们归档进仓库。

- **静息噪声学习**：原规则下，不给任何刺激、只开学习，5 s 内所有兴奋→兴奋效能从 0.10 涨到 0.20（正是预算上限），组内、组间、不成组都一样。
- **只在刺激期学习还不够**：原规则下比值 1.43，补全 +0.27 Hz、5/8 组为正。
- **四类突触**：只刺激 A 组 20 次、只在刺激期学、开预算。
  - 原规则：A→A +0.228，A→组外 +0.227，组外→A +0.085，组外→组外 +0.081。
  - 原因：组外神经元有约 3–5 Hz 的自发放电，A 高频放电时，组外的每次零星脉冲都算一次"前先后后"。
- **原因所在**：原规则 `GB_NMDA` 的同时检测项（η = 2）让单次巧合就增强。下面是 60 次配对、Δt = +10 ms 的 Δρ：
  - 原规则：1 Hz +0.053，20 Hz +0.065，40 Hz +0.112，即任何频率都增强；
  - 原规则去掉同时检测项（η = 0）：20 Hz 以下全部减弱，只有 40 Hz 增强；但 40 Hz 时前后两种时序几乎一样（+0.041 / +0.033），方向性丢失，所以不采用。
- **更正**：诊断过程中曾把 `diag-classes-rg-e0-t100.json` 误记为"突触前后钙各 0.6、η = 0"。文件里记录的参数是原值 0.8 / 1.0、η = 0。四类结果为 A→A +0.219、A→组外 +0.063、组外→A +0.058、组外→组外 +0.007。
- **新规则 `GB_RATE`**：突触前后钙各 0.6、η = 1.2，同时检测钙乘以突触后 100 ms 慢速放电迹。
  - 配对 Δρ（Δt = +10 ms）：1 Hz −0.074、5 Hz −0.113、10 Hz −0.122、20 Hz +0.040、40 Hz +0.112；20 Hz、Δt = −10 ms 时为 −0.119。即低频减弱、高频增强，方向保留。
  - 四类：A→A +0.212，A→组外 +0.010，组外→A +0.027，组外→组外 −0.004。
  - 方向（A 40 ms 接 B 40 ms，40 遍，全程学习）：A→B +0.144，B→A +0.089；A、B 到组外 −0.009，组外到 A、B −0.007，组外之间 −0.020。原规则对应为 A→B +0.090、B→A +0.064，而 A、B 到组外 +0.105，组外到 A、B +0.094，组外之间 +0.098。
  - 静息 30 s：所有类别都从 0.10 一起降到 0.069（钙规则自身的双稳态漂移），没有噪声增强。
- **300 次变体复现步骤 1**（比值为各组范围；补全为平均值与为正组数）：
  - 新规则，静息也学：1.42（1.20–1.62），+0.29、5/8，静息点燃 0%；
  - 新规则，只在刺激期学：1.91（1.61–2.26），+0.37、6/8，8%；
  - 原规则，只在刺激期学，组内加密：3.9（3.4–4.5），+10.0、8/8，61%；
  - **新规则，只在刺激期学，组内加密：8.8（8.3–9.1），+6.5、8/8，57%**。
  - "组内加密"指组内连接概率 0.25、组间 0.083，入度不变。8 组均匀连接时，每个神经元只有 8 个组内输入、64 个组间输入，组内驱动只占递归输入约 20%。

### 0.3 设计改动

1. **规则**用 `GB_RATE`，作为可选预设加入 L3 与 L4，默认值不变。
2. **只在刺激期学习**：兴奋→兴奋递归只在刺激窗口（变体的 200 ms、序列的每一项）里学；静息期冻结，并把同时检测钙 `z` 清零。抑制性可塑性全程照常。
3. **组内加密**：组内连接概率 `pIn` = 0.25，组间 `pOut` = 0.083。
4. **学习态递归增益 G_learn**：步骤 1 之后标定（第一版没有这一项）。增益通过同时缩放可塑上下限 `sMin`、`sMax` 实现，学习改写效能时增益才能保住。
5. **演化态增益网格扩展到 1 以下**：G ∈ {0.6, 0.7, 0.8, 0.9, 1.0, 1.2}。


## 2. 规则变体的实现

L3（`calcium-rule.mjs`）：

```js
/** 频率依赖变体：同时检测钙再乘突触后慢速放电迹 y2（时间常数 post2Tau，取本次脉冲之前的值，相当于三联体 STDP 的 o2 项）；
 * 突触前后钙降到 0.6，单次巧合的钙和（1.2）落在减弱区间，只有突触后持续高频放电时才越过增强阈值 */
export const GB_RATE = { ...GB_NMDA, Cpre: 0.6, Cpost: 0.6, eta: 1.2, post2Tau: 100 };
// CalciumPlasticity：构造时 this.y2 = 0; this.postFired = false;
// postSpike()：const f = this.p.post2Tau ? this.y2 : 1; this.c += this.p.Cpost + this.p.eta * this.p.Cpre * this.nmda * f; this.postFired = true;
// step() 末尾（在 c 与 nmda 衰减之后）：if (this.p.post2Tau) this.y2 = this.y2 * Math.exp(-dt / this.p.post2Tau) + (this.postFired ? 1 : 0); this.postFired = false;
```

L4（`spiking-network.mjs`，NMDA 同时检测那一段）：

```js
const y2 = gb.post2Tau ? (this.y2 ??= new Float64Array(N)) : null;
for (const g of this.plastic) {
  if (g.frozen) continue;
  const tr = g.nm ?? this.nmda, eta = g.nm ? (gb.eta * gb.tauN) / g.plastic.tauN : gb.eta;
  for (let i = 0; i < N; i++) {
    if (!fired[i]) continue;
    const f = y2 ? y2[i] : 1;
    if (f === 0) continue;
    for (const k of g.incP[i]) g.z[k] += eta * gb.Cpre * tr[g.pre[k]] * f;
  }
}
if (y2) { const d2 = Math.exp(-dt / gb.post2Tau); for (let i = 0; i < N; i++) { y2[i] *= d2; if (fired[i]) y2[i] += 1; } }
```

两边的顺序必须一致：先用本步之前的 y2 计算同时检测钙，再"先衰减、后加 1"，否则逐拍对拍会失败。不设 `post2Tau` 时 f = 1，结果与原来逐位相同。

## 3. 步骤与验收条件

```mermaid
flowchart LR
  r0["R0 准备与归档"] --> r1["R1 L3 配对"] --> r2["R2 L4 对拍"] --> r3["R3 四类与方向"] --> s1["S1 吸引子"] --> s1b["S1b 学习态增益"] --> s2["S2 有向联系"] --> s3["S3 演化态标定"] --> s4["S4 演化链"] --> s5["S5 对照"] --> s6["S6 验证"] --> s7["S7 收尾"]
```

### R0：准备与归档

- 按第 1 节建工作区。
- **归档诊断**：把临时目录里的 `diag.mjs` 复制为 `experiments/chain-diag.mjs`，`spiking-network-rg.mjs` 复制为 `experiments/spiking-network-rg.mjs`；全部 20 个 `diag-*.json` 复制到 `experiments/logs/2026-10-09-chain-diagnostics/`。
  - 复制时把 `diag-classes-rg-e0-t100.json` 改名为 `diag-classes-gbnmda-eta0.json`。
  - 另写一份 `README.md`：说明每个文件对应的条件，并注明脚本用绝对路径导入，只作存档。
  - 如果临时目录已经不存在，就跳过归档，并在记录里注明。
- **初始记录**：写 `docs/stages/2-spiking-network/attempts/attractor-chain-v2.md`，状态"进行中"，抄入本计划第 0.2、0.3 节和全部验收条件。
- 归档和记录各自单独提交。
- 验收：
  - `git worktree list` 中有 `e2-attractor-chain-v2 [exp/2/attractor-chain-v2]`；
  - `git ls-files experiments/logs/2026-10-09-chain-diagnostics` 共 21 个文件（如果做了归档）；
  - `git status --short` 为空。

### R1：L3 规则变体

按第 2 节实现，并在 `verify/L3-hebbian.mjs` 中加入检查。验收：
1. `node verify/L3-hebbian.mjs` 退出码为 0，原有检查全部通过。
2. `GB_RATE` 的 60 次配对与下列参考值一致，误差 ≤ 0.005：
   - Δt = +10 ms：1 Hz −0.074、5 Hz −0.113、10 Hz −0.122、20 Hz +0.040、40 Hz +0.112；
   - 20 Hz、Δt = −10 ms：−0.119。
3. `GB_NMDA` 不变：1 Hz、+10 ms 仍为 +0.053（误差 ≤ 0.001）。

### R2：L4 快速网络与逐拍对拍

验收：
1. `node verify/L4-emergence.mjs` 退出码为 0，其中 `GB_RATE` 的对拍（放电时刻、每个突触的效能、ATP）逐拍相同。
2. `$env:FAST=1; npm run verify` 退出码为 0。
3. `git diff --stat stage/2-spiking-network -- src` 只列出 `calcium-rule.mjs` 和 `spiking-network.mjs`。

### R3：网络层的规则验收

- **实现**：以 `experiments/chain-diag.mjs` 的 `classes`、`direction`、`rest` 三种模式为蓝本，写 `demo/rate-rule.mjs`：导入改为仓库内的相对路径，规则用 src 里的预设，网络为第一版配置（`pIn` = `pOut` = 0.1，背景 30 / 12 Hz），种子不变。
- **运行**：`GB_RATE`、`GB_NMDA`、`GB_NMDA` 加 `eta: 0` 三种规则都跑一遍，结果写入 `figures/rate-rule-results.json`。
- **验收（GB_RATE）**：
  1. 四类（只刺激 A 20 次，只在刺激期学，开预算）：A→A ≥ +0.15；A→组外 ≤ A→A 的 10%；组外→A ≤ A→A 的 20%；组外→组外 ≤ +0.01。
  2. 方向（A 40 ms 接 B 40 ms，休息 400 ms，40 遍，全程学习）：A→B − B→A ≥ 0.03；A、B 到组外、组外到 A、B、组外之间，平均 Δρ 都 ≤ +0.01。
  3. 静息 30 s（无刺激，全程学习）：各类平均 Δρ ≤ +0.005，并且组内与组间之差 ≤ 0.005。
  4. 对照只记录、不作门槛：原规则四类中 A→组外 ≥ A→A 的 50%；原规则静息 5 s 后平均 ρ ≥ 0.18。
- `node verify/L4-rate-rule.mjs` 退出码为 0。

### S1：吸引子（v2 配置，学习态 G = 1）

- **配置**：规则 `GB_RATE`（替换 `buildChain` 里写死的 `GB_NMDA`）；`pIn` 0.25、`pOut` 0.083；按 0.3 第 2 条只在刺激期学习。
- **新增 `setGain(S, G)`**：先 `S.EE.plastic.sMin = cfg.sMin * G`，`S.EE.plastic.sMax = cfg.sMax * G`，再用 ρ 重算全部 `scale`。用它代替原来的 `EE.scale[k] *= G`。
- **训练**：300 次变体，与第一版相同，缓存为 `cache/chain-v2-attractors.json`。
- 验收：
  1. 训练前，兴奋性 2–5 Hz、抑制性 5–20 Hz、Fano 因子 < 2。背景只允许在 20–32 / 8–16 Hz 内调整。
  2. 每一组的组内 / 组间效能比 ≥ 2（诊断值 8.3–9.1）。
  3. 补全 ≥ +0.5 Hz，且至少 7/8 组为正（诊断值 +6.5、8/8）。
  4. 记录 G = 1 时学习态的静息点燃率（诊断值 57%），这一项不作门槛，交给 S1b 处理。

### S1b：学习态增益标定（不重训）

- 载入 `chain-v2-attractors.json`，对 G_learn ∈ {1.0, 0.9, 0.8, 0.7, 0.6, 0.5} 各测 3 s 静息点燃率（冻结学习）和补全（与 S1 同方法）。
- 验收：存在这样的 G_learn——静息点燃率 ≤ 5%，同时补全 ≥ +0.5 Hz、≥ 7/8 组为正；取其中最大的一个。完整表格写入结果 JSON。

### S2：有向联系（学习态，G_learn）

- 与第一版步骤 2 相同（160 回合，每 2 回合插 1 次变体），但改为在 G_learn 下进行，兴奋→兴奋只在每一项与变体刺激期间学习。缓存为 `cache/chain-v2-main.json`。
- 验收：第一版步骤 2 的第 1–6 条，其中第 5 条在 G_learn 下测量；补救方法也照第一版（先加到 320 回合，再把每项缩短到 30 ms）。

### S3：演化态标定

- 网格为 G ∈ {0.6, 0.7, 0.8, 0.9, 1.0, 1.2} × b ∈ {20, 40, 60, 80} pA，增益仍用 `setGain`。
- 验收：第一版步骤 3 的四条工作点条件。组内加密的备选方案已经用掉，没有工作点就按部分成功收尾。

### S4：演化链主测试

第一版步骤 4 的五条，原样适用。

### S5：对照

- 第一版的 5a–5d 原样适用，其中 5a 用 `chain-v2-attractors.json`，5b 的缓存为 `chain-v2-shuffled.json`。
- **5e 原规则对照**（只记录、不作门槛）：在 v2 的全部设置下（组内加密、只在刺激期学、S1b 选出的 G_learn）用 `GB_NMDA` 重做 S1、S2。记录组内 / 组间比、第一版 2.3 的 Spearman 相关，以及没经历过的有序组对的平均 Δρ，并与 `GB_RATE` 对比。

### S6：验证与回归

- `verify/L4-chain.mjs` 覆盖 S1–S5 的全部检查；`verify/all.mjs` 中有 `L4g` 和 `L4h`，FAST 模式都跳过。
- 验收：
  1. `node verify/L3-hebbian.mjs`、`node verify/L4-emergence.mjs`、`node verify/L4-rate-rule.mjs`、`node verify/L4-chain.mjs` 退出码都为 0。部分成功时，`L4-chain` 要逐条列出哪些未通过。
  2. 因为共享代码改了，完整的 `npm run verify` 退出码要为 0（约 50 分钟）。

### S7：图、文档与收尾

- **图和数据**：
  - `figures/fig13-chain.png` 与 `figures/chain-results.json` 改为 v2 的结果；
  - 保留 `figures/rate-rule-results.json`；
  - 在 `docs/CACHE.md` 中加入 `chain-v2-*` 这一族文件。
- **文档**：
  - 尝试记录补上结果与结论，单独提交；
  - 第 2 段 STATUS 更新现状与尝试列表；
  - 成功时，在 README.zh-CN.md 第 3 层之后新增"频率依赖变体 GB_RATE"一节（配对参考值与原因），在第 4 层（脉冲版）之后新增"有向通道演化链"一节。
- **第 4 段的备注**：在 `s4-error` 工作区，给第 4 段 STATUS 的"下一步"补一句：第 2 段发现原规则在有自发活动的网络里不加选择地增强，深层 v2 的非特异线索和 handover 的反向学习值得用 `GB_RATE` 复查。单独提交并推送。
- **成功**（S1–S4 与 S6 全部通过）：在 `s2-network` 工作区执行 `merge --no-ff exp/2/attractor-chain-v2`，推送 stage 分支，删除 e2 工作区和 exp 分支。
- **失败或部分成功**：打附注标签 `attempt/2/attractor-chain-v2` 并推送；把记录提交挑拣进 `stage/2-spiking-network` 并推送；删除 e2 工作区和 exp 分支。
- 验收：远程分支或标签与本地一致；`git worktree list` 中不再有 e2 工作区。

## 4. 停止条件

遇到以下情况就停下并报告，不要自行改变设计：
- R1、R2、R3 任一不达标：不进入演化链；
- S1 不达标，或 S1b 找不到 G_learn；
- S2、S3 用完补救后仍不达标：按部分成功收尾；
- 需要改动第 1 节以外的 src 文件。

## 5. 预计耗时

总计约 6–8 小时，大部分是模拟：
- R1–R2 约 1 小时（含 `L4-emergence` 4 分钟）；
- R3 约 30 分钟，九次网络测试可以并行；
- S1 约 25 分钟，S1b 约 10 分钟，S2 约 25–50 分钟；
- S3 约 1 小时，可以按 G 值并行；
- S4 约 20 分钟；
- S5 约 2 小时，5b、5e 需要重训；
- S6 的完整验证约 50 分钟。
## 实测结果

待执行。

## 结论与遗留问题

待实测。
