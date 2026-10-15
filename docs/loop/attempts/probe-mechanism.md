# 机理探查：状态切换时网络内部发生了什么

日期：2026-10-13（America/Los_Angeles）。只读探查，不属于任何工作包；结论写入架构修订 `52227aa`、`4a3c223`。

## 做法

- 代码：标签 `attempt/loop/w4-diag3` 的源码（`git archive` 到临时目录），未修改任何源文件。
- 状态：主缓存 `loop-w4diag3-<id>-trained.json`（预测抵消训练之后），只读载入，未写缓存。
- 运行：s1-t25-e4 的闪电、雷声、鸟叫各一个进程，另加 s1-t40-e4 的闪电；学习全部冻结。
- 协议：3 s 静息；8 次 1 s 呈现（间隔 1.5 s）；2 次 5 s 呈现（间隔 1.5 s）。共 36 s 模拟，每个进程约 700 s 墙钟。
- 记录：相关各群的逐个脉冲；每 1 ms 的平均膜电位，以及 N1、M1、S1、V1、L 收到的兴奋、抑制与慢受体电导。

脚本：`probe.mjs`（运行与记录）、`analyze.mjs` 与 `phases.mjs`（分析）、`plot_mechanism.py`（作图）。输出：`analyze-output.txt`、`phases-output.txt`、`mechanism-switch.png`、`mechanism-volleys.png`。原始数据为 4 个 `probe-*.json`，每个约 17 MB，不进 Git，哈希见 `manifest.json`。

## 发现

### 1. 误差细胞的成团放电来自共用的小中继，不是 E1 在闪

- 每个 N1 细胞接收全部 30 个 S1 细胞（k_ref 50 > 30，p 截到 1），每个 M1 细胞接收本组全部 10 个 V1 细胞；突触释放是确定性的，同组误差细胞收到的抑制完全相同。
- 静息时 N1 每秒成团放电 2.2–3.7 次（20 个细胞中 ≥ 5 个在 5 ms 内放电），在场时 3.3–5.5 次。静息时每次成团放电之前，N1 收到的抑制在 30–40 ms 内从约 2.5 nS 降到 0.8 nS，E1 几乎不变；在场时抑制降约 40%，E1 也小幅上冲。
- E1 本身接近随机：80 个细胞的计数法诺因子 1.2–2.2，1 ms 自相关 0.13–0.19、2 ms 以后约为 0，功率谱以 80 Hz 以上为主，没有节律。
- 三个特征的 N1 在静息时相互同步（5 ms 分箱相关 0.09–0.19），因为 S1 不分组、随机连到所有 N1。

### 2. E1 的适应让状态信号不是干净的台阶，误差电路报告了反向的"第二波"

| 群 | 静息 | 出现后 0–50 | 50–150 | 150–300 | 300–1000 ms | 消失后 0–50 | 50–150 | 150–300 | 300–600 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1:闪电 | 2.9 | 25.7 | 18.1 | 12.3 | 10.8 | 0.7 | 0.2 | 1.0 | 2.5 |
| M1:闪电 | 3.9 | 0.6 | 12.9 | 14.3 | 7.4 | 22.9 | 14.4 | 4.7 | 2.9 |
| N1:闪电 | 5.3 | 36.2 | 1.9 | 4.6 | 6.4 | 0.4 | 1.1 | 6.5 | 6.3 |

（Hz；s1-t25-e4。雷声、鸟叫与 s1-t40-e4 的同类数字见 `phases-output.txt`。）

- 出现：E1 在 1–4 ms 内过半，前 50 ms 冲到 17–26 Hz，约 300 ms 内衰减到 7–11 Hz。N1 的抑制（S1，慢拷贝）20–80 ms 才涨起来，N1 在 0–50 ms 放一阵。随后 E1 衰减、V1 跟着下降，而 M1 的慢拷贝兴奋仍高，M1 在 50–300 ms 升到在场后期的 2–2.3 倍，L⁻ 零星误触发。
- 消失：E1 在 5 ms 内跌到 0.1–1 Hz，比静息还低，0.6–1 s 后才回到静息。V1 消失，M1 在 0–150 ms 放电（宽而平），N1 在 150–600 ms 回到或略高于静息。

### 3. 出现与消失不对称

在 E1 上做理想的台阶检测（之后 40 ms 对比之前 200 ms 的平均频率）：出现是状态内最大正波动的 1.9–2.2 倍；消失只有状态内最大负波动的 1.0–1.2 倍。原因是适应把在场平台压到 7–11 Hz，消失只下降 6–10 Hz。

### 4. L 的结构

10 个 L 细胞参数相同、输入相同（全连接 20 个误差细胞）、没有背景，突触为 AMPA + NMDA，等于一个对同步敏感的单细胞；计数几乎都是 10 的倍数。

## 本地归档与提取

本地附注标签：`attempt/loop/probe-mechanism`。归档目录：`experiments/logs/2026-10-13-loop-probe-mechanism/`，保留脚本、日志、图、摘要和 SHA-256 清单。清单中 17 个原始文件均已逐项核对。

原始数据目录：`D:/kimi_kairos/bio-brain-wt/probe-mechanism/`。四个 `probe-*.json` 留在该目录，不进入 Git，不删除或覆盖。标签内清单可验证其原始字节。

在主仓库执行以下命令，导出证据到新的目录：

```powershell
git archive --format=zip --output=D:/kimi_kairos/bio-brain-wt/probe-mechanism-evidence.zip attempt/loop/probe-mechanism experiments/logs/2026-10-13-loop-probe-mechanism
Expand-Archive -LiteralPath D:/kimi_kairos/bio-brain-wt/probe-mechanism-evidence.zip -DestinationPath D:/kimi_kairos/bio-brain-wt/probe-mechanism-extracted
```

本记录只收录探查结论，不把诊断源码合入集成分支。后续依据 `52227aa`、`4a3c223` 和 `C:/Users/86139/.cursor/plans/闭环架构第一版_W4边界检测重构_e5a93c17.plan.md` 执行 W4-diag4。