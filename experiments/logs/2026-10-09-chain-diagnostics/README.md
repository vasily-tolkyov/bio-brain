# 2026-10-09 有向通道诊断原始存档

这些单一种子的结果来自临时诊断目录，供追溯 v2 设计依据。它们不替代 v2 正式验收。

脚本位于 experiments/chain-diag.mjs 和 experiments/spiking-network-rg.mjs，保留原始绝对路径导入，只作存档；正式可复现实现使用 demo/rate-rule.mjs 的仓库内相对导入。

实际源目录含 21 个 JSON。主目录归档计划对应的 20 个 JSON 加本 README（共 21 个文件）；额外的 Cpre=0.55、Cpost=0.6、eta=0.3 探索结果保留在相邻目录 2026-10-09-chain-diagnostics-extra。全部 21 个 JSON 内容均原样复制。

原名 diag-classes-rg-e0-t100.json 归档为 diag-classes-gbnmda-eta0.json。它使用 Cpre=0.8、Cpost=1.0、eta=0，不能误记为突触前后钙各 0.6。

模式：rest 为仅背景、连续学习 30 s；classes 为重复刺激 A 20 次，200 ms 刺激期开兴奋学习、300 ms 静息期冻结，预算开关各一次；direction 为 A 40 ms 接 B 40 ms、休息 400 ms，40 遍且全程学习；gated/ungated 为 300 次含噪变体，分别只在刺激期/全程开兴奋学习。所有模式中抑制性可塑性在学习运行期间照常。

文件中的 plasticity 和 cfg 字段是参数依据，文件名仅作检索。初期结果未记录的参数在下表留空，其对应原始规则 GB_NMDA 和脚本初始默认网络。

| 文件 | 模式 | Cpre / Cpost | eta | post2Tau（ms） | pIn / pOut |
| --- | --- | --- | --- | --- | --- |
| diag-classes-gbnmda-eta0.json | classes | 0.8 / 1 | 0 | 100 | 0.1 / 0.1 |
| diag-classes-rg-e0.2-t100.json | classes | 0.8 / 1 | 0.2 | 100 | 0.1 / 0.1 |
| diag-classes-rg-e0.3-t100-cpre0.55-cpost0.6.json | classes | 0.55 / 0.6 | 0.3 | 100 | 0.1 / 0.1 |
| diag-classes-rg-e0.3-t100-cpre0.6-cpost0.6.json | classes | 0.6 / 0.6 | 0.3 | 100 | 0.1 / 0.1 |
| diag-classes-rg-e0.35-t100.json | classes | 0.8 / 1 | 0.35 | 100 | 0.1 / 0.1 |
| diag-classes-rg-e0.5-t100.json | classes | 0.8 / 1 | 0.5 | 100 | 0.1 / 0.1 |
| diag-classes-rg-e0.6-t100-cpre0.6-cpost0.6.json | classes | 0.6 / 0.6 | 0.6 | 100 | 0.1 / 0.1 |
| diag-classes-rg-e0.8-t100.json | classes | 0.8 / 1 | 0.8 | 100 | 0.1 / 0.1 |
| diag-classes-rg-e0.9-t100-cpre0.6-cpost0.6.json | classes | 0.6 / 0.6 | 0.9 | 100 | 0.1 / 0.1 |
| diag-classes-rg-e1.2-t100-cpre0.6-cpost0.6.json | classes | 0.6 / 0.6 | 1.2 | 100 | 0.1 / 0.1 |
| diag-classes-rg-e1.5-t100-cpre0.6-cpost0.6.json | classes | 0.6 / 0.6 | 1.5 | 100 | 0.1 / 0.1 |
| diag-classes.json | classes |  /  |  |  | 0.1 / 0.1 |
| diag-direction-base.json | direction | 0.8 / 1 | 2 |  | 0.1 / 0.1 |
| diag-direction-rg-e1.2-t100-cpre0.6-cpost0.6.json | direction | 0.6 / 0.6 | 1.2 | 100 | 0.1 / 0.1 |
| diag-gated-base-pin0.25-pout0.083.json | gated | 0.8 / 1 | 2 |  | 0.25 / 0.083 |
| diag-gated-rg-e1.2-t100-cpre0.6-cpost0.6-pin0.25-pout0.083.json | gated | 0.6 / 0.6 | 1.2 | 100 | 0.25 / 0.083 |
| diag-gated-rg-e1.2-t100-cpre0.6-cpost0.6.json | gated | 0.6 / 0.6 | 1.2 | 100 | 0.1 / 0.1 |
| diag-gated.json | gated |  /  |  |  | 0.1 / 0.1 |
| diag-rest-rg-e1.2-t100-cpre0.6-cpost0.6.json | rest | 0.6 / 0.6 | 1.2 | 100 | 0.1 / 0.1 |
| diag-rest.json | rest |  /  |  |  | 0.1 / 0.1 |
| diag-ungated-rg-e1.2-t100-cpre0.6-cpost0.6.json | ungated | 0.6 / 0.6 | 1.2 | 100 | 0.1 / 0.1 |
