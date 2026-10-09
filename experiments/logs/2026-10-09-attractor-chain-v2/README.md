# 有向通道演化链 v2 运行日志

这些日志来自 v2 工作区，诊断依据另见相邻的 `2026-10-09-chain-diagnostics` 目录。

- `r1-l3.log`：原有 L3 检查及六个 GB_RATE 配对值。
- `r2-l4.log`：原有 L4 涌现检查与三种规则的逐拍对拍。
- `r2-fast.log`、`fast-integrated.log`：分别在规则实现后、L4g/L4h 接入后运行的 FAST 回归。
- `implementation-checks.log`：旧规则逐位一致、学习期间增益保持、兴奋冻结与抑制学习并存、包含 y2 的精确续算检查。
- `r3-*.log`：三种规则在四类突触、方向和静息条件下的九项网络测量及验收。
- `s1-baseline.log`：S1 训练前基线，未调整默认 30/12 Hz 背景。

正式模拟结果与验收以 `figures/chain-results.json`、`figures/rate-rule-results.json` 和尝试记录为准。长模拟另存续算缓存；缓存不进入 Git。
