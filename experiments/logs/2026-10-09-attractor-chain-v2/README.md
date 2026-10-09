# 有向通道演化链 v2 运行日志

这些日志来自 v2 工作区，诊断依据另见相邻的 `2026-10-09-chain-diagnostics` 目录。

- `r1-l3.log`：原有 L3 检查及六个 GB_RATE 配对值。
- `r2-l4.log`：原有 L4 涌现检查与三种规则的逐拍对拍。
- `r2-fast.log`、`fast-integrated.log`：分别在规则实现后、L4g/L4h 接入后运行的 FAST 回归。
- `implementation-checks.log`：旧规则逐位一致、学习期间增益保持、兴奋冻结与抑制学习并存、包含 y2 的精确续算检查。
- `r3-*.log`：三种规则在四类突触、方向和静息条件下的九项网络测量及验收。
- `s1-baseline.log`：S1 训练前基线，未调整默认 30/12 Hz 背景。
- `s1-attractors.log`、`s1b-*.log`：300 次变体训练和完整六点学习增益表。
- `s2-sequence.log`：初轮与两次补救，最后因平均方向差不足停止。
- `finalize.log`、`chain-verify.log`：停止结果与 L4g 逐项失败、未运行列表。
- `full-verify.log`、`full-verification.json`：修改共享代码后的完整回归，2684 s；原有十组与 L4h 通过，仅 L4g 失败，总退出码为 1。元数据保留实际开始、结束时间及进程墙钟耗时。
- `cache-snapshot.json`：本次新增的 15 个链缓存与九个规则缓存的文件名、大小和 SHA-256，用于核对已完成的本机产物。缓存包含墙钟耗时，重训时文件哈希可能不同。
- `cache-integrity.json`：完整回归后，原有 50 个缓存的前后 SHA-256 全部一致；当前合计 74 个缓存。

入库日志统一换行并去除行尾空白，数值与检查输出保持原样；原始完整回归输出另保留在本机运行目录。

正式模拟结果与验收以 `figures/chain-results.json`、`figures/rate-rule-results.json` 和尝试记录为准。长模拟另存续算缓存；缓存不进入 Git。
