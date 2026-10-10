# 有向通道演化链 v3 运行记录

本轮从 `attempt/2/attractor-chain-v2` 开始，CONFIG 保持版本 2，src 无改动。目录日期按 America/Los_Angeles 当地日期记录，进程元数据使用 UTC。

- `cache-before.json`：正式模拟前的 74 个缓存，文件名、大小与 SHA-256；独立提交。
- `cache-integrity.json`：结束验证时对这 74 个输入的逐个哈希核对。
- `d1-g*.log`：六个并行增益进程，各测四个 b，总计 24 点；每点十次 F 终点线索与 3 s 静息，种子 6001。
- `l1-seq-*.log`：四个候选分别从同一份 v2 吸引子开始，G_learn=0.6，160×40 ms，种子 4001–4004。
- `l1-rule-*.log`：四个候选的四类测量（预算开启与关闭均记录）及 30 s 静息；网络为第一版均匀配置。
- `initial-jobs.json`：各进程的参数、PID、开始/结束时间和退出码。14 个初始模拟进程，逻辑 CPU=16，工作进程上限=15；每个进程只写独立缓存。
- `dose-decision-checks.log`：缩放实现、最小通过剂量、非单调剂量和 A/B/C 决策分支的实现检查。这些不是 D2 的 40 次正式试次，不能代替动力学测量。
- `fast-verify.log`、`fast-verification.json`：FAST 回归全部通过；脚本用时 284 s，进程墙钟 291.623 s，退出码 0。
- `chain-verify.log`：v3 的执行完整性、科学验收和全部失败/未运行项；源代码范围、CONFIG 与旧缓存核对也列入检查。
- `cache-snapshot.json`：最终新增的 chain-v3-* 本机产物清单；缓存含墙钟耗时，重训时哈希可不同。

背景 rho 的计算排除七条正向及其反向，取剩余 42 个有序组对的组平均。v2 320×40 ms 的实测背景为 0.0306504223，与计划中约 0.031 一致。转到未经历组的比例包含反向，反向同时单列，两项统计不互斥。

正式结果以 figures/chain-results.json 的 v3 节、figures/rate-rule-v3.json 及尝试记录为准。未执行阶段保留“未运行”和 null，不以零值替代。入库日志统一换行、去掉行尾空白，原始输出另保留在本机运行目录。
