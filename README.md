# bio-brain：逐层生物脑模型

从单个神经元一路搭到多层脉冲网络的预测编码与脑结构。每一层都有验证脚本，下一层只用上一层验证过的东西；每一层同时记代谢（ATP）与动力学两本能量账。

A layer-by-layer biological brain model, from single neurons, synapses and Hebbian learning up to multi-layer spiking networks with predictive coding. Every layer is verified by a script and keeps both a metabolic (ATP) and a dynamical energy ledger.

- 完整的技术说明：[README.zh-CN.md](README.zh-CN.md)
- 路线图、分支与标签约定：[docs/ROADMAP.md](docs/ROADMAP.md)

## 七段路线图

最终目标：用多层脉冲网络实现“经验 → 规则抽象 → 泛化预测 → 目标规划 → 行为 → 经验”的闭环。

1. 神经元、突触、赫布学习：完成。[状态](docs/stages/1-neuron-synapse-hebbian/STATUS.md)
2. 脉冲网络（吸引子与有向通道）：部分完成，分支 `stage/2-spiking-network`。[状态](docs/stages/2-spiking-network/STATUS.md)
3. 规则抽象：部分完成，分支 `stage/3-rule-abstraction`。[状态](docs/stages/3-rule-abstraction/STATUS.md)
4. 规则误差校对：部分完成，当前主攻，分支 `stage/4-error-correction`。[状态](docs/stages/4-error-correction/STATUS.md)
5. 目标规划：脉冲版未开始，分支 `stage/5-planning`。[状态](docs/stages/5-planning/STATUS.md)
6. 行动层：未开始，分支 `stage/6-action`。[状态](docs/stages/6-action/STATUS.md)
7. 通用感知层：未开始，分支 `stage/7-perception`。[状态](docs/stages/7-perception/STATUS.md)

贯穿各段：能量账（ATP 与动力学），见 README.zh-CN.md 开头。

## 运行

零依赖，Node ≥ 20。

- 全部验证：`npm run verify`（约 45 分钟；没有缓存时要先训练，更久）
- 快速验证：`FAST=1 npm run verify`（PowerShell：`$env:FAST=1; npm run verify`，约 2 分钟）
- 单独某一层或某个 demo：见 README.zh-CN.md 的“运行”一节

训练好的网络状态（`cache/`）不在仓库里：从 Releases 的 `cache-2026-10-06` 下载 `bio-brain-cache-2026-10-06.zip`，解压到仓库根目录下的 `cache/`。每个文件由哪个脚本、哪次提交产生，见 [docs/CACHE.md](docs/CACHE.md)。没有缓存时，脚本会自己训练。

## 目录

- `src/`：L1 神经元、L2 突触、L3 可塑性、L4 网络、L5 脑结构
- `demo/`：各图与实验；`verify/`：逐层验证；`figures/`：demo 生成的图
- `docs/`：路线图、各段状态、尝试记录、缓存清单
- `experiments/`：归档的实验脚本、日志与结果（存档，不保证能在当前代码上直接运行）

## 许可

本仓库没有附开源许可证：代码公开可见，但保留所有权利。如需复用，请先联系作者。
