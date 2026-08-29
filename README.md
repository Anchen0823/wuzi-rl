# wuzi-rl · 五子棋 + 自研强化学习 AI

用 **pygame** 实现的 15×15 五子棋，配套一个**完全自研**的强化学习 AI（AlphaZero 风格：MCTS + 自对弈 + 残差网络，从零开始训练）。

> **自研承诺**：本项目不使用任何外部 AI 模型、预训练权重或开源 AI 组件（不参考、不复制他人棋类 AI、MCTS、强化学习实现）。PyTorch 仅作为 GPU 张量计算与自动微分引擎；规则引擎、网络结构、MCTS/PUCT、自对弈管线、训练循环、评估系统全部自研。边界细则见 [docs/PLAN.md](docs/PLAN.md) §1.2。

## 当前状态

**M0 —— 规划完成**（游戏与 AI 尚未实现，按里程碑推进）

- [x] M0 计划书与系统设计、仓库初始化
- [ ] M1 环境搭建 + 规则引擎 + pygame 界面（人 vs 人）
- [ ] M2 纯 MCTS 基线 AI
- [ ] M3 神经网络 + 自对弈训练管线闭环
- [ ] M4 强化训练 + AI 接入（难度档、人机对战）
- [ ] M5 打磨、测试、打包

## 文档

| 文档 | 内容 |
|---|---|
| [docs/PLAN.md](docs/PLAN.md) | 计划书：目标、自研边界、里程碑、训练方案、验收标准、风险 |
| [docs/DESIGN.md](docs/DESIGN.md) | 系统设计：规则引擎、界面、状态编码、网络结构、MCTS、训练管线、性能预算 |

## 环境

- Python 3.14+（个别 wheel 不兼容时回退 3.12/3.13 venv）
- NVIDIA RTX 4060（8GB）+ CUDA；pygame + NumPy + PyTorch（CUDA 版）
- 依赖：`pip install -r requirements.txt`（PyTorch 安装命令以官方 Get Started 页面为准）

## 快速开始

> 待 M1 完成后提供（安装 → 启动游戏 → 人机对战）。

## 许可证

暂未指定——自研代码的许可证选择待定（公开仓库，建议尽早补充）。
