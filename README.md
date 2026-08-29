# wuzi-rl · 五子棋 + 自研强化学习 AI

用 **pygame** 实现的 15×15 五子棋，配套一个**完全自研**的强化学习 AI（AlphaZero 风格：MCTS + 自对弈 + 残差网络，从零开始训练）。

> **自研承诺**：本项目不使用任何外部 AI 模型、预训练权重或开源 AI 组件（不参考、不复制他人棋类 AI、MCTS、强化学习实现）。PyTorch 仅作为 GPU 张量计算与自动微分引擎；规则引擎、网络结构、MCTS/PUCT、自对弈管线、训练循环、评估系统全部自研。边界细则见 [docs/PLAN.md](docs/PLAN.md) §1.2。

## 当前状态

**M3 —— 完成**（自研神经网络 + 自对弈训练管线闭环；53 项测试全绿）

- [x] M0 计划书与系统设计、仓库初始化
- [x] M1 环境搭建（`.venv` + pygame-ce/NumPy/PyTorch 2.13.0+cu126，CUDA 冒烟通过）
- [x] M1 规则引擎 `gomoku/board.py` + 对局管理 `gomoku/game.py`（含单测）
- [x] M1 pygame 界面 `gomoku/gui.py`（人 vs 人，含冒烟测试）
- [x] M1 依赖白名单审计 `tools/dep_audit.py`
- [x] M2 纯 MCTS 基线 `ai/mcts.py` + 玩家引擎 `ai/players.py`（13 项单测）
- [x] M2 GUI 模式接入（主菜单 1-4：人vs人 / 人执黑 / 人执白 / AI 观战）+ `tools/eval.py`
- [x] M3 自研残差网络 `ai/net.py`（7.26M 参数，手写卷积/BN/全连接层）
- [x] M3 网络化 PUCT 搜索（批量叶子评估 + 虚拟损失 + Dirichlet 噪声）
- [x] M3 训练管线：`ai/selfplay.py` + `ai/replay.py` + `ai/train.py` + `ai/arena.py` + `tools/train.py`
- [ ] M4 强化训练 + AI 接入（难度档、人机对战）
- [ ] M5 打磨、测试、打包

## 文档

| 文档 | 内容 |
|---|---|
| [docs/PLAN.md](docs/PLAN.md) | 计划书：目标、自研边界、里程碑、训练方案、验收标准、风险 |
| [docs/DESIGN.md](docs/DESIGN.md) | 系统设计：规则引擎、界面、状态编码、网络结构、MCTS、训练管线、性能预算 |

## 环境

- Python 3.14+（个别 wheel 不兼容时回退 3.12/3.13 venv）
- NVIDIA RTX 4060（8GB）+ CUDA；pygame-ce（import 名 pygame）+ NumPy + PyTorch（CUDA 版）
- 依赖：`pip install -r requirements.txt`（PyTorch 安装命令以官方 Get Started 页面为准）

## 快速开始

```powershell
# 1. 创建虚拟环境并安装依赖（PyTorch 用 CUDA 版 wheel）
python -m venv .venv
.\.venv\Scripts\python -m pip install pygame-ce numpy pytest
.\.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
# 国内网络慢可改用 PyPI 镜像（阿里云），其 torch 同样自带 CUDA 运行时：
#   $env:PIP_INDEX_URL = "https://mirrors.aliyun.com/pypi/simple/"
#
# 注意：若国内镜像缺 Python 3.14 的 CUDA wheel 且官方源直连过慢，
# 可对官方 wheel 做并行分段下载（curl -r Range + 合并），
# 本项目 M1 实测：单连接约 0.2 MB/s，12 路并行约 2 MB/s。

# 2. 运行测试
.\.venv\Scripts\python -m pytest tests -q

# 3. 启动游戏（人 vs 人；R 重开 / U 悔棋）
.\.venv\Scripts\python main.py
```

## 自研边界审计

```powershell
python tools\dep_audit.py            # 检查源码 import 与权重文件
python tools\dep_audit.py --check-installed   # 连同已装 pip 包一起审计
```

## 训练 AI（M3 起可用）

```powershell
# 冒烟（小规模跑通全流程）
.\.venv\Scripts\python tools\train.py --iters 2 --games-per-iter 2 --sims-train 40 `
    --arena-games 2 --sims-eval 40 --n-blocks 2 --n-filters 64

# 正式训练（默认配置：500 局/迭代、sims 200/800、评估门 200 局；4060 挂机）
.\.venv\Scripts\python tools\train.py --iters 50

# 中断后续训
.\.venv\Scripts\python tools\train.py --iters 50 --checkpoint checkpoints\net.pt
```

产物：`checkpoints/net.pt`（模型+优化器+迭代元数据）、`runs/history.csv`（损失/采纳/耗时曲线）。

## 许可证

[MIT License](LICENSE) © 2026 Anchen0823。

模型权重（checkpoint，不入库）为独立训练资产，如后续对外分发将另行声明条款。
