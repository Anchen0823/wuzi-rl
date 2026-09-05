# wuzi-rl · 五子棋 + 自研强化学习 AI

用 **pygame** 实现的 15×15 五子棋，配套一个**完全自研**的强化学习 AI（AlphaZero 风格：MCTS + 自对弈 + 残差网络，从零开始训练）。

> **自研承诺**：本项目不使用任何外部 AI 模型、预训练权重或开源 AI 组件（不参考、不复制他人棋类 AI、MCTS、强化学习实现）。PyTorch 仅作为 GPU 张量计算与自动微分引擎；规则引擎、网络结构、MCTS/PUCT、自对弈管线、训练循环、评估系统全部自研。边界细则见 [docs/PLAN.md](docs/PLAN.md) §1.2。

## 当前状态

**M4/M5 主体完成**（网络 AI 接入 GUI；23 迭代训练；62 项测试全绿）

- [x] M0 计划书与系统设计、仓库初始化
- [x] M1 环境搭建 + 规则引擎 + 界面 + 依赖审计
- [x] M2 纯 MCTS 基线 + GUI 模式接入 + 对战评估
- [x] M3 自研残差网络 + 网络化 PUCT + 自对弈训练管线
- [x] M4 网络引擎接入 GUI（主菜单 1-6 难度档，checkpoint 自动加载/回退）
- [x] M4 训练 23 迭代 / ~2300+ 局自对弈：**loss 6.59 → 2.28**，评估门采纳 8 次
- [x] M4 棋力验收（net@400）：vs 随机 **100%** ✅ / vs 纯 MCTS **100%** ✅ / vs 贪心 **0%** ❌（需数万局量级长训）
- [x] M5 对局回放（主菜单 7）+ 训练进度输出 + 多进程并行自对弈 v2
- [ ] 持续长训（目标 vs 贪心 ≥90%）+ M5 人工验收/打包

### 模型棋力现状（2026-08-31，迭代 23）

训练 23 迭代（约 2300+ 局自对弈，RTX 4060 上约 20 小时）：

```
loss 曲线：6.59 → 2.28（持续下降，未收敛）
vs 随机：100%（10:0）    ✅
vs 纯 MCTS（限时）：100%（6:0）  ✅
vs 贪心启发式：0%（0:10） ❌  ← 目标 ≥90%，需继续训练
```

上述历史训练记录显示 loss 下降及小样本基线对局结果，尚不足以证明收敛或人类棋力等级；对贪心启发式的目标仍未达到。继续训练路径：`tools/train.py --iters N --checkpoint checkpoints/net.pt`（详见 PLAN §12）。训练产物未入库，克隆仓库不会自动获得这份历史模型。

### 首次克隆后的 AI 行为

GUI 优先加载 `checkpoints/best.pt`，缺失时尝试 `checkpoints/net.pt`；两者都不存在时，网络难度档会回退为纯 MCTS，并显示提示。人机可玩与已加载训练模型是两个不同状态。

GUI 根据 PyTorch 的 CUDA 可用性选择 GPU 或 CPU。没有 NVIDIA GPU 也有 CPU 执行路径，但训练和搜索速度取决于硬件；后面的 RTX 4060 配置是原开发环境记录。

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
git clone https://github.com/Anchen0823/wuzi-rl.git
cd wuzi-rl
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

# 3. 启动游戏（主菜单 1-6：人vs人 / 人vsAI 低-中-高 / AI观战）
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

## 代码导航

| 目录 | 内容 |
| --- | --- |
| [gomoku](gomoku/) | 棋盘规则、对局状态和 pygame 界面 |
| [ai](ai/) | MCTS、策略价值网络、自对弈、训练和评估 |
| [tools](tools/) | 训练、评估、记录汇总和依赖审计入口 |
| [tests](tests/) | 规则、搜索、训练管线与 GUI 测试 |

README 中的训练时长、胜率和测试数量是对应日期的历史记录；修改代码或重新训练后需要重新验证。

## 许可证

[MIT License](LICENSE) © 2026 Anchen0823。

模型权重（checkpoint，不入库）为独立训练资产，如后续对外分发将另行声明条款。
