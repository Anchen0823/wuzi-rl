"""AI 层：完全自研的搜索与强化学习管线（不依赖任何第三方 AI 组件）。

- mcts.py     纯 MCTS 基线（UCB1 + rollout）与网络化 PUCT 搜索（批量叶子评估）
- net.py      自研残差网络（策略 + 价值双头，仅用 torch 张量原语）
- players.py  可复用对弈引擎：随机 / 贪心启发式 / 纯 MCTS
- selfplay.py 自对弈数据生成（D4 对称增强）
- replay.py   经验池（滑动窗口 + 采样时随机对称）
- train.py    训练循环、评估门、checkpoint、历史记录
- arena.py    对战评估（网络 vs 网络 / vs 基线，自动换边）
"""
