"""全局超参数（集中管理，设计见 docs/DESIGN.md §11）。

命令行可覆盖：后续工具入口统一接收 --config 或逐项覆盖。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # ---- 棋盘 ----
    board_size: int = 15

    # ---- 网络（M3 使用） ----
    n_res_blocks: int = 6
    n_filters: int = 256

    # ---- MCTS / PUCT（M2 纯 MCTS 基线 / M3 网络化搜索） ----
    c_puct: float = 1.5
    c_ucb: float = 1.4          # 纯 MCTS 基线 UCB1 系数
    mcts_sims_menu: int = 200   # GUI 低难度档模拟次数
    sims_train: int = 200
    sims_eval: int = 800
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25
    temp_steps: int = 15
    temp: float = 1.0

    # ---- 自对弈与训练（M3 使用） ----
    games_per_iter: int = 500
    buffer_size: int = 500
    batch_size: int = 512
    lr: float = 1e-3
    wd: float = 1e-4
    lambda_l2: float = 1e-4

    # ---- 评估门（M3 使用） ----
    arena_games: int = 200
    arena_threshold: float = 0.55
