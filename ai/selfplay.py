"""自对弈数据生成（设计见 docs/DESIGN.md §8）。

每局用「当前网络 + PUCT 搜索」双方对弈，记录每步 (状态, π, z)；
终局值 z 按每步「轮到者视角」归一化（胜 +1 / 负 -1 / 和 0）。
"""

from __future__ import annotations

import random

import numpy as np

from gomoku.board import Board, BLACK, WHITE, DRAW
from .mcts import puct_search, puct_visit_distribution

# 对称映射缓存（按棋盘大小）
_MAPS_CACHE: dict[int, list[np.ndarray]] = {}


def _symmetry_maps(size: int) -> list[np.ndarray]:
    if size not in _MAPS_CACHE:
        _MAPS_CACHE[size] = Board.symmetry_maps(size)
    return _MAPS_CACHE[size]


def play_game(
    net,
    sims: int,
    c_puct: float = 1.5,
    temp_steps: int = 15,
    temp: float = 1.0,
    dirichlet_alpha: float = 0.3,
    dirichlet_eps: float = 0.25,
    batch_size: int = 64,
    device=None,
    rng: random.Random | None = None,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """一局自对弈。返回 [(state(4,15,15), pi(225), z)]，z 为每步视角。"""
    rng = rng or random.Random()
    b = Board()
    samples: list[tuple[np.ndarray, np.ndarray, int]] = []
    move_no = 0
    while not b.is_over:
        root = puct_search(
            b, net, sims, c_puct=c_puct,
            dirichlet_alpha=dirichlet_alpha, dirichlet_eps=dirichlet_eps,
            add_root_noise=True, batch_size=batch_size, device=device, rng=rng,
        )
        tau = temp if move_no < temp_steps else 0.0
        pi = puct_visit_distribution(root, tau, b.size)
        samples.append((b.encode(), pi, b.to_play))
        if tau > 0:
            np_rng = np.random.default_rng(rng.randrange(1 << 30))
            mv = int(np_rng.choice(b.size * b.size, p=pi))
        else:
            mv = int(np.argmax(pi))
        b.apply(mv)
        move_no += 1

    winner = b.winner
    out: list[tuple[np.ndarray, np.ndarray, float]] = []
    for s, pi, player in samples:
        if winner == DRAW:
            z = 0.0
        else:
            z = 1.0 if winner == player else -1.0
        out.append((s, pi, z))
    return out


def augment_samples(samples) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """D4 对称增强：每条样本 → 8 变体（状态/策略同步变换，z 不变）。"""
    size = 15
    maps = _symmetry_maps(size)
    out = []
    for s, pi, z in samples:
        flat = s.reshape(4, -1)
        for m in maps:
            s2 = flat[:, m].reshape(4, size, size)
            p2 = pi.reshape(-1)[m]
            out.append((s2, p2, z))
    return out
