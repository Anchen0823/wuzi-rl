"""对战评估（设计见 docs/DESIGN.md §10）：网络 vs 网络 / vs 基线，自动换边。"""

from __future__ import annotations

import random

import numpy as np

from gomoku.board import Board, BLACK, WHITE, DRAW
from .mcts import puct_best_move


def play_net_vs_net(
    net_a, net_b, sims: int, device=None, rng: random.Random | None = None,
    c_puct: float = 1.5,
) -> int:
    """net_a 执黑、net_b 执白一局，返回胜者（BLACK/WHITE/DRAW）。"""
    rng = rng or random.Random()
    b = Board()
    while not b.is_over:
        net = net_a if b.to_play == BLACK else net_b
        mv = puct_best_move(b, net, sims, c_puct=c_puct, device=device, rng=rng)
        b.apply(mv)
    return b.winner


def evaluate_nets(net_a, net_b, games: int, sims: int, device=None, seed: int = 0,
                  progress=None) -> dict:
    """net_a vs net_b 共 games 局（自动换边），返回 {'a': 胜场, 'b': 胜场, 'draw': 和棋}。

    progress: 可选回调 (done, total)，每局结束调用（进度显示）。
    """
    rng = random.Random(seed)
    stats = {"a": 0, "b": 0, "draw": 0}
    for i in range(games):
        if i % 2 == 0:
            w = play_net_vs_net(net_a, net_b, sims, device, rng)
        else:
            w = play_net_vs_net(net_b, net_a, sims, device, rng)
            w = DRAW if w == DRAW else (WHITE if w == BLACK else BLACK)
        stats["a" if w == BLACK else ("b" if w == WHITE else "draw")] += 1
        if progress:
            progress(i + 1, games)
    return stats


def evaluate_vs_player(
    net, player_factory, games: int, sims: int, device=None, seed: int = 0,
    progress=None,
) -> dict:
    """网络 vs 基线引擎（player_factory 无参构造），返回 {'net': 胜场, 'player': 胜场, 'draw'}。

    progress: 可选回调 (done, total)，每局结束调用。
    """
    rng = random.Random(seed)
    stats = {"net": 0, "player": 0, "draw": 0}
    for i in range(games):
        player = player_factory()
        if i % 2 == 0:  # 网络执黑
            b = Board()
            while not b.is_over:
                mv = (puct_best_move(b, net, sims, device=device, rng=rng)
                      if b.to_play == BLACK else player.move(b))
                b.apply(mv)
            w = b.winner
        else:           # 网络执白
            b = Board()
            while not b.is_over:
                mv = (player.move(b)
                      if b.to_play == BLACK else puct_best_move(b, net, sims, device=device, rng=rng))
                b.apply(mv)
            w = b.winner
            w = DRAW if w == DRAW else (WHITE if w == BLACK else BLACK)
        stats["net" if w == BLACK else ("player" if w == WHITE else "draw")] += 1
        if progress:
            progress(i + 1, games)
    return stats
