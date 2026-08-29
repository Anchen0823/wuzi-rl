"""纯 MCTS 基线单元测试：树统计、即时胜/堵、合法对局、基线强度。"""

import random

import pytest

from gomoku.board import Board, BLACK, WHITE, ONGOING
from ai.mcts import mcts_search, best_move, would_win, adjacent_moves
from ai.players import RandomPlayer, GreedyPlayer, MCTSPlayer


def play_seq(b: Board, seq: list[tuple[int, int]]):
    for r, c in seq:
        b.apply(b.idx(r, c))


# ---------- 树统计一致性 ----------

def test_tree_statistics_conservation():
    b = Board()
    root = mcts_search(b, sims=60, rng=random.Random(7))
    assert root.n == 60
    stack = [root]
    while stack:
        node = stack.pop()
        s = sum(ch.n for ch in node.children.values())
        assert node.n >= s                       # 每个节点访问 ≥ 其子节点之和
        assert abs(node.q) <= 1.0 + 1e-9         # 价值有界
        stack.extend(node.children.values())


def test_ucb_explores_multiple_moves():
    b = Board()
    b.apply(b.idx(7, 7))
    b.apply(b.idx(7, 8))
    root = mcts_search(b, sims=80, rng=random.Random(2))
    assert len(root.children) >= 5               # 多次模拟应展开多个分支


# ---------- 工具函数 ----------

def test_would_win():
    b = Board()
    play_seq(b, [(7, 5), (0, 0), (7, 6), (0, 1), (7, 7), (0, 2), (7, 8)])
    assert would_win(b, b.idx(7, 9), BLACK)
    assert not would_win(b, b.idx(7, 9), WHITE)
    assert not would_win(b, b.idx(0, 3), BLACK)


def test_adjacent_moves_empty_board_is_all():
    b = Board()
    assert len(adjacent_moves(b)) == 225


def test_adjacent_moves_near_stones_only():
    b = Board()
    b.apply(b.idx(7, 7))
    moves = adjacent_moves(b)
    assert len(moves) == 8                       # 中心占位 + 8 邻域（中心已占不算）
    assert b.idx(7, 7) not in moves
    assert b.idx(6, 6) in moves


# ---------- 即时胜 / 堵 ----------

def test_mcts_finds_immediate_win():
    """黑已有 (7,5)-(7,8) 四连且轮到黑：应走出成五点。"""
    b = Board()
    play_seq(b, [(7, 5), (0, 0), (7, 6), (0, 1), (7, 7), (0, 2), (7, 8), (0, 3)])
    assert b.to_play == BLACK
    mv = best_move(b, sims=200, rng=random.Random(3))
    assert mv in (b.idx(7, 9), b.idx(7, 4))


def test_mcts_blocks_opponent_win():
    """黑 (7,4)-(7,7) 冲四（左端 (7,3) 已被白堵，仅 (7,8) 可成五），轮到白必须堵 (7,8)。"""
    b = Board()
    play_seq(b, [(7, 4), (7, 3), (7, 5), (0, 0), (7, 6), (0, 1), (7, 7)])
    assert b.to_play == WHITE
    mv = best_move(b, sims=200, rng=random.Random(5))
    assert mv == b.idx(7, 8)


def test_greedy_blocks_immediate_win():
    b = Board()
    play_seq(b, [(7, 4), (7, 3), (7, 5), (0, 0), (7, 6), (0, 1), (7, 7)])
    mv = GreedyPlayer().move(b)
    assert mv == b.idx(7, 8)


# ---------- 合法性与完整对局 ----------

def test_best_move_is_legal():
    b = Board()
    b.apply(b.idx(7, 7))
    b.apply(b.idx(8, 8))
    assert b.is_valid(best_move(b, sims=50, rng=random.Random(1)))


def test_mcts_vs_random_full_game():
    b = Board()
    mcts = MCTSPlayer(sims=30, seed=0)
    rnd = RandomPlayer(seed=1)
    moves = 0
    while not b.is_over and moves < 225:
        mv = mcts.move(b) if b.to_play == BLACK else rnd.move(b)
        assert b.is_valid(mv)
        b.apply(mv)
        moves += 1
    assert b.is_over


@pytest.mark.parametrize("seed", range(2))
def test_mcts_beats_random(seed):
    """MCTS(40) 执黑 vs 随机：应稳定取胜。"""
    b = Board()
    mcts = MCTSPlayer(sims=40, seed=seed)
    rnd = RandomPlayer(seed=seed + 100)
    while not b.is_over:
        mv = mcts.move(b) if b.to_play == BLACK else rnd.move(b)
        b.apply(mv)
    assert b.winner == BLACK


def test_greedy_beats_random():
    b = Board()
    g = GreedyPlayer()
    rnd = RandomPlayer(seed=9)
    while not b.is_over:
        mv = g.move(b) if b.to_play == BLACK else rnd.move(b)
        b.apply(mv)
    assert b.winner == BLACK
