"""纯 MCTS 基线（无神经网络）。设计见 docs/DESIGN.md §7.4。

经典 UCB1 树搜索 + 快速 rollout：
  - 展开：懒展开，每次迭代只展开一个子节点
  - 选择：Q(s,a) + c·√(ln N_parent / (1 + N_child))，c 默认 1.4
  - 模拟：成五 > 堵对方成五 > 邻域偏置随机，直至终局
  - 回传：价值按玩家视角逐层翻转累加（w 为「本节点轮到的玩家」视角）

完全自研：不参考、不复制任何开源 MCTS / 棋类 AI 实现。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

from gomoku.board import Board, BLACK, WHITE, EMPTY, DRAW

_DIRECTIONS = ((1, 0), (0, 1), (1, 1), (1, -1))


def would_win(board: Board, move: int, player: int) -> bool:
    """若 player 在 move 落子即形成 ≥5 连，返回 True（不改变局面）。

    以 move 为中心沿 4 方向数同色连子（move 计 1），无落子副作用。
    """
    r, c = board.rc(move)
    for dr, dc in _DIRECTIONS:
        cnt = 1
        for sign in (1, -1):
            rr, cc = r + sign * dr, c + sign * dc
            while board.in_bounds(rr, cc) and board.stones[rr, cc] == player:
                cnt += 1
                rr += sign * dr
                cc += sign * dc
        if cnt >= 5:
            return True
    return False


def adjacent_moves(board: Board) -> list[int]:
    """邻域着法：与已有棋子切比雪夫距离 ≤ 1 的合法点；空盘时返回全部合法点。"""
    legal = board.legal_moves()
    if board.move_count == 0 or not legal:
        return legal
    n = board.size
    cand = set()
    for idx in np.flatnonzero(board.stones.flat != EMPTY):
        r, c = divmod(int(idx), n)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < n and 0 <= cc < n:
                    m = rr * n + cc
                    if m in legal:
                        cand.add(m)
    return sorted(cand) if cand else legal


@dataclass
class Node:
    """MCTS 树节点。w 与 q 均为「本节点轮到的玩家」视角。"""

    move: int | None
    parent: "Node | None" = None
    n: int = 0
    w: float = 0.0
    children: dict[int, "Node"] = field(default_factory=dict)
    untried: list[int] = field(default_factory=list)

    @property
    def q(self) -> float:
        return self.w / self.n if self.n else 0.0


def _ucb_select(node: Node, c: float) -> int:
    """UCB1 选择子节点，返回着法。node 必须有 children。"""
    ln = math.log(node.n + 1.0)
    best, best_score = None, -float("inf")
    for move, child in node.children.items():
        score = child.q + c * math.sqrt(ln / (1.0 + child.n))
        if score > best_score:
            best, best_score = move, score
    assert best is not None
    return best


def _neighbors(board: Board, idx: int) -> list[int]:
    """返回 (r,c) 的 8 邻域内合法格点序号。"""
    n = board.size
    r, c = divmod(idx, n)
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < n and 0 <= cc < n:
                out.append(rr * n + cc)
    return out


def _rollout(board: Board, rng: random.Random) -> float:
    """快速模拟至终局，返回「模拟开始时轮到的玩家」视角的价值（+1 / 0 / -1）。

    候选集用增量维护（落子时只并入其 8 邻域），避免每步全盘扫描。
    """
    player = board.to_play
    frontier: set[int] = set()
    for idx in np.flatnonzero(board.stones.flat != EMPTY):
        frontier.update(_neighbors(board, int(idx)))
    while not board.is_over:
        cur = board.to_play
        moves = [m for m in frontier if board.stones.flat[m] == EMPTY]
        if not moves:
            moves = board.legal_moves()
        opp = WHITE if cur == BLACK else BLACK
        chosen = None
        for m in moves:
            if would_win(board, m, cur):
                chosen = m
                break
        if chosen is None:
            for m in moves:
                if would_win(board, m, opp):
                    chosen = m
                    break
        if chosen is None:
            chosen = rng.choice(moves)
        board.apply(chosen)
        frontier.update(_neighbors(board, chosen))
    w = board.winner
    if w == player:
        return 1.0
    if w == DRAW:
        return 0.0
    return -1.0


def mcts_search(board: Board, sims: int, c: float = 1.4, rng: random.Random | None = None) -> Node:
    """在 board 的副本上运行 sims 次模拟，返回根节点（含子节点统计）。"""
    rng = rng or random.Random()
    root = Node(move=None)
    # 只展开邻域着法：225 路分支下全展开会使 UCB1 探索过宽（见测试用例），
    # 邻域剪枝（切比雪夫距离 ≤1）大幅提升搜索质量；空盘时退化为全部合法点。
    root.untried = adjacent_moves(board)
    for _ in range(sims):
        node = root
        b = board.copy()
        # 1) 选择：沿完全展开的路径下探
        while node.untried == [] and node.children:
            b.apply(_ucb_select(node, c))
            node = node.children[b.last_move]
        # 2) 展开：若可展开且未终局，懒展开一个子节点
        if node.untried and not b.is_over:
            move = node.untried.pop()
            b.apply(move)
            child = Node(move=move, parent=node)
            child.untried = adjacent_moves(b)   # 邻域剪枝
            node.children[move] = child
            node = child
        # 3) 模拟到终局
        value = _rollout(b, rng)
        # 4) 回传（价值按玩家视角逐层翻转）
        while node is not None:
            node.n += 1
            node.w += value
            value = -value
            node = node.parent
    return root


def best_move(board: Board, sims: int, c: float = 1.4, rng: random.Random | None = None) -> int:
    """返回最优着法。

    确定性强制着优先（自研轻量启发式，见 DESIGN.md §7.4）：
      1. 己方有一步成五 → 必走；
      2. 对方有一步成五 → 必堵；
    否则运行 MCTS 搜索（访问次数最多者；平局取 Q 高者）。
    """
    me = board.to_play
    opp = WHITE if me == BLACK else BLACK
    for m in board.legal_moves():
        if would_win(board, m, me):
            return m
    for m in board.legal_moves():
        if would_win(board, m, opp):
            return m
    root = mcts_search(board, sims, c, rng)
    if not root.children:
        moves = board.legal_moves()
        return moves[0] if moves else -1
    return max(root.children.items(), key=lambda kv: (kv[1].n, kv[1].q))[0]
