"""可复用的对弈引擎：随机 / 贪心启发式 / 纯 MCTS。

供 GUI（人机对战、观战）、tools/eval.py 与测试共用。
"""

from __future__ import annotations

import random

from gomoku.board import Board, BLACK, WHITE, EMPTY

_DIRECTIONS = ((1, 0), (0, 1), (1, 1), (1, -1))


class RandomPlayer:
    """均匀随机落子（最弱基线）。"""

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def move(self, board: Board) -> int:
        return self.rng.choice(board.legal_moves())


class GreedyPlayer:
    """自研贪心启发式：模拟落子评分（成五 > 活四 > 冲四 > 活三 > …）+ 防守权重。"""

    WIN = 10_000_000
    OPEN_4 = 100_000
    FOUR = 10_000
    OPEN_3 = 1_000
    THREE = 100
    OPEN_2 = 10
    TWO = 1

    def move(self, board: Board) -> int:
        legal = board.legal_moves()
        if not legal:
            return -1
        me = board.to_play
        opp = WHITE if me == BLACK else BLACK
        best, best_score = legal[0], -float("inf")
        for m in legal:
            score = self._score_move(board, m, me) + 0.9 * self._score_move(board, m, opp)
            if score > best_score:
                best, best_score = m, score
        return best

    @classmethod
    def _score_move(cls, board: Board, move: int, color: int) -> float:
        """模拟 color 在 move 落子的 4 方向连子与开放性评分。"""
        r0, c0 = board.rc(move)
        score = 0.0
        for dr, dc in _DIRECTIONS:
            cnt = 1
            open_ends = 0
            for sign in (1, -1):
                rr, cc = r0 + sign * dr, c0 + sign * dc
                while board.in_bounds(rr, cc) and board.stones[rr, cc] == color:
                    cnt += 1
                    rr += sign * dr
                    cc += sign * dc
                if board.in_bounds(rr, cc) and board.stones[rr, cc] == EMPTY:
                    open_ends += 1
            if cnt >= 5:
                score += cls.WIN
            elif cnt == 4 and open_ends == 2:
                score += cls.OPEN_4
            elif cnt == 4:
                score += cls.FOUR
            elif cnt == 3 and open_ends == 2:
                score += cls.OPEN_3
            elif cnt == 3:
                score += cls.THREE
            elif cnt == 2 and open_ends == 2:
                score += cls.OPEN_2
            elif cnt == 2:
                score += cls.TWO
        return score


class MCTSPlayer:
    """纯 MCTS 引擎（无神经网络）。"""

    def __init__(self, sims: int = 200, c: float = 1.4, seed: int | None = None):
        self.sims = sims
        self.c = c
        self.rng = random.Random(seed)

    def move(self, board: Board) -> int:
        if board.move_count == 0:
            return board.idx(board.size // 2, board.size // 2)  # 空盘首选天元
        from .mcts import best_move

        return best_move(board, self.sims, self.c, self.rng)
