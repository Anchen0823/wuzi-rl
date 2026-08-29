"""经验池（设计见 docs/DESIGN.md §8）。

滑动窗口按「局」存储原始样本；采样时随机施加一种 D4 对称
（等价于增强但内存省 8 倍，且增强多样性更强）。
"""

from __future__ import annotations

import random

import numpy as np

from gomoku.board import Board
from .selfplay import _symmetry_maps

Sample = tuple[np.ndarray, np.ndarray, float]  # (state(4,15,15), pi(225), z)


class ReplayBuffer:
    def __init__(self, capacity_games: int = 500, board_size: int = 15):
        self.capacity = capacity_games
        self.board_size = board_size
        self.games: list[list[Sample]] = []
        self._total = 0

    def add_game(self, samples: list[Sample]) -> None:
        self.games.append(samples)
        self._total += len(samples)
        while len(self.games) > self.capacity:
            self._total -= len(self.games.pop(0))

    def __len__(self) -> int:
        return self._total

    def sample_batch(self, batch_size: int, rng: random.Random | None = None):
        """随机采样 batch_size 条样本（可重复），每条施加随机 D4 对称。

        返回 (s (B,4,15,15), pi (B,225), z (B,))。
        """
        rng = rng or random.Random()
        if self._total == 0:
            raise ValueError("经验池为空，请先自对弈生成数据")
        maps = _symmetry_maps(self.board_size)
        ss, pis, zs = [], [], []
        for _ in range(batch_size):
            g = self.games[rng.randrange(len(self.games))]
            s, pi, z = g[rng.randrange(len(g))]
            m = maps[rng.randrange(len(maps))]
            ss.append(s.reshape(4, -1)[:, m].reshape(4, self.board_size, self.board_size))
            pis.append(pi.reshape(-1)[m])
            zs.append(z)
        return np.stack(ss), np.stack(pis), np.asarray(zs, dtype=np.float32)
