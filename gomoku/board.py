"""五子棋规则引擎：棋盘状态、落子、胜负判定、悔棋、状态编码与对称变换。

完全自研，无第三方 AI 组件。仅依赖 NumPy 做数组运算。
设计见 docs/DESIGN.md §3。
"""

from __future__ import annotations

import numpy as np

# 棋子常量
EMPTY = 0
BLACK = 1
WHITE = 2

# apply() 返回值
ONGOING = 0
BLACK_WIN = 1
WHITE_WIN = 2
DRAW = 3

# 四方向（竖 / 横 / 主对角 / 副对角）
_DIRECTIONS = ((1, 0), (0, 1), (1, 1), (1, -1))


class Board:
    """15×15 五子棋棋盘（自由规则、无禁手）。"""

    def __init__(self, size: int = 15):
        self.size = size
        self.reset()

    # ---------- 基础状态 ----------
    def reset(self) -> None:
        self._stones = np.zeros((self.size, self.size), dtype=np.int8)
        self._to_play = BLACK
        self._move_count = 0
        self._history: list[int] = []        # 已落子坐标序号（悔棋栈）
        self._last_move: int | None = None
        self._winner: int | None = None      # None=进行中, DRAW=和棋, 1/2=胜方

    @property
    def stones(self) -> np.ndarray:
        """只读视图：当前棋盘（size×size int8）。"""
        return self._stones

    @property
    def to_play(self) -> int:
        return self._to_play

    @property
    def move_count(self) -> int:
        return self._move_count

    @property
    def last_move(self) -> int | None:
        return self._last_move

    @property
    def winner(self) -> int | None:
        return self._winner

    @property
    def is_over(self) -> bool:
        return self._winner is not None

    # ---------- 坐标 ----------
    def idx(self, r: int, c: int) -> int:
        return r * self.size + c

    def rc(self, idx: int) -> tuple[int, int]:
        return divmod(idx, self.size)

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.size and 0 <= c < self.size

    # ---------- 规则 ----------
    def is_valid(self, move: int) -> bool:
        if not 0 <= move < self.size * self.size:
            return False
        return self._stones.flat[move] == EMPTY

    def legal_moves(self) -> list[int]:
        if self.is_over:
            return []
        return np.flatnonzero(self._stones.flat == EMPTY).tolist()

    def apply(self, move: int) -> int:
        """落子。返回 ONGOING / BLACK_WIN / WHITE_WIN / DRAW。非法落子抛 ValueError。"""
        if not self.is_valid(move):
            raise ValueError(f"非法落子: {move}")
        r, c = self.rc(move)
        player = self._to_play
        self._stones[r, c] = player
        self._history.append(move)
        self._last_move = move
        self._move_count += 1

        if self._count_line(r, c, player) >= 5:
            self._winner = player
        elif self._move_count == self.size * self.size:
            self._winner = DRAW
        else:
            self._to_play = WHITE if player == BLACK else BLACK
        return self._winner if self._winner is not None else ONGOING

    def _count_line(self, r: int, c: int, player: int) -> int:
        """增量判定：统计经过 (r,c) 的 4 条线上同色连子数（含自身）。O(1)。"""
        best = 1
        for dr, dc in _DIRECTIONS:
            cnt = 1
            for sign in (1, -1):
                rr, cc = r + sign * dr, c + sign * dc
                while self.in_bounds(rr, cc) and self._stones[rr, cc] == player:
                    cnt += 1
                    rr += sign * dr
                    cc += sign * dc
            best = max(best, cnt)
        return best

    def undo(self) -> int | None:
        """悔棋一步，返回被撤销的落子（无可悔返回 None）。"""
        if not self._history:
            return None
        move = self._history.pop()
        r, c = self.rc(move)
        self._stones[r, c] = EMPTY
        self._move_count -= 1
        self._last_move = self._history[-1] if self._history else None
        self._winner = None
        self._to_play = BLACK if self._move_count % 2 == 0 else WHITE
        return move

    def copy(self) -> "Board":
        """深拷贝（AI 后台线程推理等并发场景使用）。"""
        b = Board(self.size)
        b._stones = self._stones.copy()
        b._to_play = self._to_play
        b._move_count = self._move_count
        b._history = list(self._history)
        b._last_move = self._last_move
        b._winner = self._winner
        return b

    # ---------- 状态编码（当前玩家视角，见 DESIGN.md §5） ----------
    def encode(self, player: int | None = None) -> np.ndarray:
        """编码为 (4, size, size) float32 张量。player=视角玩家，默认轮到者。"""
        p = player if player is not None else self._to_play
        mine, theirs = (1, 2) if p == BLACK else (2, 1)
        out = np.zeros((4, self.size, self.size), dtype=np.float32)
        out[0] = self._stones == mine          # 通道0: 己方棋子
        out[1] = self._stones == theirs        # 通道1: 对方棋子
        out[2] = 1.0                           # 通道2: 常数层（轮到我）
        if self._last_move is not None:        # 通道3: 上一手落点
            r, c = self.rc(self._last_move)
            out[3, r, c] = 1.0
        return out

    # ---------- 对称变换（D4 群，见 DESIGN.md §3.5） ----------
    @staticmethod
    def symmetry_maps(size: int) -> list[np.ndarray]:
        """8 个坐标映射（225 长度索引重排表）。

        maps[k][src] = dst，含义：位置 src 经变换 k 后落在 dst。
        状态与策略共用同一张表（棋子位置与动作位置同构变换）。
        """
        grid = np.arange(size * size).reshape(size, size)
        transforms = (
            lambda r, c: (r, c),                        # 0 恒等
            lambda r, c: (c, size - 1 - r),             # 1 旋转 90°
            lambda r, c: (size - 1 - r, size - 1 - c),  # 2 旋转 180°
            lambda r, c: (size - 1 - c, r),             # 3 旋转 270°
            lambda r, c: (r, size - 1 - c),             # 4 水平镜像
            lambda r, c: (size - 1 - r, c),             # 5 垂直镜像
            lambda r, c: (c, r),                        # 6 主对角线
            lambda r, c: (size - 1 - c, size - 1 - r),  # 7 副对角线
        )
        maps = []
        for t in transforms:
            dst = np.empty_like(grid)
            for r in range(size):
                for c in range(size):
                    rr, cc = t(r, c)
                    dst[rr, cc] = grid[r, c]
            maps.append(dst.ravel())
        return maps

    def augment(
        self, state: np.ndarray, pi: np.ndarray | None = None, z: float | None = None
    ) -> list[tuple[np.ndarray, np.ndarray | None, float | None]]:
        """训练增强：返回 8 个 (state, pi, z) 变体（状态/策略同步变换，z 不变）。"""
        maps = self.symmetry_maps(self.size)
        flat = state.reshape(4, self.size * self.size)
        out = []
        for m in maps:
            s2 = flat[:, m].reshape(4, self.size, self.size)
            p2 = pi.reshape(-1)[m].reshape(self.size, self.size) if pi is not None else None
            out.append((s2, p2, z))
        return out

    # ---------- 其他 ----------
    def key(self) -> str:
        """局面规范化键（MCTS 复用/调试用）。"""
        return self._stones.tobytes().hex() + f":{self._to_play}"

    def __repr__(self) -> str:
        sym = {EMPTY: ".", BLACK: "X", WHITE: "O"}
        return "\n".join(
            " ".join(sym[int(self._stones[r, c])] for c in range(self.size))
            for r in range(self.size)
        )
