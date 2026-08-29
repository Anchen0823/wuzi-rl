"""pygame 五子棋界面（M1：人 vs 人，含悔棋/重开/胜负提示）。

完全自研：仅用 pygame 图形与事件，无任何第三方 AI 组件。
AI 接入在 M2/M4 加入（后台线程推理），本模块保持与 AI 层解耦。
设计见 docs/DESIGN.md §4。
"""

from __future__ import annotations

import pygame

from .board import Board, BLACK, WHITE, ONGOING, BLACK_WIN, WHITE_WIN, DRAW

CELL = 40                 # 网格间距（像素）
MARGIN = 40               # 棋盘边距
PANEL_H = 60              # 状态栏高度
STAR_POINTS = ((3, 3), (11, 3), (3, 11), (11, 11), (7, 7))  # 15 路星位
COLORS = {
    "bg": (222, 184, 135),     # 木色棋盘
    "line": (60, 40, 20),
    "black": (20, 20, 20),
    "white": (245, 245, 245),
    "edge": (200, 60, 60),
    "text": (40, 30, 20),
    "panel": (240, 230, 200),
    "last": (255, 80, 80),     # 上一手 / 胜利连线高亮
}
NAME = {BLACK: "黑方", WHITE: "白方"}


class GomokuApp:
    """人 vs 人五子棋应用。"""

    def __init__(self, size: int = 15):
        pygame.init()
        pygame.display.set_caption("五子棋（人 vs 人）｜ R 重开 · U 悔棋")
        self.size = size
        self.board_size = size * CELL + 2 * MARGIN
        self.screen = pygame.display.set_mode((self.board_size, self.board_size + PANEL_H))
        self.font = pygame.font.SysFont("microsoftyahei", 24)
        self.board = Board(size)
        self.message = "黑方先手"
        self._win_line: list[tuple[int, int]] | None = None

    # ---------- 事件 ----------
    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.QUIT:
            pygame.quit()
            raise SystemExit
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                self._restart()
            elif event.key == pygame.K_u and not self.board.is_over:
                self._undo()
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._on_click(event.pos)

    def _on_click(self, pos: tuple[int, int]) -> None:
        if self.board.is_over:
            return
        r = round((pos[1] - MARGIN) / CELL)
        c = round((pos[0] - MARGIN) / CELL)
        if not (0 <= r < self.size and 0 <= c < self.size):
            return
        move = self.board.idx(r, c)
        if not self.board.is_valid(move):
            self.message = "此处已有棋子"
            return
        res = self.board.apply(move)
        if res == ONGOING:
            self.message = f"{NAME[self.board.to_play]} 行棋"
        else:
            self._finish(res)

    def _finish(self, res: int) -> None:
        if res == BLACK_WIN:
            self._win_line = self._find_win_line(BLACK)
            self.message = "黑方胜！(R 重开 / U 悔棋)"
        elif res == WHITE_WIN:
            self._win_line = self._find_win_line(WHITE)
            self.message = "白方胜！(R 重开 / U 悔棋)"
        else:
            self._win_line = None
            self.message = "和棋（棋盘已满）"

    def _find_win_line(self, player: int) -> list[tuple[int, int]] | None:
        """找出包含上一手的连五（任意 ≥5 连续同色）。"""
        b, last = self.board, self.board.last_move
        if last is None:
            return None
        r0, c0 = b.rc(last)
        for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
            cells = [(r0, c0)]
            for sign in (1, -1):
                rr, cc = r0 + sign * dr, c0 + sign * dc
                while b.in_bounds(rr, cc) and b.stones[rr, cc] == player:
                    cells.append((rr, cc))
                    rr += sign * dr
                    cc += sign * dc
            if len(cells) >= 5:
                return cells
        return None

    def _undo(self) -> None:
        if self.board.undo() is not None:
            self._win_line = None
            self.message = f"{NAME[self.board.to_play]} 行棋（已悔棋）"

    def _restart(self) -> None:
        self.board.reset()
        self._win_line = None
        self.message = "黑方先手"

    # ---------- 渲染 ----------
    def draw(self) -> None:
        screen = self.screen
        screen.fill(COLORS["panel"])

        # 棋盘底
        edge = self.board_size - 2 * MARGIN + 40
        pygame.draw.rect(screen, COLORS["bg"], (MARGIN - 20, MARGIN - 20, edge, edge))
        # 网格
        for i in range(self.size):
            p = MARGIN + i * CELL
            pygame.draw.line(screen, COLORS["line"], (MARGIN, p), (self.board_size - MARGIN, p), 1)
            pygame.draw.line(screen, COLORS["line"], (p, MARGIN), (p, self.board_size - MARGIN), 1)
        # 星位
        for r, c in STAR_POINTS:
            cx, cy = MARGIN + c * CELL, MARGIN + r * CELL
            pygame.draw.circle(screen, COLORS["line"], (cx, cy), 4)
        # 棋子
        for r in range(self.size):
            for c in range(self.size):
                v = int(self.board.stones[r, c])
                if v == 0:
                    continue
                cx, cy = MARGIN + c * CELL, MARGIN + r * CELL
                color = COLORS["black"] if v == BLACK else COLORS["white"]
                pygame.draw.circle(screen, color, (cx, cy), CELL // 2 - 3)
                pygame.draw.circle(screen, COLORS["line"], (cx, cy), CELL // 2 - 3, 1)
        # 上一手标记
        if self.board.last_move is not None:
            r, c = self.board.rc(self.board.last_move)
            cx, cy = MARGIN + c * CELL, MARGIN + r * CELL
            pygame.draw.circle(screen, COLORS["last"], (cx, cy), 5)
        # 胜利连线
        if self._win_line:
            pts = [(MARGIN + c * CELL, MARGIN + r * CELL) for r, c in self._win_line]
            pygame.draw.lines(screen, COLORS["last"], False, pts, 4)
        # 状态栏
        text = self.font.render(self.message, True, COLORS["text"])
        screen.blit(text, (MARGIN, self.board_size + 15))
        pygame.display.flip()

    def run(self) -> None:
        clock = pygame.time.Clock()
        while True:
            for event in pygame.event.get():
                self.handle_event(event)
            self.draw()
            clock.tick(60)
