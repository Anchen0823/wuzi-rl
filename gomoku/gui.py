"""pygame 五子棋界面（M2：人vs人 / 人vsAI / AIvsAI 观战）。

完全自研：仅用 pygame 图形与事件 + 自研 AI 引擎（ai/players.py），
无任何第三方 AI 组件。AI 推理在后台线程执行，不阻塞 UI。
设计见 docs/DESIGN.md §4。
"""

from __future__ import annotations

import queue
import threading

import pygame

from ai.players import MCTSPlayer
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

# 主菜单：按键 → (模式, AI 执子, 说明)
# 模式 pvc 中 ai_player 为 AI 执子颜色，人类执另一色；cvc 双方均为 AI。
MODES = {
    pygame.K_1: ("pvp", None, "人 vs 人"),
    pygame.K_2: ("pvc", WHITE, "人 vs AI（人执黑）"),
    pygame.K_3: ("pvc", BLACK, "人 vs AI（人执白）"),
    pygame.K_4: ("cvc", None, "AI vs AI 观战"),
}
MENU_LINES = (
    "五子棋 · 完全自研 AI",
    "",
    "1  人 vs 人",
    "2  人 vs AI（人执黑）",
    "3  人 vs AI（人执白）",
    "4  AI vs AI 观战",
    "",
    "ESC  退出",
)


class GomokuApp:
    """五子棋应用：主菜单 + 对局（含 AI 后台线程）。"""

    def __init__(self, size: int = 15, mode: str | None = None, sims: int = 200,
                 ai_player: int | None = None):
        pygame.init()
        pygame.display.set_caption("五子棋（M2）｜ 1-4 模式 · R 重开 · U 悔棋 · ESC 菜单")
        self.size = size
        self.board_size = size * CELL + 2 * MARGIN
        self.screen = pygame.display.set_mode((self.board_size, self.board_size + PANEL_H))
        self.font = pygame.font.SysFont("microsoftyahei", 24)
        self.menu_font = pygame.font.SysFont("microsoftyahei", 26)
        self.sims = sims
        self.board = Board(size)
        self.message = "黑方先手"
        self._win_line: list[tuple[int, int]] | None = None

        # 对局模式状态
        self.screen_state = "game" if mode is not None else "menu"
        self.mode: str | None = mode            # pvp / pvc / cvc
        self.ai_player: int | None = ai_player  # pvc 模式下 AI 执子颜色
        self._ai_thread: threading.Thread | None = None
        self._ai_result: "queue.Queue[int]" = queue.Queue()
        if mode is not None:
            self._enter_mode(mode, ai_player)

    # ---------- 模式 ----------
    def _enter_mode(self, mode: str, ai_player: int | None = None) -> None:
        self.mode = mode
        self.ai_player = ai_player
        self._reset_board()
        self.screen_state = "game"

    def _reset_board(self) -> None:
        self.board.reset()
        self._win_line = None
        self.message = "黑方先手"

    # ---------- 事件 ----------
    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.QUIT:
            pygame.quit()
            raise SystemExit
        if event.type == pygame.KEYDOWN:
            if self.screen_state == "menu":
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    raise SystemExit
                if event.key in MODES:
                    m, ai, _ = MODES[event.key]
                    self._enter_mode(m, ai)
                return
            if event.key == pygame.K_r:
                self._reset_board()
            elif event.key == pygame.K_u and not self.board.is_over:
                self._undo()
            elif event.key == pygame.K_ESCAPE:
                self.screen_state = "menu"      # 返回主菜单
        if (
            self.screen_state == "game"
            and event.type == pygame.MOUSEBUTTONDOWN
            and event.button == 1
            and self._human_can_play()
        ):
            self._on_click(event.pos)

    def _human_can_play(self) -> bool:
        if self.board.is_over:
            return False
        if self.mode == "cvc":
            return False
        if self.mode == "pvc":
            return self.board.to_play != self.ai_player
        return True

    def _on_click(self, pos: tuple[int, int]) -> None:
        r = round((pos[1] - MARGIN) / CELL)
        c = round((pos[0] - MARGIN) / CELL)
        if not (0 <= r < self.size and 0 <= c < self.size):
            return
        move = self.board.idx(r, c)
        if not self.board.is_valid(move):
            self.message = "此处已有棋子"
            return
        self._apply(move)

    def _apply(self, move: int) -> None:
        res = self.board.apply(move)
        if res == ONGOING:
            self.message = f"{NAME[self.board.to_play]} 行棋"
        else:
            self._finish(res)

    def _finish(self, res: int) -> None:
        if res == BLACK_WIN:
            self._win_line = self._find_win_line(BLACK)
            self.message = "黑方胜！(R 重开 / ESC 菜单)"
        elif res == WHITE_WIN:
            self._win_line = self._find_win_line(WHITE)
            self.message = "白方胜！(R 重开 / ESC 菜单)"
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

    # ---------- AI 后台线程 ----------
    def _ai_active(self) -> bool:
        if self.screen_state != "game" or self.board.is_over:
            return False
        if self.mode == "cvc":
            return True
        if self.mode == "pvc":
            return self.board.to_play == self.ai_player
        return False

    def _maybe_start_ai(self) -> None:
        if not self._ai_active():
            return
        if self._ai_thread is not None and self._ai_thread.is_alive():
            return
        b = self.board.copy()
        engine = MCTSPlayer(sims=self.sims)
        self.message = "AI 思考中…"
        self._ai_thread = threading.Thread(target=self._compute_ai, args=(b, engine), daemon=True)
        self._ai_thread.start()

    def _compute_ai(self, b: Board, engine: MCTSPlayer) -> None:
        try:
            move = engine.move(b)
            self._ai_result.put(move)
        except Exception:
            self._ai_result.put(-1)

    def _consume_ai(self) -> None:
        if not self._ai_active():
            return
        try:
            move = self._ai_result.get_nowait()
        except queue.Empty:
            return
        if not self.board.is_valid(move):
            self.message = "AI 内部错误，请 R 重开"
            return
        self._apply(move)

    # ---------- 渲染 ----------
    def draw(self) -> None:
        if self.screen_state == "menu":
            self._draw_menu()
            return
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

    def _draw_menu(self) -> None:
        screen = self.screen
        screen.fill(COLORS["panel"])
        y = 80
        for line in MENU_LINES:
            surf = self.menu_font.render(line, True, COLORS["text"])
            screen.blit(surf, (60, y))
            y += 44
        pygame.display.flip()

    def run(self) -> None:
        clock = pygame.time.Clock()
        while True:
            self._maybe_start_ai()
            for event in pygame.event.get():
                self.handle_event(event)
            self._consume_ai()
            self.draw()
            clock.tick(60)
