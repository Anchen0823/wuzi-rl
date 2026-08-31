"""pygame 五子棋界面（M4：人vs人 / 人vsAI 三难度 / AIvsAI 观战）。

完全自研：仅用 pygame 图形与事件 + 自研 AI 引擎（ai/players.py），
无任何第三方 AI 组件。AI 推理在后台线程执行，不阻塞 UI。
网络引擎从 checkpoints/net.pt 加载（缺失时自动回退纯 MCTS）。
设计见 docs/DESIGN.md §4。
"""

from __future__ import annotations

import json
import queue
import threading
from pathlib import Path

import pygame
import torch

from ai.net import GomokuNet
from ai.players import MCTSPlayer, NetPlayer
from ai.train import net_from_checkpoint
from .board import Board, BLACK, WHITE, ONGOING, BLACK_WIN, WHITE_WIN, DRAW
from .game import Game

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
DEFAULT_CHECKPOINT = "checkpoints/best.pt"   # 最强模型（评估门采纳）
FALLBACK_CHECKPOINT = "checkpoints/net.pt"   # 最新模型兜底
GAME_DIR = "runs/games"   # 对局记录（回放数据源）

# 主菜单：按键 → (模式, AI 执子, 引擎, 模拟数, 说明)
# pvc 中 ai_player 为 AI 执子颜色；engine: mcts=纯 MCTS, net=网络+PUCT。
MODES = {
    pygame.K_1: ("pvp", None, None, 0, "人 vs 人"),
    pygame.K_2: ("pvc", WHITE, "mcts", 200, "人 vs AI·低（纯 MCTS 200）"),
    pygame.K_3: ("pvc", WHITE, "net", 400, "人 vs AI·中（网络 400）"),
    pygame.K_4: ("pvc", WHITE, "net", 800, "人 vs AI·高（网络 800）"),
    pygame.K_5: ("pvc", BLACK, "net", 800, "人执白 vs AI·高（网络 800）"),
    pygame.K_6: ("cvc", None, "net", 400, "AI vs AI 观战（网络 400）"),
}
MENU_LINES = (
    "五子棋 · 完全自研 AI",
    "",
    "1  人 vs 人",
    "2  人 vs AI·低（纯 MCTS 200 模拟）",
    "3  人 vs AI·中（网络 400 模拟）",
    "4  人 vs AI·高（网络 800 模拟）",
    "5  人执白 vs AI·高（网络 800 模拟）",
    "6  AI vs AI 观战（网络 400 模拟）",
    "7  回放上一局（空格/退格步进）",
    "",
    "ESC  退出",
)


class GomokuApp:
    """五子棋应用：主菜单 + 对局（含 AI 后台线程）。"""

    def __init__(self, size: int = 15, mode: str | None = None, sims: int = 200,
                 ai_player: int | None = None, engine: str = "mcts", net=None):
        pygame.init()
        pygame.display.set_caption("五子棋（M4）｜ 1-6 模式 · R 重开 · U 悔棋 · ESC 菜单")
        self.size = size
        self.board_size = size * CELL + 2 * MARGIN
        self.screen = pygame.display.set_mode((self.board_size, self.board_size + PANEL_H))
        self.font = pygame.font.SysFont("microsoftyahei", 24)
        self.menu_font = pygame.font.SysFont("microsoftyahei", 26)
        self.board = Board(size)
        self.message = "黑方先手"
        self._win_line: list[tuple[int, int]] | None = None

        # 网络引擎（加载最强模型 best.pt，缺失回退最新 net.pt；均缺失时网络档自动回退纯 MCTS）
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if net is not None:
            self._net = net.to(self._device)
            self._net_loaded = True
        else:
            self._net, self._net_loaded = None, False
            for path in (DEFAULT_CHECKPOINT, FALLBACK_CHECKPOINT):
                try:
                    self._net = net_from_checkpoint(path, device=self._device)
                    self._net_loaded = True
                    break
                except FileNotFoundError:
                    continue
        if self._net_loaded:
            self._net.eval()

        # 对局模式状态
        self.screen_state = "game" if mode is not None else "menu"
        self.mode: str | None = mode            # pvp / pvc / cvc
        self.ai_player: int | None = ai_player  # pvc 模式下 AI 执子颜色
        self.engine: str = engine              # mcts / net
        self.ai_sims: int = sims
        self._ai_thread: threading.Thread | None = None
        self._ai_result: "queue.Queue[int]" = queue.Queue()
        # 回放状态
        self.replay_board: Board | None = None
        self.replay_moves: list[int] = []
        self.replay_idx = 0
        self.replay_result: int | None = None
        self.replay_record: dict = {}
        if mode is not None:
            self._enter_mode(mode, ai_player, engine, sims)

    # ---------- 模式 ----------
    def _enter_mode(self, mode: str, ai_player: int | None = None,
                    engine: str = "mcts", sims: int = 200) -> None:
        self.mode = mode
        self.ai_player = ai_player
        self.engine = engine
        self.ai_sims = sims
        self._reset_board()
        self.screen_state = "game"

    def _reset_board(self) -> None:
        self.board.reset()
        self._win_line = None
        self.message = "黑方先手"

    # ---------- 回放 ----------
    def _enter_replay(self) -> None:
        """加载最新一局记录并进入回放（空格=下一步，退格=上一步）。"""
        files = sorted(Path(GAME_DIR).glob("game_*.json"))
        if not files:
            self.screen_state = "menu"
            return
        self.replay_record = Game.load_record(str(files[-1]))
        self.replay_board = Board(self.replay_record.get("size", self.size))
        self.replay_moves = list(self.replay_record.get("moves", []))
        self.replay_idx = 0
        self.replay_result = self.replay_record.get("result")
        self.screen_state = "replay"

    def _replay_step(self, delta: int) -> None:
        """步进回放；delta=±1。进入终局时显示胜负。"""
        assert self.replay_board is not None
        n = len(self.replay_moves)
        self.replay_idx = max(0, min(n, self.replay_idx + delta))
        self.replay_board.reset()
        for mv in self.replay_moves[: self.replay_idx]:
            self.replay_board.apply(mv)

    # ---------- 事件 ----------
    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.QUIT:
            pygame.quit()
            raise SystemExit
        if event.type == pygame.KEYDOWN:
            if self.screen_state == "replay":
                if event.key == pygame.K_ESCAPE:
                    self.screen_state = "menu"
                elif event.key == pygame.K_SPACE:
                    self._replay_step(+1)
                elif event.key == pygame.K_BACKSPACE:
                    self._replay_step(-1)
                return
            if self.screen_state == "menu":
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    raise SystemExit
                if event.key == pygame.K_7:
                    self._enter_replay()
                elif event.key in MODES:
                    m, ai, eng, sims, _ = MODES[event.key]
                    self._enter_mode(m, ai, eng, sims)
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
        self._save_record()

    def _save_record(self) -> None:
        """对局结束自动保存记录（回放数据源）。"""
        import time

        Path(GAME_DIR).mkdir(parents=True, exist_ok=True)
        path = Path(GAME_DIR) / f"game_{time.strftime('%Y%m%d_%H%M%S')}.json"
        Game.save_game(Game.from_record({
            "size": self.size, "moves": self.board.history, "result": self.board.winner,
        }), str(path))

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

    def _make_engine(self):
        """按当前档位构造引擎；网络档缺模型时回退纯 MCTS。"""
        if self.engine == "net" and self._net is not None:
            return NetPlayer(self._net, sims=self.ai_sims, device=self._device)
        return MCTSPlayer(sims=self.ai_sims)

    def _maybe_start_ai(self) -> None:
        if not self._ai_active():
            return
        if self._ai_thread is not None and self._ai_thread.is_alive():
            return
        b = self.board.copy()
        engine = self._make_engine()
        if self.engine == "net" and self._net is None:
            self.message = "未找到网络模型，回退纯 MCTS"
        else:
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
    def _draw_board_state(self, screen, board: Board, win_line, message: str) -> None:
        """绘制棋盘 + 状态栏（对局与回放共用）。"""
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
                v = int(board.stones[r, c])
                if v == 0:
                    continue
                cx, cy = MARGIN + c * CELL, MARGIN + r * CELL
                color = COLORS["black"] if v == BLACK else COLORS["white"]
                pygame.draw.circle(screen, color, (cx, cy), CELL // 2 - 3)
                pygame.draw.circle(screen, COLORS["line"], (cx, cy), CELL // 2 - 3, 1)
        # 上一手标记
        if board.last_move is not None:
            r, c = board.rc(board.last_move)
            cx, cy = MARGIN + c * CELL, MARGIN + r * CELL
            pygame.draw.circle(screen, COLORS["last"], (cx, cy), 5)
        # 胜利连线
        if win_line:
            pts = [(MARGIN + c * CELL, MARGIN + r * CELL) for r, c in win_line]
            pygame.draw.lines(screen, COLORS["last"], False, pts, 4)
        # 状态栏
        text = self.font.render(message, True, COLORS["text"])
        screen.blit(text, (MARGIN, self.board_size + 15))
        pygame.display.flip()

    def draw(self) -> None:
        if self.screen_state == "menu":
            self._draw_menu()
        elif self.screen_state == "replay":
            self._draw_replay()
        else:
            self._draw_board_state(self.screen, self.board, self._win_line, self.message)

    def _draw_replay(self) -> None:
        board = self.replay_board or Board(self.size)
        if self.replay_idx >= len(self.replay_moves) and self.replay_result is not None:
            if self.replay_result == BLACK_WIN:
                msg = f"回放 {self.replay_idx}/{len(self.replay_moves)} 手 · 黑方胜"
            elif self.replay_result == WHITE_WIN:
                msg = f"回放 {self.replay_idx}/{len(self.replay_moves)} 手 · 白方胜"
            else:
                msg = f"回放 {self.replay_idx}/{len(self.replay_moves)} 手 · 和棋"
        else:
            msg = f"回放 {self.replay_idx}/{len(self.replay_moves)} 手（空格/退格步进，ESC 返回）"
        win_line = self._find_win_line_for(board)
        self._draw_board_state(self.screen, board, win_line, msg)

    def _find_win_line_for(self, board: Board) -> list[tuple[int, int]] | None:
        """回放用：任意 ≥5 连子的首段（简单查找）。"""
        for r in range(self.size):
            for c in range(self.size):
                v = int(board.stones[r, c])
                if v == 0:
                    continue
                for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
                    cells = []
                    rr, cc = r, c
                    while board.in_bounds(rr, cc) and board.stones[rr, cc] == v:
                        cells.append((rr, cc))
                        rr += dr
                        cc += dc
                    if len(cells) >= 5:
                        return cells
        return None

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
