"""M5 测试：对局记录保存/加载与 GUI 回放。"""

import json
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

from gomoku.board import Board, BLACK, WHITE
from gomoku.game import Game
from gomoku import gui
from gomoku.gui import GomokuApp


def _click(app: GomokuApp, r: int, c: int) -> None:
    pos = (gui.MARGIN + c * gui.CELL, gui.MARGIN + r * gui.CELL)
    app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=pos))


def test_game_record_roundtrip(tmp_path):
    g = Game()
    for r, c in [(7, 7), (7, 8), (7, 6), (8, 7), (8, 8)]:
        g.play(g.board.idx(r, c))
    path = tmp_path / "rec.json"
    Game.save_game(g, str(path))
    rec = Game.load_record(str(path))
    g2 = Game.from_record(rec)
    assert g2.moves == g.moves
    assert g2.board.key() == g.board.key()


def test_gui_autosave_and_replay(tmp_path, monkeypatch):
    monkeypatch.setattr(gui, "GAME_DIR", str(tmp_path))
    app = GomokuApp(mode="pvp")
    try:
        # 下完一局（黑五连胜）
        for r, c in [(7, 5), (8, 0), (7, 6), (8, 1), (7, 7), (8, 2), (7, 8), (8, 3), (7, 9)]:
            _click(app, r, c)
        assert app.board.winner == BLACK
        files = list(tmp_path.glob("game_*.json"))
        assert len(files) == 1
        rec = json.loads(files[0].read_text(encoding="utf-8"))
        assert len(rec["moves"]) == 9
        assert rec["result"] == BLACK

        # 回放：回菜单 → 按键 7 进入 → 步进/退格 → 终局 → ESC
        app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
        assert app.screen_state == "menu"
        app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_7))
        assert app.screen_state == "replay"
        assert app.replay_idx == 0
        app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
        app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
        assert app.replay_idx == 2
        assert app.replay_board.move_count == 2
        app.draw()  # 渲染一帧
        app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_BACKSPACE))
        assert app.replay_idx == 1
        for _ in range(20):  # 步进到终局
            app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
        assert app.replay_idx == 9
        app.draw()
        app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
        assert app.screen_state == "menu"
    finally:
        pygame.quit()


def test_replay_no_records_stays_in_menu(tmp_path, monkeypatch):
    monkeypatch.setattr(gui, "GAME_DIR", str(tmp_path))
    app = GomokuApp()
    try:
        app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_7))
        assert app.screen_state == "menu"
    finally:
        pygame.quit()
