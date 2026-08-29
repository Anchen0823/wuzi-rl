"""pygame 界面冒烟测试（SDL dummy 视频驱动，无窗口运行）。"""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

from gomoku import gui
from gomoku.gui import GomokuApp


def _click(app: GomokuApp, r: int, c: int) -> None:
    pos = (gui.MARGIN + c * gui.CELL, gui.MARGIN + r * gui.CELL)
    app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=pos))


def test_gui_place_stones_and_draw():
    app = GomokuApp()
    try:
        _click(app, 7, 7)
        _click(app, 7, 8)
        assert app.board.move_count == 2
        assert app.board.to_play == gui.BLACK
        assert "黑方 行棋" in app.message
        app.draw()  # 渲染一帧不报错
    finally:
        pygame.quit()


def test_gui_win_message_and_restart():
    app = GomokuApp()
    try:
        # 黑连五：(7,5)..(7,9)，白隔开应一手
        for r, c in [(7, 5), (8, 0), (7, 6), (8, 1), (7, 7), (8, 2), (7, 8), (8, 3), (7, 9)]:
            _click(app, r, c)
        assert app.board.winner == gui.BLACK
        assert "黑方胜" in app.message
        assert app._win_line is not None
        app.draw()
        # R 重开
        app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r))
        assert app.board.move_count == 0
        assert app._win_line is None
    finally:
        pygame.quit()


def test_gui_undo():
    app = GomokuApp()
    try:
        _click(app, 7, 7)
        _click(app, 7, 8)
        app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_u))
        assert app.board.move_count == 1
        assert app.board.to_play == gui.WHITE
    finally:
        pygame.quit()
