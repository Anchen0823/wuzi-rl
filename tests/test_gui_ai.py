"""GUI AI 集成冒烟测试（SDL dummy）：AIvsAI 观战模式完整对局。"""

import os
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

from gomoku.gui import GomokuApp


def test_ai_vs_ai_game_completes():
    """cvc 模式：双 AI（纯 MCTS 低模拟数）完整下一局，不卡死、能终局。"""
    app = GomokuApp(mode="cvc", sims=30)
    try:
        deadline = time.time() + 300
        while time.time() < deadline:
            app._maybe_start_ai()
            app._consume_ai()
            if app.board.is_over:
                break
            time.sleep(0.005)
        assert app.board.is_over, "AIvsAI 未在时限内终局"
        app.draw()  # 终局渲染一帧
        # 等待后台线程结束，避免悬挂
        if app._ai_thread is not None and app._ai_thread.is_alive():
            app._ai_thread.join(timeout=5)
        assert not app._ai_thread.is_alive()
    finally:
        pygame.quit()
