"""M4 测试：checkpoint 架构推断与 GUI 网络引擎接入。"""

import os
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame
import pytest
import torch

from ai.net import GomokuNet
from ai.players import NetPlayer
from ai.train import save_checkpoint, net_from_checkpoint
from gomoku.board import Board, BLACK, WHITE
from gomoku.gui import GomokuApp

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_net(**kw):
    defaults = dict(board_size=15, n_blocks=2, n_filters=32)
    defaults.update(kw)
    return GomokuNet(**defaults)


# ---------- checkpoint ----------

def test_checkpoint_roundtrip_and_arch_inference(tmp_path):
    net = make_net()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    path = str(tmp_path / "net.pt")
    save_checkpoint(net, opt, path, {"iteration": 3})
    # 无架构元数据也能从 state_dict 推断（模拟旧版）
    net2 = net_from_checkpoint(path, device="cpu")
    assert net2.num_params() == net.num_params()
    assert len(net2.blocks) == 2
    assert net2.stem[0].weight.shape[0] == 32
    # 前向一致
    net.eval()
    net2.eval()
    x = torch.randn(1, 4, 15, 15)
    with torch.no_grad():
        p1, v1 = net(x)
        p2, v2 = net2(x)
    assert torch.allclose(p1, p2, atol=1e-6)
    assert torch.allclose(v1, v2, atol=1e-6)


def test_checkpoint_without_optimizer(tmp_path):
    """best.pt 场景：仅保存模型（optimizer=None）可正常保存/加载。"""
    net = make_net()
    path = str(tmp_path / "best.pt")
    save_checkpoint(net, None, path, {"iteration": 5, "adopted": True})
    net2 = net_from_checkpoint(path, device="cpu")
    assert net2.num_params() == net.num_params()
    meta = torch.load(path, map_location="cpu")["meta"]
    assert meta["adopted"] is True


def test_gui_prefers_best_checkpoint(tmp_path, monkeypatch):
    """best.pt 与 net.pt 同时存在时，GUI 优先加载 best.pt。"""
    net_best = make_net(n_filters=32)
    net_latest = make_net(n_filters=64)   # 不同架构，便于区分
    best_path = tmp_path / "best.pt"
    net_path = tmp_path / "net.pt"
    save_checkpoint(net_best, None, str(best_path))
    save_checkpoint(net_latest, None, str(net_path))
    monkeypatch.setattr("gomoku.gui.DEFAULT_CHECKPOINT", str(best_path))
    monkeypatch.setattr("gomoku.gui.FALLBACK_CHECKPOINT", str(net_path))
    app = GomokuApp()   # 默认主菜单（会触发网络加载）
    try:
        assert app._net_loaded
        assert app._net.stem[0].weight.shape[0] == 32   # 加载的是 best（32 通道）
    finally:
        pygame.quit()


def test_net_player_moves_legally():
    net = make_net().to(DEV)
    b = Board()
    b.apply(b.idx(7, 7))
    b.apply(b.idx(7, 8))
    p = NetPlayer(net, sims=20, device=DEV, seed=1)
    mv = p.move(b)
    assert b.is_valid(mv)


# ---------- GUI 网络引擎 ----------

def test_gui_cvc_net_mode_completes():
    """cvc + 网络引擎：小网络低模拟数完整下一局。"""
    net = make_net().to(DEV)
    app = GomokuApp(mode="cvc", engine="net", sims=12, net=net)
    try:
        deadline = time.time() + 300
        while time.time() < deadline:
            app._maybe_start_ai()
            app._consume_ai()
            if app.board.is_over:
                break
            time.sleep(0.005)
        assert app.board.is_over, "AIvsAI（网络）未在时限内终局"
        app.draw()
        if app._ai_thread is not None and app._ai_thread.is_alive():
            app._ai_thread.join(timeout=5)
        assert not app._ai_thread.is_alive()
    finally:
        pygame.quit()


def test_gui_net_fallback_when_no_checkpoint(tmp_path, monkeypatch):
    """网络档但无 checkpoint（best/net 均缺失）→ 回退纯 MCTS 并提示。"""
    monkeypatch.setattr("gomoku.gui.DEFAULT_CHECKPOINT", str(tmp_path / "missing_best.pt"))
    monkeypatch.setattr("gomoku.gui.FALLBACK_CHECKPOINT", str(tmp_path / "missing_net.pt"))
    app = GomokuApp(mode="pvc", engine="net", sims=20, ai_player=WHITE)
    try:
        assert app._net is None
        engine = app._make_engine()
        assert engine.__class__.__name__ == "MCTSPlayer"
        app.board.apply(app.board.idx(7, 7))   # 人类黑先
        app._maybe_start_ai()
        assert "回退" in app.message           # 落子前已提示回退
        deadline = time.time() + 60
        while app.board.move_count < 2 and time.time() < deadline:
            app._consume_ai()
            time.sleep(0.005)
        assert app.board.move_count == 2       # 白（回退 MCTS）已应一手
    finally:
        pygame.quit()
