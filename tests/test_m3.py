"""M3 训练管线测试：网络、PUCT、自对弈、经验池、训练循环。"""

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gomoku.board import Board, BLACK, WHITE
from ai.net import GomokuNet
from ai.mcts import puct_search, puct_visit_distribution, puct_best_move, _evaluate, _evaluate_batch
from ai.selfplay import play_game, augment_samples
from ai.replay import ReplayBuffer
from ai.train import train_step, run_iteration

DEV = torch.device("cpu")


def make_net(**kw):
    defaults = dict(board_size=15, n_blocks=2, n_filters=32)
    defaults.update(kw)
    return GomokuNet(**defaults)


def play_seq(b: Board, seq: list[tuple[int, int]]):
    for r, c in seq:
        b.apply(b.idx(r, c))


# ---------- 网络 ----------

def test_net_forward_shapes():
    net = make_net()
    x = torch.zeros(4, 4, 15, 15)
    p, v = net(x)
    assert p.shape == (4, 225)
    assert v.shape == (4, 1)


def test_net_initial_value_near_zero():
    net = make_net()
    net.eval()
    with torch.no_grad():
        _, v = net(torch.zeros(2, 4, 15, 15))
    assert v.abs().max().item() < 0.05


def test_net_forward_masked():
    net = make_net()
    net.eval()
    legal = torch.zeros(1, 225)
    legal[0, :10] = 1
    with torch.no_grad():
        p, _ = net.forward_masked(torch.zeros(1, 4, 15, 15), legal)
    assert abs(p.sum().item() - 1.0) < 1e-5
    assert p[0, 10:].max().item() == 0.0   # 非法点概率为 0


def test_net_param_count_full():
    n = GomokuNet(n_blocks=6, n_filters=256).num_params()
    assert 6_000_000 < n < 8_000_000       # 设计预算 ~7.2M


@pytest.mark.skipif(not torch.cuda.is_available(), reason="无 CUDA")
def test_net_gpu_forward():
    net = make_net().cuda()
    p, v = net(torch.zeros(8, 4, 15, 15, device="cuda"))
    torch.cuda.synchronize()
    assert p.shape == (8, 225)
    assert v.shape == (8, 1)


# ---------- PUCT ----------

def test_puct_tree_stats_and_distribution():
    net = make_net()
    b = Board()
    b.apply(b.idx(7, 7))
    b.apply(b.idx(7, 8))
    root = puct_search(b, net, sims=60, batch_size=32, device=DEV, rng=random.Random(1))
    assert root.n == 60
    pi = puct_visit_distribution(root, temp=1.0, board_size=15)
    assert abs(pi.sum() - 1.0) < 1e-5
    assert pi[b.idx(7, 7)] == 0.0 and pi[b.idx(7, 8)] == 0.0   # 已占位概率为 0
    pi0 = puct_visit_distribution(root, temp=0.0, board_size=15)
    assert pi0.sum() == 1.0 and pi0.max() == 1.0


def test_puct_root_noise_preserves_distribution():
    net = make_net()
    b = Board()
    root = puct_search(b, net, sims=10, add_root_noise=True, device=DEV, rng=random.Random(2))
    s = sum(ch.p for ch in root.children.values())
    assert abs(s - 1.0) < 1e-4
    for ch in root.children.values():
        assert ch.p > 0


def test_puct_finds_immediate_win_and_block():
    net = make_net()
    b = Board()
    play_seq(b, [(7, 5), (0, 0), (7, 6), (0, 1), (7, 7), (0, 2), (7, 8), (0, 3)])
    assert puct_best_move(b, net, sims=30, device=DEV) in (b.idx(7, 9), b.idx(7, 4))
    b2 = Board()
    play_seq(b2, [(7, 4), (7, 3), (7, 5), (0, 0), (7, 6), (0, 1), (7, 7)])
    assert puct_best_move(b2, net, sims=30, device=DEV) == b2.idx(7, 8)


def test_batch_eval_matches_single_eval():
    net = make_net()
    net.eval()
    boards = []
    for i in range(5):
        b = Board()
        b.apply(b.idx(7 + i, 7))
        b.apply(b.idx(8, 8))
        boards.append(b)
    singles = [_evaluate(b, net, DEV) for b in boards]
    ps, vs = _evaluate_batch(boards, net, DEV)
    for (p1, v1), (p2, v2) in zip(singles, zip(ps, vs)):
        assert np.allclose(p1, p2, atol=1e-6)
        assert abs(v1 - v2) < 1e-6


# ---------- 自对弈 ----------

def test_selfplay_game_valid():
    net = make_net()
    samples = play_game(net, sims=20, batch_size=16, device=DEV, rng=random.Random(3))
    assert len(samples) >= 9
    for s, pi, z in samples:
        assert s.shape == (4, 15, 15)
        assert abs(pi.sum() - 1.0) < 1e-5
        assert z in (-1.0, 0.0, 1.0)
    # 终局一致性：黑方视角样本的 z 全部相同，白方为其相反数（或全 0 和棋）
    bz = {z for i, (_, _, z) in enumerate(samples) if i % 2 == 0}
    wz = {z for i, (_, _, z) in enumerate(samples) if i % 2 == 1}
    assert len(bz) == 1 and len(wz) == 1
    assert list(bz)[0] == -list(wz)[0]


def test_augment_samples():
    net = make_net()
    samples = play_game(net, sims=10, batch_size=8, device=DEV, rng=random.Random(4))
    aug = augment_samples(samples[:2])
    assert len(aug) == 16
    for s, pi, z in aug:
        assert s.shape == (4, 15, 15)
        assert pi.shape == (225,)
        assert abs(pi.sum() - 1.0) < 1e-5


# ---------- 经验池 ----------

def test_replay_buffer():
    rng = random.Random(5)
    buf = ReplayBuffer(capacity_games=3, board_size=15)
    net = make_net()
    for _ in range(4):
        buf.add_game(play_game(net, sims=8, batch_size=8, device=DEV, rng=rng))
    assert len(buf.games) == 3                 # 滑动窗口
    s, pi, z = buf.sample_batch(16, rng)
    assert s.shape == (16, 4, 15, 15)
    assert pi.shape == (16, 225)
    assert z.shape == (16,)
    assert abs(pi.sum(axis=1) - 1.0).max() < 1e-5


# ---------- 训练循环 ----------

def test_train_step_finite():
    net = make_net()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    s = np.random.rand(8, 4, 15, 15).astype(np.float32)
    pi = np.full((8, 225), 1 / 225, dtype=np.float32)
    z = np.array([1, -1, 0, 1, -1, 0, 1, 1], dtype=np.float32)
    l, lp, lv = train_step(net, opt, s, pi, z, 1e-4, DEV)
    assert np.isfinite(l) and np.isfinite(lp) and np.isfinite(lv)


def test_mini_iteration_smoke():
    net = make_net()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    cfg = SimpleNamespace(
        games_per_iter=2, sims_train=16, c_puct=1.5, temp_steps=3, temp=1.0,
        dirichlet_alpha=0.3, dirichlet_eps=0.25, batch_size=32, lambda_l2=1e-4,
    )
    buf = ReplayBuffer(capacity_games=5, board_size=15)
    res = run_iteration(net, opt, buf, cfg, DEV, random.Random(6), log=lambda *a: None)
    assert res["buffer"] > 0
    assert np.isfinite(res["loss"])
