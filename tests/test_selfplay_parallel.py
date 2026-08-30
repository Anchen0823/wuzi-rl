"""多进程并行自对弈（v2）测试：CPU 运行，验证样本合法性与数量。"""

import numpy as np

from ai.net import GomokuNet
from ai.selfplay_parallel import parallel_selfplay


def make_net(**kw):
    defaults = dict(board_size=15, n_blocks=2, n_filters=32)
    defaults.update(kw)
    return GomokuNet(**defaults)


def test_parallel_selfplay_cpu():
    net = make_net()
    samples = parallel_selfplay(
        net, n_games=2, n_workers=2, sims=10, batch_size=8, device="cpu", seed=1,
    )
    assert len(samples) > 0
    for s, pi, z in samples[:10]:
        assert s.shape == (4, 15, 15)
        assert abs(pi.sum() - 1.0) < 1e-5
        assert z in (-1.0, 0.0, 1.0)
    # 至少存在 ±1 样本（对局有胜负）或全 0（和棋），且无其他值
    assert set(samples[i][2] for i in range(len(samples))) <= {-1.0, 0.0, 1.0}


def test_parallel_selfplay_game_count():
    net = make_net()
    # 3 局 / 2 worker（分配 2+1）
    samples = parallel_selfplay(net, n_games=3, n_workers=2, sims=8, batch_size=8,
                                device="cpu", seed=2)
    assert len(samples) >= 3 * 9  # 每局至少 9 步 × 8 增强
