"""多进程并行自对弈（v2，设计见 docs/DESIGN.md §8.2）。

N 个 worker 进程各自独立跑 MCTS（共享同一份只读网络权重），
GPU 前向由各 worker 自行批量完成，CPU 树搜索与 GPU 评估并行重叠，
整体自对弈吞吐约提升 2–4 倍。

Windows 下 multiprocessing 使用 spawn：worker 函数须为模块级、
可导入，权重以 state_dict（纯张量）传递。

用法（tools/train.py --workers N 自动启用）。
"""

from __future__ import annotations

import multiprocessing as mp
import random

from .selfplay import augment_samples, play_game


def _worker(
    rank: int,
    net_state_dict: dict,
    n_blocks: int,
    n_filters: int,
    board_size: int,
    n_games: int,
    sims: int,
    c_puct: float,
    temp_steps: int,
    temp: float,
    dirichlet_alpha: float,
    dirichlet_eps: float,
    batch_size: int,
    device: str,
    queue,
    seed: int,
) -> None:
    """worker 进程：玩 n_games 局自对弈，增强后逐局放入 queue。"""
    import torch

    from .net import GomokuNet

    dev = torch.device(device)
    net = GomokuNet(board_size=board_size, n_blocks=n_blocks, n_filters=n_filters).to(dev)
    net.load_state_dict(net_state_dict)
    net.eval()
    rng = random.Random(seed + rank)
    for _ in range(n_games):
        samples = play_game(
            net, sims, c_puct=c_puct, temp_steps=temp_steps, temp=temp,
            dirichlet_alpha=dirichlet_alpha, dirichlet_eps=dirichlet_eps,
            batch_size=batch_size, device=dev, rng=rng,
        )
        queue.put(augment_samples(samples))


def parallel_selfplay(
    net,
    n_games: int,
    n_workers: int = 4,
    sims: int = 200,
    c_puct: float = 1.5,
    temp_steps: int = 15,
    temp: float = 1.0,
    dirichlet_alpha: float = 0.3,
    dirichlet_eps: float = 0.25,
    batch_size: int = 64,
    device: str | None = None,
    seed: int = 0,
    log=print,
) -> list:
    """并行自对弈 n_games 局，返回增强样本列表（[(s, pi, z)]）。

    注意：worker 均从同一个 checkpoint 权重出发，各局互不影响；
    调用方负责在全部结束后加载最新网络再进入下一阶段。
    """
    import time

    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    n_blocks = len(net.blocks)
    n_filters = net.stem[0].weight.shape[0]
    board_size = net.board_size

    state_dict = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    games_per_worker = [n_games // n_workers] * n_workers
    for i in range(n_games % n_workers):
        games_per_worker[i] += 1

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [
        ctx.Process(
            target=_worker,
            args=(
                rank, state_dict, n_blocks, n_filters, board_size, games_per_worker[rank],
                sims, c_puct, temp_steps, temp, dirichlet_alpha, dirichlet_eps,
                batch_size, device, queue, seed,
            ),
        )
        for rank in range(n_workers)
    ]
    for p in procs:
        p.start()
    t0 = time.time()
    games = []
    every = max(1, n_games // 10)
    for i in range(n_games):
        games.append(queue.get())
        if (i + 1) % every == 0 or i + 1 == n_games:
            elapsed = time.time() - t0
            remaining = elapsed / (i + 1) * (n_games - i - 1)
            log(f"[并行自对弈] {i + 1}/{n_games} ({(i + 1) / n_games * 100:.0f}%) | "
                f"已用 {elapsed:.0f}s | 预计剩余 {remaining:.0f}s")
    for p in procs:
        p.join()
    return [s for game in games for s in game]
