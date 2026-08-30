"""训练循环（设计见 docs/DESIGN.md §9）。

L = (z − v)² − πᵀ log p + λ‖θ‖²；自对弈 → 训练 → 评估门 → checkpoint。
"""

from __future__ import annotations

import csv
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from .arena import evaluate_nets
from .replay import ReplayBuffer
from .selfplay import augment_samples, play_game


def train_step(net, optimizer, s, pi, z, lambda_l2: float, device) -> tuple[float, float, float]:
    """单步训练，返回 (总损失, 策略损失, 价值损失)。"""
    net.train()
    x = torch.from_numpy(s).to(device)
    pi_t = torch.from_numpy(pi).to(device)
    z_t = torch.from_numpy(z).to(device)
    logits, v = net(x)
    logp = torch.log_softmax(logits, dim=1)
    loss_pi = -(pi_t * logp).sum(dim=1).mean()
    loss_v = ((z_t - v.squeeze(1)) ** 2).mean()
    l2 = lambda_l2 * sum(p.pow(2).sum() for p in net.parameters())
    loss = loss_pi + loss_v + l2
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item()), float(loss_pi.item()), float(loss_v.item())


def _progress_report(log, tag: str, done: int, total: int, t0: float, every: int | None = None) -> None:
    """进度报告：每 every 个单位打印一次（含已用/预计剩余时间）。

    自研实现（不引第三方进度库），每行独立输出，对管道/日志友好。
    """
    if total <= 0 or done <= 0:
        return
    if every is None:
        every = max(1, total // 10)
    if done % every != 0 and done != total:
        return
    elapsed = time.time() - t0
    remaining = elapsed / done * (total - done)
    log(f"[{tag}] {done}/{total} ({done / total * 100:.0f}%) | "
        f"已用 {elapsed:.0f}s | 预计剩余 {remaining:.0f}s")


def train_epoch(net, optimizer, buffer: ReplayBuffer, steps: int, batch_size: int,
                lambda_l2: float, device, rng, log=print) -> tuple[float, float, float]:
    """从经验池采样训练 steps 步，返回平均 (总, 策略, 价值) 损失。"""
    total = pi_loss = v_loss = 0.0
    t0 = time.time()
    for i in range(steps):
        s, pi, z = buffer.sample_batch(batch_size, rng)
        l, lp, lv = train_step(net, optimizer, s, pi, z, lambda_l2, device)
        total += l
        pi_loss += lp
        v_loss += lv
        _progress_report(log, "训练", i + 1, steps, t0, every=max(100, steps // 10))
    n = max(steps, 1)
    return total / n, pi_loss / n, v_loss / n


def run_iteration(net, optimizer, buffer: ReplayBuffer, cfg, device, rng,
                  log=print, workers: int = 1) -> dict:
    """一个迭代：自对弈 cfg.games_per_iter 局（增强入池）→ 训练约一个 epoch。

    workers>1 时启用多进程并行自对弈（v2，见 DESIGN.md §8.2）。
    """
    if workers > 1:
        from .selfplay_parallel import parallel_selfplay

        samples = parallel_selfplay(
            net, cfg.games_per_iter, n_workers=workers,
            sims=cfg.sims_train, c_puct=cfg.c_puct,
            temp_steps=cfg.temp_steps, temp=cfg.temp,
            dirichlet_alpha=cfg.dirichlet_alpha, dirichlet_eps=cfg.dirichlet_eps,
            batch_size=64, device=str(device), seed=rng.randrange(1 << 30),
            log=log,
        )
        buffer.add_game(samples)
    else:
        t0 = time.time()
        for i in range(cfg.games_per_iter):
            samples = play_game(
                net, cfg.sims_train, cfg.c_puct, cfg.temp_steps, cfg.temp,
                cfg.dirichlet_alpha, cfg.dirichlet_eps, batch_size=64, device=device, rng=rng,
            )
            buffer.add_game(augment_samples(samples))
            _progress_report(log, "自对弈", i + 1, cfg.games_per_iter, t0)
    steps = max(1, len(buffer) // cfg.batch_size)
    avg = train_epoch(net, optimizer, buffer, steps, cfg.batch_size, cfg.lambda_l2, device, rng, log=log)
    log(f"迭代自对弈完成：新增 {cfg.games_per_iter} 局，经验池 {len(buffer)} 条样本")
    return {"loss": avg[0], "loss_pi": avg[1], "loss_v": avg[2], "buffer": len(buffer)}


def maybe_adopt(net, best_net, cfg, device, rng, log=print) -> bool:
    """评估门：新网 vs 当前最佳，胜率 ≥ 阈值则采纳。返回是否采纳。"""
    t0 = time.time()

    def prog(done, total):
        _progress_report(log, "评估门", done, total, t0, every=max(1, cfg.arena_games // 10))

    stats = evaluate_nets(net, best_net, cfg.arena_games, cfg.sims_eval, device,
                          seed=rng.randrange(1 << 30), progress=prog)
    total = stats["a"] + stats["b"] + stats["draw"]
    winrate = stats["a"] / total if total else 0.0
    adopted = winrate >= cfg.arena_threshold
    log(f"评估门：新网 vs 最佳 {stats}，胜率 {winrate:.1%}，{'采纳' if adopted else '保留最佳'}")
    return adopted


def save_checkpoint(net, optimizer, path, meta: dict | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    meta = dict(meta or {})
    meta.setdefault("n_blocks", len(net.blocks))
    meta.setdefault("n_filters", net.stem[0].weight.shape[0])
    meta.setdefault("board_size", net.board_size)
    torch.save({"model": net.state_dict(), "optimizer": optimizer.state_dict(), "meta": meta}, path)


def net_from_checkpoint(path, device=None) -> "GomokuNet":
    """从 checkpoint 重建网络：架构从 state_dict 推断（兼容无架构元数据的旧版）。"""
    from .net import GomokuNet

    ckpt = torch.load(path, map_location="cpu")
    sd = ckpt["model"]
    meta = ckpt.get("meta", {})
    n_filters = int(meta.get("n_filters", sd["stem.0.weight"].shape[0]))
    n_blocks = int(meta.get("n_blocks", 0))
    if n_blocks == 0:  # 从 state_dict 推断
        while f"blocks.{n_blocks}.conv1.weight" in sd:
            n_blocks += 1
    board_size = int(meta.get("board_size",
                              int(math.sqrt(sd["policy_fc.weight"].shape[1] // 2))))
    net = GomokuNet(board_size=board_size, n_blocks=n_blocks, n_filters=n_filters)
    net.load_state_dict(sd)
    if device is not None:
        net = net.to(device)
    return net


def load_checkpoint(net, optimizer, path) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    net.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("meta", {})


def append_history(csv_path, row: dict) -> None:
    p = Path(csv_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    keys = list(row.keys())
    new_file = not p.exists()
    with p.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        if new_file:
            w.writeheader()
        w.writerow(row)
