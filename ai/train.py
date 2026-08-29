"""训练循环（设计见 docs/DESIGN.md §9）。

L = (z − v)² − πᵀ log p + λ‖θ‖²；自对弈 → 训练 → 评估门 → checkpoint。
"""

from __future__ import annotations

import csv
import random
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


def train_epoch(net, optimizer, buffer: ReplayBuffer, steps: int, batch_size: int,
                lambda_l2: float, device, rng) -> tuple[float, float, float]:
    """从经验池采样训练 steps 步，返回平均 (总, 策略, 价值) 损失。"""
    total = pi_loss = v_loss = 0.0
    for _ in range(steps):
        s, pi, z = buffer.sample_batch(batch_size, rng)
        l, lp, lv = train_step(net, optimizer, s, pi, z, lambda_l2, device)
        total += l
        pi_loss += lp
        v_loss += lv
    n = max(steps, 1)
    return total / n, pi_loss / n, v_loss / n


def run_iteration(net, optimizer, buffer: ReplayBuffer, cfg, device, rng, log=print) -> dict:
    """一个迭代：自对弈 cfg.games_per_iter 局（增强入池）→ 训练约一个 epoch。"""
    for _ in range(cfg.games_per_iter):
        samples = play_game(
            net, cfg.sims_train, cfg.c_puct, cfg.temp_steps, cfg.temp,
            cfg.dirichlet_alpha, cfg.dirichlet_eps, batch_size=64, device=device, rng=rng,
        )
        buffer.add_game(augment_samples(samples))
    steps = max(1, len(buffer) // cfg.batch_size)
    avg = train_epoch(net, optimizer, buffer, steps, cfg.batch_size, cfg.lambda_l2, device, rng)
    log(f"迭代自对弈完成：新增 {cfg.games_per_iter} 局，经验池 {len(buffer)} 条样本")
    return {"loss": avg[0], "loss_pi": avg[1], "loss_v": avg[2], "buffer": len(buffer)}


def maybe_adopt(net, best_net, cfg, device, rng) -> bool:
    """评估门：新网 vs 当前最佳，胜率 ≥ 阈值则采纳。返回是否采纳。"""
    stats = evaluate_nets(net, best_net, cfg.arena_games, cfg.sims_eval, device, seed=rng.randrange(1 << 30))
    total = stats["a"] + stats["b"] + stats["draw"]
    winrate = stats["a"] / total if total else 0.0
    adopted = winrate >= cfg.arena_threshold
    print(f"评估门：新网 vs 最佳 {stats}，胜率 {winrate:.1%}，{'采纳' if adopted else '保留最佳'}")
    return adopted


def save_checkpoint(net, optimizer, path, meta: dict | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": net.state_dict(), "optimizer": optimizer.state_dict(), "meta": meta or {}}, path)


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
