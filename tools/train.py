"""训练入口：自对弈 → 训练 → 评估门 → checkpoint（见 PLAN.md §6 / DESIGN.md §9）。

用法：
    # 冒烟（小规模跑通全流程）
    python tools/train.py --iters 2 --games-per-iter 2 --sims-train 40 \
        --arena-games 2 --sims-eval 40 --n-blocks 2 --n-filters 64

    # 正式训练（默认配置，4060 上长跑；Ctrl+C 后 --checkpoint 续训）
    python tools/train.py --iters 50
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from config import Config
from ai.net import GomokuNet
from ai.replay import ReplayBuffer
from ai.train import (
    run_iteration,
    maybe_adopt,
    save_checkpoint,
    load_checkpoint,
    append_history,
)
from ai.arena import evaluate_vs_player
from ai.players import RandomPlayer, GreedyPlayer


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--games-per-iter", type=int, default=None)
    ap.add_argument("--sims-train", type=int, default=None)
    ap.add_argument("--sims-eval", type=int, default=None)
    ap.add_argument("--arena-games", type=int, default=None)
    ap.add_argument("--n-blocks", type=int, default=None)
    ap.add_argument("--n-filters", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--eval-baseline-every", type=int, default=0,
                    help="每 N 迭代对战一次贪心/随机基线并记录（0=关闭）")
    ap.add_argument("--workers", type=int, default=1,
                    help="并行自对弈进程数（v2，>1 时启用，吞吐约 2-4 倍）")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--checkpoint", default=None, help="续训：加载已有 checkpoint")
    ap.add_argument("--out", default="checkpoints/net.pt")
    ap.add_argument("--history", default="runs/history.csv")
    args = ap.parse_args(argv)

    BEST_OUT = "checkpoints/best.pt"   # 评估门采纳的“最强模型”（GUI/对战优先加载）

    overrides = {}
    if args.games_per_iter:
        overrides["games_per_iter"] = args.games_per_iter
    if args.sims_train:
        overrides["sims_train"] = args.sims_train
    if args.sims_eval:
        overrides["sims_eval"] = args.sims_eval
    if args.arena_games:
        overrides["arena_games"] = args.arena_games
    if args.batch_size:
        overrides["batch_size"] = args.batch_size
    if args.lr:
        overrides["lr"] = args.lr
    cfg = replace(Config(), **overrides)
    n_blocks = args.n_blocks or cfg.n_res_blocks
    n_filters = args.n_filters or cfg.n_filters

    device = torch.device(args.device)
    rng = random.Random(0)

    net = GomokuNet(board_size=cfg.board_size, n_blocks=n_blocks, n_filters=n_filters).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    start_iter = 0
    if args.checkpoint and Path(args.checkpoint).exists():
        meta = load_checkpoint(net, optimizer, args.checkpoint)
        start_iter = meta.get("iteration", 0)
        print(f"续训：从迭代 {start_iter} 继续（{args.checkpoint}）")

    buffer = ReplayBuffer(capacity_games=cfg.buffer_size, board_size=cfg.board_size)
    best_net = GomokuNet(board_size=cfg.board_size, n_blocks=n_blocks, n_filters=n_filters).to(device)
    best_net.load_state_dict(net.state_dict())

    print(f"配置：iters={args.iters} games/iter={cfg.games_per_iter} sims_train={cfg.sims_train} "
          f"sims_eval={cfg.sims_eval} arena={cfg.arena_games} batch={cfg.batch_size} "
          f"lr={cfg.lr} blocks={n_blocks} filters={n_filters} device={device}")
    print(f"网络参数量：{net.num_params() / 1e6:.2f}M")

    for it in range(start_iter, start_iter + args.iters):
        t0 = time.time()
        res = run_iteration(net, optimizer, buffer, cfg, device, rng, workers=args.workers)
        dt = time.time() - t0
        adopted = maybe_adopt(net, best_net, cfg, device, rng)
        if adopted:
            best_net.load_state_dict(net.state_dict())
            save_checkpoint(best_net, None, BEST_OUT, {"iteration": it + 1})  # 最强模型
        save_checkpoint(net, optimizer, args.out, {"iteration": it + 1})
        row = {
            "iteration": it + 1,
            "loss": round(res["loss"], 4),
            "loss_pi": round(res["loss_pi"], 4),
            "loss_v": round(res["loss_v"], 4),
            "buffer": res["buffer"],
            "adopted": adopted,
            "seconds": round(dt, 1),
            "vs_random": "",
            "vs_greedy": "",
        }
        # 定期对战基线，观察学习曲线（vs 贪心更能反映棋力成长）
        if args.eval_baseline_every > 0 and (it + 1) % args.eval_baseline_every == 0:
            for name, fac in (("random", RandomPlayer), ("greedy", GreedyPlayer)):
                st = evaluate_vs_player(net, fac, 4, min(cfg.sims_eval, 300), device, seed=1)
                row[f"vs_{name}"] = f"{st.get('net', 0)}-{st.get('player', 0)}"
                print(f"  [基线] vs {name}: {st}")
        append_history(args.history, row)
        print(f"迭代 {it + 1} 完成：loss={res['loss']:.4f} (pi={res['loss_pi']:.4f} v={res['loss_v']:.4f}) "
              f"耗时 {dt:.0f}s，checkpoint → {args.out}")

    print("===== 最终 vs 基线 =====")
    for name, fac in (("random", RandomPlayer), ("greedy", GreedyPlayer)):
        st = evaluate_vs_player(net, fac, 4, min(cfg.sims_eval, 100), device, seed=1)
        print(f"  vs {name}: {st}")
    # 若本段从未采纳且无历史 best.pt，落盘当前网络作为 best 兜底
    if not Path(BEST_OUT).exists():
        save_checkpoint(best_net, None, BEST_OUT, {"iteration": start_iter + args.iters})
        print(f"无采纳记录，best.pt 使用初始/续训网络（{BEST_OUT}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
