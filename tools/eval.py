"""命令行对战评估：随机 / 贪心 / 纯 MCTS / 网络引擎两两对战（换边双局制）。

用法：
    python tools/eval.py --games 20 --sims 200 --p1 mcts --p2 greedy
    python tools/eval.py --games 10 --sims 400 --p1 net --p2 greedy   # 网络引擎（加载 checkpoints/net.pt）
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 脚本直跑时保证包可导入

import torch

from gomoku.board import Board, BLACK, WHITE, DRAW
from ai.players import RandomPlayer, GreedyPlayer, MCTSPlayer, NetPlayer
from ai.train import net_from_checkpoint

FACTORIES = {
    "random": lambda sims: RandomPlayer(),
    "greedy": lambda sims: GreedyPlayer(),
    "mcts": lambda sims: MCTSPlayer(sims=sims, seed=0),
    "net": lambda sims: NetPlayer(
        net_from_checkpoint(
            "checkpoints/net.pt",
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        ),
        sims=sims, seed=0,
    ),
}


def play_one(black, white) -> int:
    """双方各下一局（不换边），返回胜者。"""
    b = Board()
    while not b.is_over:
        mv = (black if b.to_play == BLACK else white).move(b)
        b.apply(mv)
    return b.winner


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=20, help="对局数（自动黑白换边，偶数）")
    ap.add_argument("--sims", type=int, default=200, help="MCTS 模拟次数/步")
    ap.add_argument("--p1", choices=list(FACTORIES), default="mcts")
    ap.add_argument("--p2", choices=list(FACTORIES), default="greedy")
    args = ap.parse_args(argv)

    p1 = FACTORIES[args.p1](args.sims)
    p2 = FACTORIES[args.p2](args.sims)

    p1_wins = p2_wins = draws = 0
    for i in range(args.games):
        black, white = (p1, p2) if i % 2 == 0 else (p2, p1)
        w = play_one(black, white)
        if w == DRAW:
            draws += 1
        elif (w == BLACK) == (black is p1):
            p1_wins += 1
        else:
            p2_wins += 1

    total = args.games
    print(f"对战: {args.p1}(sims={args.sims}) vs {args.p2}  共 {total} 局（黑白换边）")
    print(f"  {args.p1} 胜 {p1_wins} | {args.p2} 胜 {p2_wins} | 和棋 {draws}")
    print(f"  {args.p1} 胜率 {p1_wins / total:.1%}（换边对称，随机基线期望 50%）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
