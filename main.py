"""启动入口。

用法：
    python main.py                          # 主菜单（1-6 选模式）
    python main.py --mode pvc --engine net --sims 400   # 直接人机对战（网络 400）
"""

import argparse

from gomoku.gui import GomokuApp, WHITE

# 模式 → (AI 执子, 默认引擎, 默认模拟数)
MODE_AI = {
    "pvp": (None, "mcts", 200),
    "pvc": (WHITE, "net", 400),
    "cvc": (None, "net", 400),
}


def main() -> None:
    ap = argparse.ArgumentParser(description="五子棋启动器（完全自研 AI）")
    ap.add_argument("--mode", choices=["pvp", "pvc", "cvc"], default=None,
                    help="直接进入指定模式（默认主菜单）")
    ap.add_argument("--engine", choices=["mcts", "net"], default=None,
                    help="AI 引擎：mcts=纯 MCTS，net=网络+PUCT（默认按模式）")
    ap.add_argument("--sims", type=int, default=None, help="模拟次数/步（难度档）")
    args = ap.parse_args()

    ai_player, engine, sims = MODE_AI[args.mode] if args.mode else (None, "mcts", 200)
    if args.engine:
        engine = args.engine
    if args.sims:
        sims = args.sims
    app = GomokuApp(mode=args.mode, ai_player=ai_player, engine=engine, sims=sims)
    app.run()


if __name__ == "__main__":
    main()
