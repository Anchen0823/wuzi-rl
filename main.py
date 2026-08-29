"""启动入口。

用法：
    python main.py                          # 主菜单（1-4 选模式）
    python main.py --mode pvc --sims 100    # 直接进入人机对战（AI 执白）
"""

import argparse

from gomoku.gui import GomokuApp, WHITE

MODE_AI = {"pvp": None, "pvc": WHITE, "cvc": None}


def main() -> None:
    ap = argparse.ArgumentParser(description="五子棋启动器（完全自研 AI）")
    ap.add_argument("--mode", choices=["pvp", "pvc", "cvc"], default=None,
                    help="直接进入指定模式（默认主菜单）")
    ap.add_argument("--sims", type=int, default=200, help="纯 MCTS 模拟次数/步（难度档）")
    args = ap.parse_args()
    app = GomokuApp(mode=args.mode, sims=args.sims,
                    ai_player=MODE_AI[args.mode] if args.mode else None)
    app.run()


if __name__ == "__main__":
    main()
