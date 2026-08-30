"""训练进度汇总：读取 history CSV，输出 loss 趋势、评估门采纳、vs 基线胜率。

用法：
    python tools/summary.py [--csv runs/history_long2.csv] [--last N]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="runs/history_long2.csv")
    ap.add_argument("--last", type=int, default=20, help="显示最近 N 个迭代")
    args = ap.parse_args(argv)

    path = Path(args.csv)
    if not path.exists():
        print(f"[FAIL] 找不到 {path}")
        return 1

    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("[FAIL] 空记录")
        return 1

    rows = rows[-args.last:]
    print(f"记录文件：{path}（共 {len(rows)} 行）")
    print(f"{'迭代':>4} {'loss':>8} {'loss_pi':>8} {'loss_v':>7} {'采纳':>4} {'vs随机':>6} {'vs贪心':>6} {'耗时(s)':>8}")
    for r in rows:
        print(f"{r.get('iteration',''):>4} {r.get('loss',''):>8} {r.get('loss_pi',''):>8} "
              f"{r.get('loss_v',''):>7} {str(r.get('adopted','')):>4} "
              f"{r.get('vs_random',''):>6} {r.get('vs_greedy',''):>6} {r.get('seconds',''):>8}")

    losses = [float(r["loss"]) for r in rows if r.get("loss")]
    if losses:
        print(f"\nloss 趋势：{losses[0]:.4f} → {losses[-1]:.4f}"
              f"（区间 {len(losses)} 迭代，变化 {losses[-1] - losses[0]:+.4f}）")
    adopted = sum(1 for r in rows if r.get("adopted") == "True")
    print(f"评估门采纳：{adopted}/{len(rows)} 次")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
