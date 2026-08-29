"""自研边界守护：依赖白名单审计（docs/PLAN.md §1.2 落地工具）。

用法：
    python tools/dep_audit.py [--repo PATH] [--check-installed]

检查：
  1) 仓库内 .py 源码的 import 是否引入禁用 AI 组件库；
  2) 仓库内是否混入外部模型/权重文件；
  3) --check-installed：已安装 pip 包是否超出白名单（基础工具）。
"""

from __future__ import annotations

import argparse
import importlib.metadata
import pathlib
import re
import sys

# 允许的基础工具（通用计算/图形，非 AI 组件）
ALLOWLIST = {"pygame", "pygame-ce", "numpy", "torch", "pytest", "pip", "setuptools", "wheel"}

# 禁用：任何 AI 模型 / RL / 组件库
FORBIDDEN_IMPORTS = (
    "stable_baselines",
    "sb3",
    "tensorflow",
    "keras",
    "torchvision",
    "torchaudio",
    "transformers",
    "timm",
    "gym",
    "ray",
    "jax",
    "flax",
    "openai",
    "langchain",
    "sklearn",
    "scikit_learn",
    "lightgbm",
    "xgboost",
    "catboost",
    "mlflow",
    "optuna",
)

# 外部权重/模型文件扩展名
WEIGHT_EXTS = {
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
    ".pb",
    ".h5",
    ".hdf5",
    ".safetensors",
    ".bin",
    ".pkl",
    ".joblib",
}

SKIP_DIRS = {".git", ".venv", "checkpoints", "data", "logs", "runs", "__pycache__", ".pytest_cache"}

_IMPORT_RE = re.compile(r"^\s*(?:import|from)\s+([\w.]+)")


def iter_py_files(repo: pathlib.Path):
    for p in repo.rglob("*.py"):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        yield p


def scan_imports(repo: pathlib.Path) -> list[str]:
    bad = []
    for f in iter_py_files(repo):
        for i, line in enumerate(f.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            m = _IMPORT_RE.match(line)
            if not m:
                continue
            mod = m.group(1).split(".")[0]
            if mod in FORBIDDEN_IMPORTS:
                bad.append(f"{f}:{i}: import {mod}（禁用 AI 组件）")
    return bad


def scan_weights(repo: pathlib.Path) -> list[str]:
    bad = []
    for p in repo.rglob("*"):
        if p.is_file() and p.suffix.lower() in WEIGHT_EXTS:
            if any(part in SKIP_DIRS for part in p.parts):
                continue
            bad.append(str(p))
    return bad


def scan_installed() -> list[str]:
    installed = {d.metadata["Name"].lower() for d in importlib.metadata.distributions()}
    return sorted(installed - ALLOWLIST)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=".", help="仓库根目录")
    ap.add_argument("--check-installed", action="store_true", help="同时审计已安装 pip 包")
    args = ap.parse_args(argv)

    repo = pathlib.Path(args.repo).resolve()
    problems: list[tuple[str, str]] = []
    problems += [("import", x) for x in scan_imports(repo)]
    problems += [("weight", x) for x in scan_weights(repo)]
    if args.check_installed:
        problems += [("installed", x) for x in scan_installed()]

    if problems:
        print(f"[FAIL] 依赖审计发现 {len(problems)} 处违规：")
        for kind, x in problems:
            print(f"  [{kind}] {x}")
        return 1
    print("[OK] 依赖审计通过：无禁用 AI 组件、无外部权重文件。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
