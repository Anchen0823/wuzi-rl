"""纯 MCTS 基线（无神经网络）。设计见 docs/DESIGN.md §7.4。

经典 UCB1 树搜索 + 快速 rollout：
  - 展开：懒展开，每次迭代只展开一个子节点
  - 选择：Q(s,a) + c·√(ln N_parent / (1 + N_child))，c 默认 1.4
  - 模拟：成五 > 堵对方成五 > 邻域偏置随机，直至终局
  - 回传：价值按玩家视角逐层翻转累加（w 为「本节点轮到的玩家」视角）

完全自研：不参考、不复制任何开源 MCTS / 棋类 AI 实现。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np
import torch

from gomoku.board import Board, BLACK, WHITE, EMPTY, DRAW

_DIRECTIONS = ((1, 0), (0, 1), (1, 1), (1, -1))


def would_win(board: Board, move: int, player: int) -> bool:
    """若 player 在 move 落子即形成 ≥5 连，返回 True（不改变局面）。

    以 move 为中心沿 4 方向数同色连子（move 计 1），无落子副作用。
    """
    r, c = board.rc(move)
    for dr, dc in _DIRECTIONS:
        cnt = 1
        for sign in (1, -1):
            rr, cc = r + sign * dr, c + sign * dc
            while board.in_bounds(rr, cc) and board.stones[rr, cc] == player:
                cnt += 1
                rr += sign * dr
                cc += sign * dc
        if cnt >= 5:
            return True
    return False


def adjacent_moves(board: Board) -> list[int]:
    """邻域着法：与已有棋子切比雪夫距离 ≤ 1 的合法点；空盘时返回全部合法点。"""
    legal = board.legal_moves()
    if board.move_count == 0 or not legal:
        return legal
    n = board.size
    cand = set()
    for idx in np.flatnonzero(board.stones.flat != EMPTY):
        r, c = divmod(int(idx), n)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < n and 0 <= cc < n:
                    m = rr * n + cc
                    if m in legal:
                        cand.add(m)
    return sorted(cand) if cand else legal


@dataclass
class Node:
    """MCTS 树节点。w 与 q 均为「本节点轮到的玩家」视角。"""

    move: int | None
    parent: "Node | None" = None
    n: int = 0
    w: float = 0.0
    children: dict[int, "Node"] = field(default_factory=dict)
    untried: list[int] = field(default_factory=list)

    @property
    def q(self) -> float:
        return self.w / self.n if self.n else 0.0


def _ucb_select(node: Node, c: float) -> int:
    """UCB1 选择子节点，返回着法。node 必须有 children。"""
    ln = math.log(node.n + 1.0)
    best, best_score = None, -float("inf")
    for move, child in node.children.items():
        score = child.q + c * math.sqrt(ln / (1.0 + child.n))
        if score > best_score:
            best, best_score = move, score
    assert best is not None
    return best


def _neighbors(board: Board, idx: int) -> list[int]:
    """返回 (r,c) 的 8 邻域内合法格点序号。"""
    n = board.size
    r, c = divmod(idx, n)
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < n and 0 <= cc < n:
                out.append(rr * n + cc)
    return out


def _rollout(board: Board, rng: random.Random) -> float:
    """快速模拟至终局，返回「模拟开始时轮到的玩家」视角的价值（+1 / 0 / -1）。

    候选集用增量维护（落子时只并入其 8 邻域），避免每步全盘扫描。
    """
    player = board.to_play
    frontier: set[int] = set()
    for idx in np.flatnonzero(board.stones.flat != EMPTY):
        frontier.update(_neighbors(board, int(idx)))
    while not board.is_over:
        cur = board.to_play
        moves = [m for m in frontier if board.stones.flat[m] == EMPTY]
        if not moves:
            moves = board.legal_moves()
        opp = WHITE if cur == BLACK else BLACK
        chosen = None
        for m in moves:
            if would_win(board, m, cur):
                chosen = m
                break
        if chosen is None:
            for m in moves:
                if would_win(board, m, opp):
                    chosen = m
                    break
        if chosen is None:
            chosen = rng.choice(moves)
        board.apply(chosen)
        frontier.update(_neighbors(board, chosen))
    w = board.winner
    if w == player:
        return 1.0
    if w == DRAW:
        return 0.0
    return -1.0


def mcts_search(board: Board, sims: int, c: float = 1.4, rng: random.Random | None = None) -> Node:
    """在 board 的副本上运行 sims 次模拟，返回根节点（含子节点统计）。"""
    rng = rng or random.Random()
    root = Node(move=None)
    # 只展开邻域着法：225 路分支下全展开会使 UCB1 探索过宽（见测试用例），
    # 邻域剪枝（切比雪夫距离 ≤1）大幅提升搜索质量；空盘时退化为全部合法点。
    root.untried = adjacent_moves(board)
    for _ in range(sims):
        node = root
        b = board.copy()
        # 1) 选择：沿完全展开的路径下探
        while node.untried == [] and node.children:
            b.apply(_ucb_select(node, c))
            node = node.children[b.last_move]
        # 2) 展开：若可展开且未终局，懒展开一个子节点
        if node.untried and not b.is_over:
            move = node.untried.pop()
            b.apply(move)
            child = Node(move=move, parent=node)
            child.untried = adjacent_moves(b)   # 邻域剪枝
            node.children[move] = child
            node = child
        # 3) 模拟到终局
        value = _rollout(b, rng)
        # 4) 回传（价值按玩家视角逐层翻转）
        while node is not None:
            node.n += 1
            node.w += value
            value = -value
            node = node.parent
    return root


def best_move(board: Board, sims: int, c: float = 1.4, rng: random.Random | None = None) -> int:
    """返回最优着法。

    确定性强制着优先（自研轻量启发式，见 DESIGN.md §7.4）：
      1. 己方有一步成五 → 必走；
      2. 对方有一步成五 → 必堵；
    否则运行 MCTS 搜索（访问次数最多者；平局取 Q 高者）。
    """
    me = board.to_play
    opp = WHITE if me == BLACK else BLACK
    for m in board.legal_moves():
        if would_win(board, m, me):
            return m
    for m in board.legal_moves():
        if would_win(board, m, opp):
            return m
    root = mcts_search(board, sims, c, rng)
    if not root.children:
        moves = board.legal_moves()
        return moves[0] if moves else -1
    return max(root.children.items(), key=lambda kv: (kv[1].n, kv[1].q))[0]


# ============================================================
# 网络化 PUCT 搜索（AlphaZero 风格，见 DESIGN.md §7.2）
# ============================================================


@dataclass
class PuctNode:
    """PUCT 树节点。w 与 q 为「本节点轮到的玩家」视角；p 为父视角先验。"""

    move: int | None = None
    parent: "PuctNode | None" = None
    n: int = 0
    w: float = 0.0
    p: float = 0.0
    children: dict[int, "PuctNode"] = field(default_factory=dict)
    virtual_loss: int = 0

    @property
    def q(self) -> float:
        """虚拟损失计入价值（n=0 且挂起评估时视为输）。"""
        return (self.w - self.virtual_loss) / max(self.n, 1)


def _puct_score(child: PuctNode, parent_n: int, c_puct: float) -> float:
    q = (child.w - child.virtual_loss) / max(child.n, 1)
    u = c_puct * child.p * math.sqrt(parent_n + 1.0) / (1.0 + child.n)
    return q + u


def _to_tensor(board: Board, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(board.encode()).unsqueeze(0).to(device)


def _evaluate(board: Board, net, device: torch.device) -> tuple[np.ndarray, float]:
    """单局面评估：返回 (合法掩码后的策略分布 (225,), 价值标量)。"""
    legal = board.legal_moves()
    mask = np.zeros(board.size * board.size, dtype=np.float32)
    mask[legal] = 1.0
    with torch.no_grad():
        p, v = net.forward_masked(_to_tensor(board, device),
                                  torch.from_numpy(mask).unsqueeze(0).to(device))
    return p.cpu().numpy()[0], float(v.item())


def _evaluate_batch(boards: list[Board], net, device: torch.device):
    """批量叶子评估：一次 GPU 前向处理整批局面。"""
    xs = torch.stack([torch.from_numpy(b.encode()) for b in boards]).to(device)
    masks = np.zeros((len(boards), boards[0].size * boards[0].size), dtype=np.float32)
    for i, b in enumerate(boards):
        masks[i, b.legal_moves()] = 1.0
    with torch.no_grad():
        ps, vs = net.forward_masked(xs, torch.from_numpy(masks).to(device))
    return ps.cpu().numpy(), vs.cpu().numpy().reshape(-1)


def _terminal_value(board: Board) -> float:
    """终局价值（终局时轮到者视角）：胜 +1 / 负 -1 / 和 0。"""
    w = board.winner
    if w == DRAW:
        return 0.0
    return 1.0 if w == board.to_play else -1.0


def _puct_select(node: PuctNode, c_puct: float) -> int:
    best, best_score = None, -float("inf")
    for move, child in node.children.items():
        s = _puct_score(child, node.n, c_puct)
        if s > best_score:
            best, best_score = move, s
    assert best is not None
    return best


def _select(root: PuctNode, board: Board, c_puct: float):
    """从根下探到叶子。返回 (叶节点, 路径[根→叶], 叶局面副本)。"""
    b = board.copy()
    node = root
    path = [root]
    while node.children:
        mv = _puct_select(node, c_puct)
        b.apply(mv)
        node = node.children[mv]
        path.append(node)
    return node, path, b


def _expand(node: PuctNode, board: Board, priors: np.ndarray) -> None:
    for m in board.legal_moves():
        node.children[m] = PuctNode(move=m, parent=node, p=float(priors[m]))


def _backup(path: list[PuctNode], value: float) -> None:
    """沿路径回传（价值按玩家视角逐层翻转）。"""
    v = value
    for node in reversed(path):
        node.n += 1
        node.w += v
        v = -v


def puct_search(
    board: Board,
    net,
    sims: int,
    c_puct: float = 1.5,
    dirichlet_alpha: float = 0.3,
    dirichlet_eps: float = 0.25,
    add_root_noise: bool = True,
    batch_size: int = 64,
    device: torch.device | None = None,
    rng: random.Random | None = None,
) -> PuctNode:
    """网络化 PUCT 搜索（批量叶子评估 + 虚拟损失）。

    - 根节点先验 = 网络策略 + Dirichlet 噪声（α=0.3，ε=0.25，可关）
    - 每轮选叶：Q + c_puct·P·√N/(1+n)，挂起叶子加虚拟损失防重复
    - 叶子评估：同批攒满 batch_size 后统一 GPU 前向
    - 终局叶子直接以真实胜负回传，不评估
    """
    rng = rng or random.Random()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.eval()
    root = PuctNode()
    legal = board.legal_moves()
    with torch.no_grad():
        priors, _ = _evaluate(board, net, device)
    noise = None
    if add_root_noise:
        np_rng = np.random.default_rng(rng.randrange(1 << 30))
        noise = np_rng.dirichlet([dirichlet_alpha] * len(legal))
    for k, m in enumerate(legal):
        p = float(priors[m])
        if noise is not None:
            p = (1.0 - dirichlet_eps) * p + dirichlet_eps * float(noise[k])
        root.children[m] = PuctNode(move=m, parent=root, p=p)

    done = 0
    with torch.no_grad():
        while done < sims:
            batch: list[tuple[PuctNode, list[PuctNode], Board]] = []
            for _ in range(min(batch_size, sims - done)):
                leaf, path, leaf_board = _select(root, board, c_puct)
                if leaf_board.is_over:
                    _backup(path, _terminal_value(leaf_board))
                else:
                    leaf.virtual_loss += 1
                    batch.append((leaf, path, leaf_board))
                done += 1
            if batch:
                ps, vs = _evaluate_batch([b for _, _, b in batch], net, device)
                for (leaf, path, leaf_board), p_i, v_i in zip(batch, ps, vs):
                    _expand(leaf, leaf_board, p_i)
                    _backup(path, v_i)
    return root


def puct_visit_distribution(root: PuctNode, temp: float = 1.0, board_size: int = 15) -> np.ndarray:
    """根节点访问分布 π（225 向量）。temp ≤ 0 → one-hot（访问最多者）。"""
    pi = np.zeros(board_size * board_size, dtype=np.float32)
    if not root.children:
        return pi
    if temp <= 0:
        mv = max(root.children, key=lambda m: root.children[m].n)
        pi[mv] = 1.0
        return pi
    for m, ch in root.children.items():
        pi[m] = ch.n ** (1.0 / temp)
    s = pi.sum()
    return pi / s if s > 0 else pi


def puct_best_move(
    board: Board,
    net,
    sims: int,
    c_puct: float = 1.5,
    device: torch.device | None = None,
    rng: random.Random | None = None,
) -> int:
    """网络化最优着法：强制着（成五/堵五）优先，否则 PUCT 搜索取访问最多者。"""
    me = board.to_play
    opp = WHITE if me == BLACK else BLACK
    for m in board.legal_moves():
        if would_win(board, m, me):
            return m
    for m in board.legal_moves():
        if would_win(board, m, opp):
            return m
    root = puct_search(board, net, sims, c_puct=c_puct, rng=rng, device=device)
    if not root.children:
        moves = board.legal_moves()
        return moves[0] if moves else -1
    return max(root.children, key=lambda m: root.children[m].n)
