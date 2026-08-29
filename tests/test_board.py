"""规则引擎与对局管理单元测试（pytest）。"""

import numpy as np
import pytest

from gomoku.board import (
    Board,
    BLACK,
    WHITE,
    ONGOING,
    BLACK_WIN,
    WHITE_WIN,
    DRAW,
)
from gomoku.game import Game


def play_seq(b: Board, seq: list[tuple[int, int]]) -> int:
    """按 (r, c) 序列落子，返回最后一手的 apply 结果。"""
    res = ONGOING
    for r, c in seq:
        res = b.apply(b.idx(r, c))
    return res


# ---------- 基础规则 ----------

def test_black_starts():
    assert Board().to_play == BLACK


def test_alternation():
    b = Board()
    b.apply(b.idx(7, 7))
    assert b.to_play == WHITE
    b.apply(b.idx(7, 8))
    assert b.to_play == BLACK


def test_invalid_occupied_raises():
    b = Board()
    m = b.idx(7, 7)
    b.apply(m)
    assert not b.is_valid(m)
    with pytest.raises(ValueError):
        b.apply(m)


def test_invalid_out_of_range():
    b = Board()
    assert not b.is_valid(-1)
    assert not b.is_valid(15 * 15)
    with pytest.raises(ValueError):
        b.apply(-1)
    with pytest.raises(ValueError):
        b.apply(15 * 15)


def test_legal_moves_after_over():
    b = Board()
    play_seq(b, [(7, 5), (8, 0), (7, 6), (8, 1), (7, 7), (8, 2), (7, 8), (8, 3), (7, 9)])
    assert b.is_over
    assert b.legal_moves() == []


# ---------- 胜负判定：四个方向 ----------

def test_horizontal_win():
    b = Board()
    res = play_seq(b, [(7, 5), (8, 0), (7, 6), (8, 1), (7, 7), (8, 2), (7, 8), (8, 3), (7, 9)])
    assert res == BLACK_WIN
    assert b.winner == BLACK
    assert b.is_over


def test_vertical_win():
    b = Board()
    res = play_seq(b, [(5, 7), (0, 0), (6, 7), (0, 1), (7, 7), (0, 2), (8, 7), (0, 3), (9, 7)])
    assert res == BLACK_WIN


def test_diagonal_win():
    b = Board()
    res = play_seq(b, [(3, 3), (0, 0), (4, 4), (0, 1), (5, 5), (0, 2), (6, 6), (0, 3), (7, 7)])
    assert res == BLACK_WIN


def test_anti_diagonal_win():
    b = Board()
    res = play_seq(b, [(3, 11), (0, 0), (4, 10), (0, 1), (5, 9), (0, 2), (6, 8), (0, 3), (7, 7)])
    assert res == BLACK_WIN


def test_white_wins():
    b = Board()
    res = play_seq(
        b,
        [(0, 0), (7, 5), (0, 1), (7, 6), (0, 2), (7, 7), (0, 3), (7, 8), (14, 14), (7, 9)],
    )
    assert res == WHITE_WIN
    assert b.winner == WHITE


def test_six_in_row_also_wins():
    """自由规则：超过五连同样判定胜。"""
    b = Board()
    res = play_seq(
        b,
        [(7, 5), (0, 0), (7, 6), (0, 1), (7, 7), (0, 2), (7, 8), (0, 3), (7, 9), (0, 4), (7, 10)],
    )
    assert res == BLACK_WIN


# ---------- 和棋 ----------

def _no_five_template(size: int) -> np.ndarray:
    """构造一个满盘且无五连的模板：黑=(r+2c)%4<2，任意方向同色连子 ≤2。"""
    r = np.arange(size)[:, None]
    c = np.arange(size)[None, :]
    return ((r + 2 * c) % 4 < 2).astype(np.int8) * BLACK + (
        (r + 2 * c) % 4 >= 2
    ).astype(np.int8) * WHITE


def test_draw_full_board():
    """225 手交替落满棋盘且无五连 → 和棋。模板黑白数 = 113/112，可构成合法交替对局。"""
    b = Board()
    tmpl = _no_five_template(15)
    blacks = list(zip(*np.where(tmpl == BLACK)))
    whites = list(zip(*np.where(tmpl == WHITE)))
    assert len(blacks) == 113 and len(whites) == 112
    res = ONGOING
    for i in range(225):
        cell = blacks[i // 2] if i % 2 == 0 else whites[i // 2]
        res = b.apply(b.idx(*cell))
    assert res == DRAW
    assert b.winner == DRAW
    assert b.is_over
    assert b.move_count == 225


# ---------- 悔棋 ----------

def test_undo_restores_state():
    b = Board()
    b.apply(b.idx(7, 7))
    b.apply(b.idx(8, 8))
    b.apply(b.idx(7, 8))
    assert b.to_play == WHITE
    m = b.undo()
    assert m == b.idx(7, 8)
    assert b.to_play == BLACK
    assert b.last_move == b.idx(8, 8)
    assert b.move_count == 2
    assert b.stones[b.rc(b.idx(7, 8))] == 0


def test_undo_win_then_replay():
    b = Board()
    play_seq(b, [(7, 5), (8, 0), (7, 6), (8, 1), (7, 7), (8, 2), (7, 8), (8, 3), (7, 9)])
    assert b.winner == BLACK
    b.undo()
    assert b.winner is None
    assert not b.is_over
    # 撤销黑方胜子后，轮到黑方重新行棋
    assert b.to_play == BLACK
    assert b.move_count == 8


def test_undo_empty_returns_none():
    assert Board().undo() is None


# ---------- 状态编码 ----------

def test_encode_shape_and_channels():
    b = Board()
    b.apply(b.idx(7, 7))  # 黑
    b.apply(b.idx(8, 8))  # 白，上一手
    s = b.encode()  # 视角：当前轮到黑
    assert s.shape == (4, 15, 15)
    assert s.dtype == np.float32
    assert s[0, 7, 7] == 1.0   # 己方（黑）棋子
    assert s[0, 8, 8] == 0.0
    assert s[1, 8, 8] == 1.0   # 对方（白）棋子
    assert s[2].sum() == 15 * 15  # 常数层全 1
    assert s[3, 8, 8] == 1.0   # 上一手
    assert s[3, 7, 7] == 0.0


# ---------- 对称变换 ----------

def test_symmetry_maps_are_permutations():
    maps = Board.symmetry_maps(15)
    assert len(maps) == 8
    for m in maps:
        assert m.shape == (225,)
        assert sorted(m.tolist()) == list(range(225))
    # 恒等映射 = identity
    assert (maps[0] == np.arange(225)).all()


def test_augment_matches_manual_transform():
    b = Board()
    play_seq(b, [(7, 7), (0, 0), (7, 8), (1, 1), (8, 8), (2, 2), (7, 6)])
    s = b.encode()
    pi = np.arange(225, dtype=np.float32).reshape(15, 15)  # 伪策略：位置即值
    aug = b.augment(s, pi, z=1.0)
    assert len(aug) == 8
    for i, (s2, p2, z2) in enumerate(aug):
        m = Board.symmetry_maps(15)[i]
        # 状态：dst 位置的值 = src 位置的值
        flat = s.reshape(4, -1)
        expect_s = flat[:, m].reshape(4, 15, 15)
        assert np.array_equal(s2, expect_s)
        # 策略：同样的位置重排
        expect_p = pi.reshape(-1)[m].reshape(15, 15)
        assert np.array_equal(p2, expect_p)
        assert z2 == 1.0


def test_symmetry_augment_ident():
    b = Board()
    play_seq(b, [(7, 7), (0, 0)])
    s = b.encode()
    s0, _, _ = b.augment(s, z=0.0)[0]
    assert np.array_equal(s0, s)


# ---------- key ----------

def test_key_distinguishes_states():
    b = Board()
    b.apply(b.idx(7, 7))
    k1 = b.key()
    b.apply(b.idx(8, 8))
    k2 = b.key()
    b.undo()
    assert b.key() == k1
    assert k1 != k2


# ---------- 对局管理 ----------

def test_game_play_undo_record():
    g = Game()
    g.play(g.board.idx(7, 7))
    g.play(g.board.idx(7, 8))
    assert g.moves == [g.board.idx(7, 7), g.board.idx(7, 8)]
    assert g.undo() == g.board.idx(7, 8)
    rec = g.to_record()
    assert rec["moves"] == [g.board.idx(7, 7)]
    g2 = Game.from_record(rec)
    assert g2.board.key() == g.board.key()
    assert g2.moves == g.moves
    g.reset()
    assert g.move_count == 0
