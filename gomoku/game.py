"""对局管理：回合流转、悔棋、对局记录与回放序列化。

与界面、AI 解耦：GUI 与 AI 都只依赖 Game / Board 的公开接口。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .board import Board


@dataclass
class Game:
    """一局五子棋：封装 Board 与走子历史。"""

    size: int = 15
    board: Board = field(init=False)
    moves: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.board = Board(self.size)

    def play(self, move: int) -> int:
        """落子并记录，返回规则引擎的结果码。"""
        result = self.board.apply(move)
        self.moves.append(move)
        return result

    def undo(self) -> int | None:
        """悔棋一步，返回被撤销的落子（无可悔返回 None）。"""
        if self.board.undo() is not None:
            return self.moves.pop()
        return None

    def reset(self) -> None:
        self.board.reset()
        self.moves.clear()

    @property
    def over(self) -> bool:
        return self.board.is_over

    @property
    def winner(self) -> int | None:
        return self.board.winner

    @property
    def move_count(self) -> int:
        return self.board.move_count

    # ---------- 对局记录（回放） ----------
    def to_record(self) -> dict:
        return {"size": self.size, "moves": self.moves, "result": self.winner}

    @classmethod
    def from_record(cls, record: dict) -> "Game":
        g = cls(size=record.get("size", 15))
        for move in record["moves"]:
            g.play(move)
        return g

    # ---------- 记录存取（JSON） ----------
    @staticmethod
    def save_record(record: dict, path: str) -> None:
        import json
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=1)

    @staticmethod
    def load_record(path: str) -> dict:
        import json

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def save_game(game: "Game", path: str) -> None:
        Game.save_record(game.to_record(), path)
