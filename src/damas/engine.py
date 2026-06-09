from __future__ import annotations
from typing import Any

State  = dict[str, Any]
Action = tuple[int, ...]

def legal_moves(state: State) -> list[Action]:
    raise NotImplementedError

def step(state: State, action: Action) -> State:
    raise NotImplementedError

def is_terminal(state: State) -> bool:
    raise NotImplementedError

def result(state: State) -> int | None:
    raise NotImplementedError

def encode(state: State) -> list[float]:
    raise NotImplementedError

def initial_state() -> State:
    board = [0] * 32
    for i in range(12):
        board[i] = 1
    for i in range(20, 32):
        board[i] = -1
    return {
        "board": board,
        "turn": 1,
        "no_capture_count": 0,
    }