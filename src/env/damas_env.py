# TODO
from __future__ import annotations

import sys
import os
from typing import Any, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# motor
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, ".."))

from damas.engine import (
    initial_state,
    legal_moves,
    step as engine_step,
    is_terminal,
    result as engine_result,
    encode,
    _promotion_row,
    _JUMP_OVER,
)

def _build_action_catalogue() -> list[tuple[int, ...]]:
    catalogue: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()

    from damas.engine import NEIGHBORS

    # Movimientos simples 
    for sq in range(32):
        for d in ("ul", "ur", "dl", "dr"):
            nb = NEIGHBORS[sq].get(d)
            if nb is not None:
                t = (sq, nb)
                if t not in seen:
                    seen.add(t)
                    catalogue.append(t)
    all_mids = set(_JUMP_OVER.keys())  # (src, dst) pairs

    def _reachable(sq: int, path: tuple[int, ...], depth: int) -> None:
        if depth == 0:
            return
        from damas.engine import NEIGHBORS as _NB
        for d in ("ul", "ur", "dl", "dr"):
            mid = _NB[sq].get(d)
            if mid is None:
                continue
            land = _NB[mid].get(d)
            if land is None:
                continue
            if (sq, land) not in all_mids:
                continue
            new_path = path + (land,)
            if new_path not in seen:
                seen.add(new_path)
                catalogue.append(new_path)
            _reachable(land, new_path, depth - 1)

    for sq in range(32):
        _reachable(sq, (sq,), 5)

    return catalogue


_ACTION_CATALOGUE: list[tuple[int, ...]] = _build_action_catalogue()
_ACTION_INDEX: dict[tuple[int, ...], int] = {
    a: i for i, a in enumerate(_ACTION_CATALOGUE)
}
N_ACTIONS = len(_ACTION_CATALOGUE)

OBS_DIM = 160  # 5 canales × 32


class DamasEnv(gym.Env):

    metadata = {"render_modes": ["ansi"], "render_fps": 1}
    observation_space = spaces.Box(
        low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
    )
    action_space = spaces.Discrete(N_ACTIONS)

    def __init__(
        self,
        capture_reward: float = 0.0,
        king_reward: float = 0.0,
        illegal_action_mode: str = "raise",
        render_mode: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.capture_reward = capture_reward
        self.king_reward = king_reward
        self.illegal_action_mode = illegal_action_mode
        self.render_mode = render_mode

        self._state: dict = {}
        self._done: bool = True
        self._info: dict = {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        if options is not None and "state" in options:
            self._state = options["state"]
        else:
            self._state = initial_state()

        self._done = False
        obs = self._observe()
        self._info = self._build_info()
        return obs, self._info

    def step(
        self, action: int | tuple[int, ...]
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._done:
            raise RuntimeError("Call reset() before step().")
        if isinstance(action, (tuple, list)):
            action_tuple = tuple(action)
        else:
            if not (0 <= action < N_ACTIONS):
                return self._handle_illegal(f"Action index {action} out of range")
            action_tuple = _ACTION_CATALOGUE[action]
        legal = legal_moves(self._state)
        if action_tuple not in legal:
            return self._handle_illegal(f"Illegal action: {action_tuple}")

        reward = 0.0
        board_before = list(self._state["board"])

        if self.capture_reward != 0.0:
            captured_count = self._count_captures(action_tuple)
            reward += self.capture_reward * captured_count

        if self.king_reward != 0.0:
            reward += self.king_reward * self._count_promotions(
                action_tuple, board_before
            )
        self._state = engine_step(self._state, action_tuple)

        terminated = is_terminal(self._state)
        if terminated:
            res = engine_result(self._state)
            if res is None:
                terminal_reward = 0.0
            elif res == 0:
                terminal_reward = 0.0
            else:
                prev_turn = -self._state["turn"]
                terminal_reward = 1.0 if res == prev_turn else -1.0
            reward += terminal_reward
            self._done = True

        obs = self._observe()
        self._info = self._build_info()
        return obs, float(reward), terminated, False, self._info

    def render(self) -> Optional[str]:
        rendered = _render_ansi(self._state)
        if self.render_mode == "ansi":
            print(rendered)
        return rendered

    def legal_action_mask(self) -> np.ndarray:
        mask = np.zeros(N_ACTIONS, dtype=bool)
        for a in legal_moves(self._state):
            idx = _ACTION_INDEX.get(a)
            if idx is not None:
                mask[idx] = True
        return mask

    def action_to_tuple(self, idx: int) -> tuple[int, ...]:
        return _ACTION_CATALOGUE[idx]

    def tuple_to_action(self, t: tuple[int, ...]) -> int:
        if t not in _ACTION_INDEX:
            raise KeyError(f"Action {t} not in catalogue")
        return _ACTION_INDEX[t]

    @property
    def state(self) -> dict:
        return self._state


    def _observe(self) -> np.ndarray:
        return np.array(encode(self._state), dtype=np.float32)

    def _build_info(self) -> dict:
        legal = legal_moves(self._state)
        legal_indices = [
            _ACTION_INDEX[a] for a in legal if a in _ACTION_INDEX
        ]
        return {
            "legal_actions": legal_indices,
            "legal_tuples": legal,
            "turn": self._state["turn"],
            "no_capture_count": self._state.get("no_capture_count", 0),
        }

    def _handle_illegal(
        self, msg: str
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self.illegal_action_mode == "raise":
            raise ValueError(msg)
        # "lose" mode
        self._done = True
        obs = self._observe()
        return obs, -1.0, True, False, self._build_info()

    def _count_captures(self, action: tuple[int, ...]) -> int:
        count = 0
        for i in range(len(action) - 1):
            if (action[i], action[i + 1]) in _JUMP_OVER:
                count += 1
        return count

    def _count_promotions(
        self, action: tuple[int, ...], board_before: list[int]
    ) -> int:
        src = action[0]
        dst = action[-1]
        piece = board_before[src]
        if abs(piece) == 1:
            turn = 1 if piece > 0 else -1
            if dst in _promotion_row(turn):
                return 1
        return 0 

_PIECE_CHAR = {0: ".", 1: "r", -1: "b", 2: "R", -2: "B"}
_COL_LABELS = "  a b c d e f g h"
def _render_ansi(state: dict) -> str:
    board = state["board"]
    turn = state["turn"]
    lines = [_COL_LABELS]
    sq = 0
    for row in range(8):
        cells = []
        for col in range(8):
            even_row = row % 2 == 0
            if (col % 2) == (row % 2):
                cells.append(" ")
            else:
                cells.append(_PIECE_CHAR.get(board[sq], "?"))
                sq += 1
        lines.append(f"{row + 1} {''.join(cells)}")
    lines.append(f"Turn: {'RED (+1)' if turn == 1 else 'BLACK (-1)'}")
    no_cap = state.get("no_capture_count", 0)
    lines.append(f"No-capture count: {no_cap}/80")
    return "\n".join(lines)

def register() -> None:
    if "Damas-v0" not in gym.envs.registry:
        gym.register(
            id="Damas-v0",
            entry_point="env.damas_env:DamasEnv",
        )