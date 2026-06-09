"""Memoria de experiencia (replay buffer) del agente DQN (issue #11).

Guarda transiciones con los estados CRUDOS del motor (dict ligero). El ``encode``
y la máscara de legalidad se recalculan al muestrear, para mantener la memoria
liviana y desacoplar el buffer de PyTorch.
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

from damas.engine import State


@dataclass
class Transition:
    state: State
    action_index: int
    reward: float
    next_state: State
    done: bool


class ReplayBuffer:
    """Cola circular de transiciones con muestreo aleatorio uniforme."""

    def __init__(self, capacity: int = 50_000):
        self._buffer: deque[Transition] = deque(maxlen=capacity)

    def push(self, state: State, action_index: int, reward: float,
             next_state: State, done: bool) -> None:
        self._buffer.append(Transition(state, action_index, reward, next_state, done))

    def sample(self, batch_size: int) -> list[Transition]:
        """Devuelve `batch_size` transiciones al azar (sin reemplazo)."""
        return random.sample(self._buffer, batch_size)

    def __len__(self) -> int:
        return len(self._buffer)
