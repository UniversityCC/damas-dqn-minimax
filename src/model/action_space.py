"""Mapeo entre las acciones del motor y los índices de la cabeza de salida de la red Q.

La red emite un Q-valor por cada par (origen, destino) posible: 32 × 32 = 1024.
Este módulo traduce entre esos índices y las acciones del motor, y construye la
máscara de movimientos legales para anular los ilegales antes del argmax.
"""
from __future__ import annotations

import torch

from damas.engine import legal_moves, State, Action

NUM_SQUARES = 32
NUM_ACTIONS = NUM_SQUARES * NUM_SQUARES  # 1024


def action_to_index(action: Action) -> int:
    """Índice de la cabeza de salida para una acción (usa su origen y destino)."""
    origin, destination = action[0], action[-1]
    return origin * NUM_SQUARES + destination


def index_to_action(index: int, state: State) -> Action:
    """Acción legal de `state` que corresponde a `index`.

    Devuelve el movimiento de ``legal_moves`` cuyo (origen, destino) coincide con
    el índice. Lanza ``ValueError`` si ninguno coincide (índice ilegal en este
    estado). Cuando lleguen las capturas en cadena (#5), si dos cadenas comparten
    extremos se devuelve la primera que coincida.
    """
    for move in legal_moves(state):
        if action_to_index(move) == index:
            return move
    raise ValueError(f"El índice {index} no corresponde a un movimiento legal en este estado")


def legal_action_mask(state: State) -> torch.Tensor:
    """Vector booleano de tamaño 1024: ``True`` en las acciones legales de `state`."""
    mask = torch.zeros(NUM_ACTIONS, dtype=torch.bool)
    for move in legal_moves(state):
        mask[action_to_index(move)] = True
    return mask
