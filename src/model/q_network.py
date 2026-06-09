"""Red Q del agente DQN para Damas.

MLP que recibe la codificación del tablero (160 floats de ``damas.encode``) y
emite un Q-valor por cada par (origen, destino) posible (1024). La selección de
la mejor acción aplica la máscara de legalidad antes del argmax.

La exploración epsilon-greedy, el replay buffer y la red objetivo viven en el
agente DQN (issue #11); aquí solo está la arquitectura y el forward.
"""
from __future__ import annotations

import torch
from torch import nn

from damas.engine import encode, State, Action
from .action_space import NUM_ACTIONS, legal_action_mask, index_to_action

INPUT_SIZE = 160  # 5 canales × 32 casillas (salida de damas.encode)


class QNetwork(nn.Module):
    """Aproximador de la función de valor-acción Q(s, ·)."""

    def __init__(self, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_SIZE, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, NUM_ACTIONS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: tensor (batch, 160) → Q-valores (batch, 1024)."""
        return self.net(x)

    def state_to_tensor(self, state: State) -> torch.Tensor:
        """Convierte un estado del motor en el tensor de entrada (1, 160)."""
        return torch.tensor(encode(state), dtype=torch.float32).unsqueeze(0)

    @torch.no_grad()
    def masked_q_values(self, state: State) -> torch.Tensor:
        """Q-valores (1024,) con los movimientos ilegales puestos a ``-inf``."""
        q = self.forward(self.state_to_tensor(state)).squeeze(0)
        mask = legal_action_mask(state)
        return q.masked_fill(~mask, float("-inf"))

    @torch.no_grad()
    def best_action(self, state: State) -> Action:
        """Movimiento legal con mayor Q (greedy). Para evaluar la red; la política
        de entrenamiento (epsilon-greedy) es del agente (#11)."""
        q = self.masked_q_values(state)
        index = int(torch.argmax(q).item())
        return index_to_action(index, state)
