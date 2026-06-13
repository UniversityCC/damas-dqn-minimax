"""Agente DQN para Damas: red online + red objetivo + replay buffer + ε-greedy (issue #11).

El bucle de auto-juego que genera las partidas y llama a ``act``/``remember``/``learn``
es el issue #12. Aquí está el agente y su paso de aprendizaje con target negamax
(adecuado para un juego de suma cero de dos jugadores).
"""
from __future__ import annotations

import random

import torch
from torch import nn

from damas.engine import legal_moves, encode, State, Action
from model.q_network import QNetwork
from model.action_space import action_to_index, index_to_action, legal_action_mask
from .replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(
        self,
        gamma: float = 0.99,
        lr: float = 1e-3,
        batch_size: int = 64,
        buffer_capacity: int = 50_000,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_steps: int = 50_000,
        target_update_freq: int = 1000,
        device: str = "cpu",
        use_target: bool = True,
    ):
        self.device = torch.device(device)
        self.use_target = use_target
        self.online = QNetwork().to(self.device)
        self.target = QNetwork().to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)

        self.gamma = gamma
        self.batch_size = batch_size
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.target_update_freq = target_update_freq
        self.learn_steps = 0

    @property
    def epsilon(self) -> float:
        """ε actual: decae linealmente de eps_start a eps_end según los pasos de aprendizaje."""
        frac = min(1.0, self.learn_steps / self.eps_decay_steps)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def act(self, state: State, greedy: bool = False) -> Action:
        """Selecciona acción ε-greedy (greedy=True fuerza explotación, para evaluar)."""
        moves = legal_moves(state)
        if not moves:
            raise ValueError("No hay movimientos legales en este estado")
        if not greedy and random.random() < self.epsilon:
            return random.choice(moves)
        with torch.no_grad():
            x = torch.tensor(encode(state), dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.online(x).squeeze(0)
            mask = legal_action_mask(state).to(self.device)
            index = int(torch.argmax(q.masked_fill(~mask, float("-inf"))).item())
        return index_to_action(index, state)

    def remember(self, state: State, action: Action, reward: float,
                 next_state: State, done: bool) -> None:
        self.buffer.push(state, action_to_index(action), reward, next_state, done)

    def learn(self) -> float | None:
        """Un paso de optimización. Devuelve la pérdida, o None si aún no hay batch."""
        if len(self.buffer) < self.batch_size:
            return None
        batch = self.buffer.sample(self.batch_size)

        states = torch.tensor([encode(t.state) for t in batch],
                              dtype=torch.float32, device=self.device)
        actions = torch.tensor([t.action_index for t in batch],
                               dtype=torch.long, device=self.device)
        rewards = torch.tensor([t.reward for t in batch],
                               dtype=torch.float32, device=self.device)
        dones = torch.tensor([t.done for t in batch],
                             dtype=torch.bool, device=self.device)

        # Q(s, a) según la red online
        q_pred = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target negamax: y = r - γ · max_legales Q_bootstrap(s')
        # Si use_target=False se usa la red online para bootstrap (sin red objetivo).
        bootstrap_net = self.target if self.use_target else self.online
        with torch.no_grad():
            next_states = torch.tensor([encode(t.next_state) for t in batch],
                                       dtype=torch.float32, device=self.device)
            masks = torch.stack([legal_action_mask(t.next_state) for t in batch]).to(self.device)
            q_next = bootstrap_net(next_states).masked_fill(~masks, float("-inf"))
            best_next = q_next.max(dim=1).values
            best_next = torch.where(torch.isfinite(best_next), best_next,
                                    torch.zeros_like(best_next))  # next sin legales -> 0
            target = rewards + torch.where(dones, torch.zeros_like(rewards),
                                           -self.gamma * best_next)

        loss = nn.functional.smooth_l1_loss(q_pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_update_freq == 0:
            self.update_target()
        return float(loss.item())

    def update_target(self) -> None:
        """Copia dura de los pesos de la red online a la red objetivo.
        No-op cuando use_target=False."""
        if self.use_target:
            self.target.load_state_dict(self.online.state_dict())

    def save(self, path: str) -> None:
        """Guarda un checkpoint: red online, red objetivo, optimizador y pasos de aprendizaje."""
        torch.save(
            {
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "learn_steps": self.learn_steps,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Restaura un checkpoint guardado con save() (continúa el entrenamiento donde quedó)."""
        ckpt = torch.load(path, map_location=self.device)
        self.online.load_state_dict(ckpt["online"])
        self.target.load_state_dict(ckpt["target"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.learn_steps = ckpt["learn_steps"]
