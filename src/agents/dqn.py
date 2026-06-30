"""Agente DQN para Damas: red online + red objetivo + replay buffer + ε-greedy (issue #11).

El bucle de auto-juego que genera las partidas y llama a ``act``/``remember``/``learn``
es el issue #12. Aquí está el agente y su paso de aprendizaje con target negamax
(adecuado para un juego de suma cero de dos jugadores).
"""
from __future__ import annotations

import random

import torch
from torch import nn

from damas.engine import legal_moves, encode, is_terminal, State, Action
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
        double_dqn: bool = True,
        soft_tau: float | None = None,
        search_target_depth: int | None = None,
        hidden: int | tuple[int, ...] = (512, 512),
    ):
        self.device = torch.device(device)
        self.use_target = use_target
        self.double_dqn = double_dqn
        self.soft_tau = soft_tau
        # Maestro A: si se define, el target bootstrap usa una BÚSQUEDA negamax de esta
        # profundidad en s' (la red como evaluador de hojas) en vez de max Q(s'),
        # inyectando lookahead táctico en la señal de aprendizaje (estilo expert iteration).
        self.search_target_depth = search_target_depth
        self._target_searcher = None             # se construye perezosamente
        self.hidden = hidden
        self.online = QNetwork(hidden).to(self.device)
        self.target = QNetwork(hidden).to(self.device)
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

        # Target negamax: y = r - γ · best_next, con best_next el valor del mejor
        # movimiento legal en s' (Double DQN si está activado; ver _bootstrap_values).
        with torch.no_grad():
            if self.search_target_depth:
                best_next = self._search_bootstrap_values([t.next_state for t in batch])
            else:
                next_states = torch.tensor([encode(t.next_state) for t in batch],
                                           dtype=torch.float32, device=self.device)
                masks = torch.stack([legal_action_mask(t.next_state) for t in batch]).to(self.device)
                best_next = self._bootstrap_values(next_states, masks)
            target = rewards + torch.where(dones, torch.zeros_like(rewards),
                                           -self.gamma * best_next)

        loss = nn.functional.smooth_l1_loss(q_pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_steps += 1
        if self.soft_tau is not None:
            self.update_target()                                 # soft update (Polyak) cada paso
        elif self.learn_steps % self.target_update_freq == 0:
            self.update_target()                                 # copia dura periódica
        return float(loss.item())

    def _bootstrap_values(self, next_states: torch.Tensor,
                          masks: torch.Tensor) -> torch.Tensor:
        """Valor del mejor movimiento legal en cada s' para el bootstrap negamax.

        Con ``double_dqn`` (y red objetivo activa) la acción se SELECCIONA con la red
        online y se EVALÚA con la red objetivo (Double DQN), reduciendo la
        sobreestimación. En otro caso se usa el máximo de la red de bootstrap.
        Los estados sin movimientos legales aportan 0.
        """
        has_legal = masks.any(dim=1)
        if self.double_dqn and self.use_target:
            q_online = self.online(next_states).masked_fill(~masks, float("-inf"))
            a_star = q_online.argmax(dim=1, keepdim=True)        # selección: red online
            best = self.target(next_states).gather(1, a_star).squeeze(1)  # evaluación: red objetivo
        else:
            net = self.target if self.use_target else self.online
            q_next = net(next_states).masked_fill(~masks, float("-inf"))
            best = q_next.max(dim=1).values
        return torch.where(has_legal, best, torch.zeros_like(best))

    def _make_target_searcher(self):
        """Buscador negamax (el del híbrido, ya testeado) con la red como evaluador de
        hojas, pero con terminales en escala ±1 para no descentrar la escala del target."""
        from agents.hybrid import HybridAgent, dqn_value_fn
        from damas.engine import result as _result

        class _TargetSearcher(HybridAgent):
            @staticmethod
            def _term(state):
                r = _result(state)                       # perspectiva del que acaba de mover
                if r is None or r == 0:
                    return 0.0
                return 1.0 if r == state["turn"] else -1.0   # ±1 (escala de la recompensa)

        return _TargetSearcher(eval_fn=dqn_value_fn(self.online),
                               depth=self.search_target_depth,
                               use_ordering=True, use_cache=True)

    def _search_bootstrap_values(self, next_states) -> torch.Tensor:
        """V(s') por BÚSQUEDA negamax a profundidad ``search_target_depth`` (Maestro A).

        Es el valor de s' desde la perspectiva de su jugador en turno, igual que el
        ``max Q(s')`` que reemplaza, pero con lookahead táctico de varias jugadas.
        """
        if self._target_searcher is None:
            self._target_searcher = self._make_target_searcher()
        s = self._target_searcher
        s._cache.clear()        # la red es fija dentro del paso -> caché válida en todo el batch
        d = self.search_target_depth
        vals = []
        for st in next_states:
            if is_terminal(st) or not legal_moves(st):
                vals.append(0.0)
            else:
                vals.append(s._negamax(st, d, float("-inf"), float("inf")))
        return torch.tensor(vals, dtype=torch.float32, device=self.device)

    def update_target(self) -> None:
        """Sincroniza la red objetivo con la online.

        Con ``soft_tau`` aplica Polyak (θ_target ← τ·θ_online + (1-τ)·θ_target),
        pensado para llamarse en cada paso; si no, copia dura. No-op si use_target=False.
        """
        if not self.use_target:
            return
        if self.soft_tau is not None:
            with torch.no_grad():
                for tp, op in zip(self.target.parameters(), self.online.parameters()):
                    tp.mul_(1.0 - self.soft_tau).add_(self.soft_tau * op.data)
        else:
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
