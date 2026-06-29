"""Agente híbrido: búsqueda negamax con poda alfa-beta y un evaluador ENCHUFABLE.

Idea (AlphaZero en versión ligera): el DQN aporta la intuición posicional como evaluación
de hojas, y la búsqueda alfa-beta aporta el *lookahead* táctico que al DQN reactivo le
falta para vencer a Minimax de profundidad alta (d=5, d=6).

El evaluador `eval_fn(state) -> float` se mide SIEMPRE desde la perspectiva del jugador en
turno (consistente con el target negamax del DQN). Hay dos:
  - ``dqn_value_fn(net)``      : V(s) = max_a Q(s,a) de la red (net = DQNAgent.online).
  - ``heuristic_value_fn()``   : la heurística a mano envuelta a perspectiva del mover
                                  (sirve como CONTROL para aislar el aporte del aprendizaje).

La red SOLO se invoca en las hojas. La ordenación de jugadas usa la heurística rápida
(sin red) y la PV del iterative deepening, para maximizar la poda sin disparar el costo.
"""
from __future__ import annotations

import time
from collections.abc import Callable

from damas.engine import legal_moves, step, is_terminal, result, State, Action
from agents.heuristic import evaluate

_INF = float("inf")
_WIN = 1.0e6  # magnitud de un resultado terminal; domina cualquier valor de hoja


def dqn_value_fn(net) -> Callable[[State], float]:
    """Evaluador de hoja con la red: V(s)=max_a Q(s,a), perspectiva del que mueve."""
    def _v(state: State) -> float:
        return net.masked_q_values(state).max().item()
    return _v


def heuristic_value_fn() -> Callable[[State], float]:
    """Evaluador de hoja con la heurística a mano, en perspectiva del que mueve (control)."""
    def _v(state: State) -> float:
        v = evaluate(state)                       # perspectiva-rojo (absoluta)
        return v if state["turn"] == 1 else -v    # -> perspectiva del mover
    return _v


class HybridAgent:
    """Negamax + alfa-beta con evaluador enchufable. Expone ``choose_action`` (compatible
    con el torneo) y cuenta nodos por jugada para la evaluación cost-matched."""

    def __init__(self, eval_fn: Callable[[State], float], depth: int = 4,
                 time_budget: float | None = None, use_ordering: bool = True,
                 use_cache: bool = True, name: str | None = None) -> None:
        self.eval_fn = eval_fn
        self.depth = depth
        self.time_budget = time_budget          # si se da -> iterative deepening por tiempo
        self.use_ordering = use_ordering
        self.use_cache = use_cache
        self.nodes = 0                           # nodos de la última jugada (cost-matched)
        self._cache: dict = {}                   # LeafEvaluationCache: (board, turn) -> valor
        tag = f"t={time_budget}s" if time_budget else f"k={depth}"
        self.name = name or f"Hybrid({tag})"

    # ---- evaluación de hojas (con caché) y terminal ----
    def _leaf(self, state: State) -> float:
        if not self.use_cache:
            return self.eval_fn(state)
        key = (tuple(state["board"]), state["turn"])
        v = self._cache.get(key)
        if v is None:
            v = self.eval_fn(state)
            self._cache[key] = v
        return v

    @staticmethod
    def _term(state: State) -> float:
        r = result(state)                        # perspectiva del que acaba de mover
        if r is None or r == 0:
            return 0.0
        return _WIN if r == state["turn"] else -_WIN

    # ---- ordenación de jugadas (heurística rápida, SIN red) + PV ----
    def _order_key(self, state: State, action: Action) -> float:
        child = step(state, action)
        v = evaluate(child)
        v = v if child["turn"] == 1 else -v      # valor del oponente (mover de child)
        return -v                                # negamax: mayor = mejor para el mover actual

    def _ordered_moves(self, state: State, pv_move: Action | None = None) -> list[Action]:
        moves = legal_moves(state)
        if not self.use_ordering or len(moves) <= 1:
            return moves
        moves = sorted(moves, key=lambda a: self._order_key(state, a), reverse=True)
        if pv_move is not None and pv_move in moves:
            moves = [pv_move] + [m for m in moves if m != pv_move]
        return moves

    # ---- núcleo: negamax con poda alfa-beta ----
    def _negamax(self, state: State, depth: int, alpha: float, beta: float) -> float:
        self.nodes += 1
        if is_terminal(state):
            return self._term(state)
        if depth == 0:
            return self._leaf(state)
        best = -_INF
        for a in self._ordered_moves(state):
            v = -self._negamax(step(state, a), depth - 1, -beta, -alpha)
            if v > best:
                best = v
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break                            # poda alfa-beta
        return best

    def _search(self, state: State, depth: int, pv_move: Action | None = None):
        """Búsqueda raíz a profundidad ``depth``. Devuelve (mejor_accion, valor)."""
        best_a, best_v, alpha = None, -_INF, -_INF
        for a in self._ordered_moves(state, pv_move):
            v = -self._negamax(step(state, a), depth - 1, -_INF, -alpha)
            if v > best_v:
                best_v, best_a = v, a
            if v > alpha:
                alpha = v
        return best_a, best_v

    # ---- interfaz pública ----
    def choose_action(self, state: State) -> Action | None:
        self.nodes = 0
        moves = legal_moves(state)
        if not moves:
            return None
        if self.time_budget is None:
            best_a, _ = self._search(state, self.depth)
            return best_a
        # iterative deepening por tiempo: completa cada profundidad, usa la PV para ordenar
        t0 = time.perf_counter()
        best_a, pv, d = moves[0], None, 1
        while time.perf_counter() - t0 < self.time_budget:
            a, _ = self._search(state, d, pv_move=pv)
            if a is not None:
                best_a, pv = a, a
            d += 1
        return best_a
