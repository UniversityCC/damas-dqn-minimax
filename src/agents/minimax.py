from __future__ import annotations
from damas import legal_moves, step, is_terminal, result, State, Action
from agents.heuristic import evaluate

_INF = float("inf")


class MinimaxAgent:
    """
    Minimax agent with alpha-beta pruning.

    Parameters
    ----------
    depth  : search depth (clamped to [3, 6])
    player : 1 = red (maximiser), -1 = black (minimiser)
    """

    def __init__(self, depth: int = 4, player: int = 1) -> None:
        self.depth  = max(3, min(6, depth))
        self.player = player
        self.name   = f"Minimax(d={self.depth})"

    def choose_action(self, state: State) -> Action | None:
        """Return the best legal action for whoever's turn it is in *state*.

        Uses state["turn"] (not self.player) so the agent plays correctly
        regardless of which colour it was assigned — needed for tournaments
        where agents alternate colours across games.
        """
        moves = legal_moves(state)
        if not moves:
            return None

        best_action = moves[0]
        alpha, beta = -_INF, _INF
        maximizing  = state["turn"] == 1   # red maximises, black minimises

        if maximizing:
            best_val = -_INF
            for action in moves:
                val = self.minimax(step(state, action), self.depth - 1, alpha, beta)
                if val > best_val:
                    best_val = val
                    best_action = action
                alpha = max(alpha, best_val)
        else:
            best_val = _INF
            for action in moves:
                val = self.minimax(step(state, action), self.depth - 1, alpha, beta)
                if val < best_val:
                    best_val = val
                    best_action = action
                beta = min(beta, best_val)

        return best_action

    def minimax(self, state: State, depth: int, alpha: float, beta: float) -> float:
        """Alpha-beta minimax. Red is always the maximising player."""
        if is_terminal(state):
            r = result(state)
            return (r if r is not None else 0) * 1_000.0

        if depth == 0:
            return evaluate(state)

        moves = legal_moves(state)
        maximizing = state["turn"] == 1

        if maximizing:
            value = -_INF
            for action in moves:
                value = max(value, self.minimax(step(state, action), depth - 1, alpha, beta))
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return value
        else:
            value = _INF
            for action in moves:
                value = min(value, self.minimax(step(state, action), depth - 1, alpha, beta))
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value
