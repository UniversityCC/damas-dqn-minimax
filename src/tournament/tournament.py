"""
Orquestador de torneo de ronda completa entre dos agentes de Damas.

Soporta cualquier agente que tenga choose_action(state) o act(state, greedy).
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Any

from damas import initial_state, step, is_terminal, result, State, Action


# ---------------------------------------------------------------------------
# Interfaz unificada de agente
# ---------------------------------------------------------------------------

def _get_action(agent: Any, state: State) -> Action | None:
    """Obtiene una acción de cualquier agente (MinimaxAgent o DQNAgent)."""
    if hasattr(agent, "choose_action"):
        return agent.choose_action(state)
    if hasattr(agent, "act"):
        return agent.act(state, greedy=True)
    raise AttributeError(
        f"El agente {type(agent).__name__} no tiene choose_action ni act"
    )


def _agent_name(agent: Any) -> str:
    return getattr(agent, "name", type(agent).__name__)


# ---------------------------------------------------------------------------
# Estructura de resultados
# ---------------------------------------------------------------------------

@dataclass
class GameRecord:
    game:        int
    red_agent:   str
    black_agent: str
    result:      int   # 1=rojo gana, -1=negro gana, 0=empate
    half_moves:  int


@dataclass
class TournamentStats:
    agent_a_name: str
    agent_b_name: str
    games: list[GameRecord] = field(default_factory=list)

    # ---- contadores ----

    def _wins(self, name: str) -> int:
        return sum(
            1 for g in self.games
            if (g.red_agent == name and g.result == 1)
            or (g.black_agent == name and g.result == -1)
        )

    @property
    def agent_a_wins(self) -> int:
        return self._wins(self.agent_a_name)

    @property
    def agent_b_wins(self) -> int:
        return self._wins(self.agent_b_name)

    @property
    def draws(self) -> int:
        return sum(1 for g in self.games if g.result == 0)

    @property
    def total(self) -> int:
        return len(self.games)

    # ---- salidas ----

    def save_csv(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["game", "red_agent", "black_agent", "result", "half_moves"],
            )
            writer.writeheader()
            for g in self.games:
                writer.writerow({
                    "game":        g.game,
                    "red_agent":   g.red_agent,
                    "black_agent": g.black_agent,
                    "result":      g.result,
                    "half_moves":  g.half_moves,
                })

    def print_summary(self) -> None:
        n = self.total or 1
        sep = "=" * 52
        print(f"\n{sep}")
        print(f"  Torneo: {self.agent_a_name} vs {self.agent_b_name}")
        print(f"  Partidas jugadas: {self.total}")
        print(sep)
        for name, wins in [
            (self.agent_a_name, self.agent_a_wins),
            (self.agent_b_name, self.agent_b_wins),
            ("Empates",         self.draws),
        ]:
            bar = "█" * wins
            print(f"  {name:<26} {wins:>3}  ({wins / n * 100:5.1f}%)  {bar}")
        print(sep)


# ---------------------------------------------------------------------------
# Lógica de partida y torneo
# ---------------------------------------------------------------------------

def play_one_game(
    agent_red:   Any,
    agent_black: Any,
    max_half_moves: int = 300,
) -> tuple[int, int]:
    """Juega una partida completa. Devuelve (result, half_moves_played)."""
    state = initial_state()
    half_move = 0
    for half_move in range(max_half_moves):
        if is_terminal(state):
            break
        agent  = agent_red if state["turn"] == 1 else agent_black
        action = _get_action(agent, state)
        if action is None:
            break
        state = step(state, action)
    r = result(state)
    return (r if r is not None else 0), half_move + 1


def run_tournament(
    agent_a: Any,
    agent_b: Any,
    n_games:        int  = 20,
    max_half_moves: int  = 300,
    csv_path:       str | None = None,
    verbose:        bool = True,
) -> TournamentStats:
    """
    Torneo de ronda completa: agent_a y agent_b alternan colores cada partida.

    Parámetros
    ----------
    agent_a / agent_b : cualquier agente con choose_action o act
    n_games           : número total de partidas
    csv_path          : si se indica, guarda los resultados en ese CSV
    verbose           : imprime el resultado de cada partida
    """
    name_a = _agent_name(agent_a)
    name_b = _agent_name(agent_b)
    stats  = TournamentStats(agent_a_name=name_a, agent_b_name=name_b)

    for i in range(1, n_games + 1):
        # Alternar colores: partidas impares → a=rojo, pares → b=rojo
        if i % 2 == 1:
            red, black           = agent_a, agent_b
            red_name, black_name = name_a,  name_b
        else:
            red, black           = agent_b, agent_a
            red_name, black_name = name_b,  name_a

        res, moves = play_one_game(red, black, max_half_moves)

        record = GameRecord(
            game=i,
            red_agent=red_name,
            black_agent=black_name,
            result=res,
            half_moves=moves,
        )
        stats.games.append(record)

        if verbose:
            if res == 1:
                winner = red_name
            elif res == -1:
                winner = black_name
            else:
                winner = "Empate"
            print(
                f"  [{i:>2}/{n_games}]  rojo={red_name:<22} "
                f"negro={black_name:<22}  → {winner}  ({moves} t)"
            )

    if csv_path:
        stats.save_csv(csv_path)
        if verbose:
            print(f"\nResultados guardados en: {csv_path}")

    stats.print_summary()
    return stats
