"""Evaluación ESTOCÁSTICA de un checkpoint DQN vs Minimax.

Motivo: ``eval/run.py`` juega el DQN en modo greedy y Minimax es determinista, así
que cada partida es idéntica y solo varía el color -> 2 "muestras" por checkpoint.
Aquí el DQN juega con un ε pequeño que introduce variedad real, dando una
distribución de resultados estadísticamente significativa (issue #42).

Uso:
    python src/eval/stochastic.py --checkpoint models/checkpoint_final.pt --depth 3 --games 30
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

# Path setup: permite ejecutar como `python src/eval/stochastic.py`
_HERE = Path(__file__).resolve().parent
for _c in (_HERE.parent, _HERE):
    if (_c / "damas").is_dir():
        sys.path.insert(0, str(_c))
        break

from damas.engine import initial_state, step, is_terminal, result
from agents.dqn import DQNAgent
from agents.minimax import MinimaxAgent


def play_one(agent: DQNAgent, mm: MinimaxAgent, dqn_color: int, max_half: int) -> str:
    """Una partida DQN (ε-greedy) vs Minimax. Devuelve 'W'/'L'/'D' desde el DQN."""
    s = initial_state()
    half = 0
    while not is_terminal(s) and half < max_half:
        a = agent.act(s, greedy=False) if s["turn"] == dqn_color else mm.choose_action(s)
        s = step(s, a)
        half += 1
    res = result(s) if is_terminal(s) else 0
    if res == dqn_color:
        return "W"
    if res in (0, None):
        return "D"
    return "L"


def evaluate_stochastic(agent: DQNAgent, depth: int = 3, games: int = 30,
                        eps: float = 0.05, max_half: int = 400,
                        seed_base: int = 1000) -> dict:
    """Juega ``games`` partidas (alternando color, semillas pareadas) vs Minimax(depth).

    Fija ε constante = ``eps`` para introducir variedad. Devuelve W/L/D y win-rate.
    """
    agent.eps_start = agent.eps_end = eps          # ε constante para variar partidas
    mm = MinimaxAgent(depth=depth)
    w = l = d = 0
    for g in range(games):
        random.seed(seed_base + g)                 # misma apertura por índice -> pareado
        r = play_one(agent, mm, 1 if g % 2 == 0 else -1, max_half)
        w += r == "W"; l += r == "L"; d += r == "D"
    return {"wins": w, "losses": l, "draws": d, "games": games,
            "win_rate": w / games, "no_loss_rate": (w + d) / games}


def main() -> None:
    p = argparse.ArgumentParser(description="Eval estocástica de un checkpoint DQN vs Minimax")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--games", type=int, default=30)
    p.add_argument("--eps", type=float, default=0.05)
    p.add_argument("--max-half", type=int, default=400)
    p.add_argument("--seed-base", type=int, default=1000)
    args = p.parse_args()

    agent = DQNAgent()
    agent.load(args.checkpoint)
    r = evaluate_stochastic(agent, args.depth, args.games, args.eps,
                            args.max_half, args.seed_base)
    print(f"vs Minimax d={args.depth}: W/L/D = {r['wins']}/{r['losses']}/{r['draws']} "
          f"| win {100*r['win_rate']:.0f}% (no-loss {100*r['no_loss_rate']:.0f}%)")


if __name__ == "__main__":
    main()
