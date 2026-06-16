"""
Evaluacion DQN vs Minimax a distintas profundidades (issue #15).

Carga un checkpoint del DQN y lo enfrenta contra MinimaxAgent a profundidades
3, 4, 5 y 6. Cada confrontacion corre N partidas con colores alternados para
eliminar el sesgo de quien empieza. El resultado es una matriz de win-rate por
profundidad que sirve como entregable directo para el paper.

Uso
---
  python src/eval/run.py --checkpoint models/checkpoint_final.pt
  python src/eval/run.py --checkpoint models/checkpoint_final.pt --games 40
  python src/eval/run.py --checkpoint models/checkpoint_final.pt --depths 3 4 5 6 --out data/eval.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from damas.engine import legal_moves
from agents.minimax import MinimaxAgent
from tournament.tournament import run_tournament


# ---------------------------------------------------------------------------
# Wrapper de nombre para DQNAgent
# ---------------------------------------------------------------------------

class _NamedDQN:
    """Envuelve DQNAgent con un nombre legible y fuerza modo greedy."""

    def __init__(self, agent: DQNAgent, label: str = "DQN") -> None:
        self._agent = agent
        self.name   = label

    def choose_action(self, state):
        return self._agent.act(state, greedy=True)


# ---------------------------------------------------------------------------
# Estructura de resultados
# ---------------------------------------------------------------------------

@dataclass
class DepthResult:
    depth:       int
    dqn_wins:    int
    mm_wins:     int
    draws:       int
    total:       int
    win_rate:    float   # fraccion de victorias del DQN sobre total


# ---------------------------------------------------------------------------
# Evaluacion por profundidad
# ---------------------------------------------------------------------------

def evaluate_vs_depth(
    dqn_agent: _NamedDQN,
    depth: int,
    n_games: int,
    verbose: bool,
) -> DepthResult:
    minimax = MinimaxAgent(depth=depth, player=1)
    stats = run_tournament(
        dqn_agent,
        minimax,
        n_games=n_games,
        verbose=verbose,
    )
    return DepthResult(
        depth=depth,
        dqn_wins=stats.agent_a_wins,
        mm_wins=stats.agent_b_wins,
        draws=stats.draws,
        total=stats.total,
        win_rate=stats.agent_a_wins / stats.total if stats.total > 0 else 0.0,
    )


# ---------------------------------------------------------------------------
# Runner principal
# ---------------------------------------------------------------------------

def run_eval(
    checkpoint: str,
    depths:     list[int]   = None,
    n_games:    int         = 20,
    csv_path:   str | None  = None,
    verbose:    bool        = True,
) -> list[DepthResult]:
    if depths is None:
        depths = [3, 4, 5, 6]

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(
            f"No se encontro el checkpoint: {checkpoint}\n"
            "Ejecuta selfplay.py primero para generar el modelo entrenado."
        )

    if verbose:
        print(f"Cargando checkpoint: {checkpoint}")

    from agents.dqn import DQNAgent
    agent = DQNAgent()
    agent.load(checkpoint)
    dqn = _NamedDQN(agent, label=f"DQN(ep{agent.learn_steps})")

    if verbose:
        print(f"Modelo listo — {agent.learn_steps} pasos de aprendizaje")
        print(f"Partidas por profundidad: {n_games}  |  Profundidades: {depths}\n")

    results: list[DepthResult] = []
    t0 = time.time()

    for d in depths:
        if verbose:
            print(f"  Profundidad {d}:")
        r = evaluate_vs_depth(dqn, depth=d, n_games=n_games, verbose=verbose)
        results.append(r)

    elapsed = time.time() - t0

    _print_matrix(results, dqn.name)

    if verbose:
        print(f"\nTiempo total: {elapsed:.1f}s")

    if csv_path:
        _save_csv(results, csv_path, dqn.name)
        if verbose:
            print(f"Resultados guardados en: {csv_path}")

    return results


# ---------------------------------------------------------------------------
# Salidas
# ---------------------------------------------------------------------------

def _print_matrix(results: list[DepthResult], dqn_label: str) -> None:
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  Matriz win-rate: {dqn_label} vs Minimax")
    print(sep)
    print(f"  {'Prof':>5}  {'DQN gana':>9}  {'MM gana':>8}  {'Empates':>8}  {'Win-rate':>9}")
    print(sep)
    for r in results:
        print(
            f"  {r.depth:>5}  {r.dqn_wins:>9}  {r.mm_wins:>8}  "
            f"{r.draws:>8}  {r.win_rate * 100:>8.1f}%"
        )
    print(sep)


def _save_csv(results: list[DepthResult], path: str, dqn_label: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dqn_label", "depth", "dqn_wins", "mm_wins",
                        "draws", "total", "win_rate"],
        )
        writer.writeheader()
        for r in results:
            writer.writerow({
                "dqn_label": dqn_label,
                "depth":     r.depth,
                "dqn_wins":  r.dqn_wins,
                "mm_wins":   r.mm_wins,
                "draws":     r.draws,
                "total":     r.total,
                "win_rate":  round(r.win_rate, 4),
            })


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluacion DQN vs Minimax a distintas profundidades (issue #15)"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="models/checkpoint_final.pt",
        help="Ruta al checkpoint del DQN (default: models/checkpoint_final.pt)",
    )
    parser.add_argument(
        "--depths", type=int, nargs="+", default=[3, 4, 5, 6],
        help="Profundidades de Minimax a evaluar (default: 3 4 5 6)",
    )
    parser.add_argument(
        "--games", type=int, default=20,
        help="Partidas por profundidad (default: 20, usar par para equilibrar colores)",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Ruta del CSV de resultados (ej: results/eval_depth.csv)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suprime el detalle de cada partida",
    )
    args = parser.parse_args()

    run_eval(
        checkpoint=args.checkpoint,
        depths=args.depths,
        n_games=args.games,
        csv_path=args.out,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
