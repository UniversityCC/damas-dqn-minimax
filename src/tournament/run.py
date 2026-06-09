"""
Script de torneo de ronda completa entre agentes Minimax.

Uso
---
  python src/tournament/run.py
  python src/tournament/run.py --games 20 --depth-a 3 --depth-b 4
  python src/tournament/run.py --games 10 --out data/my_results.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Añade src/ al path para importar damas, agents y tournament
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.minimax import MinimaxAgent
from tournament.tournament import run_tournament


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Torneo de ronda completa entre agentes Minimax de Damas"
    )
    parser.add_argument(
        "--games",   type=int, default=10,
        help="Número de partidas (default: 10)",
    )
    parser.add_argument(
        "--depth-a", type=int, default=3,
        help="Profundidad del agente A (default: 3)",
    )
    parser.add_argument(
        "--depth-b", type=int, default=4,
        help="Profundidad del agente B (default: 4)",
    )
    parser.add_argument(
        "--out",     type=str, default="data/tournament_results.csv",
        help="Ruta del CSV de resultados (default: data/tournament_results.csv)",
    )
    args = parser.parse_args()

    agent_a = MinimaxAgent(depth=args.depth_a, player=1)
    agent_b = MinimaxAgent(depth=args.depth_b, player=1)

    print(f"Torneo: {agent_a.name} vs {agent_b.name}  ({args.games} partidas)\n")
    run_tournament(
        agent_a, agent_b,
        n_games=args.games,
        csv_path=args.out,
    )


if __name__ == "__main__":
    main()
