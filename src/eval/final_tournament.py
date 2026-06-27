"""
Torneo final, ELO relativo y metricas completas (issue #16).

Enfrenta todas las variantes en round-robin:
  DQN(checkpoint) vs Minimax(d=3), Minimax(d=4), Minimax(d=5), Minimax(d=6)
  y entre los propios Minimax.

Calcula ELO relativo (K=32, base=1500), win-rate, tiempo/jugada y longitud
media. Guarda CSV con resultados por partida, tabla de ELO y figuras PNG.

Uso
---
  python src/eval/final_tournament.py --checkpoint models/checkpoint_final.pt
  python src/eval/final_tournament.py --checkpoint models/checkpoint_final.pt --games 20 --out results/final
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from damas.engine import legal_moves, initial_state, step, is_terminal, result
from agents.minimax import MinimaxAgent
from tournament.tournament import run_tournament, TournamentStats


# ---------------------------------------------------------------------------
# ELO
# ---------------------------------------------------------------------------

class EloTracker:
    """Calcula ELO relativo entre un conjunto de agentes."""

    def __init__(self, agents: list[str], base: float = 1500.0, k: float = 32.0):
        self.ratings: dict[str, float] = {a: base for a in agents}
        self.k = k

    def expected(self, a: str, b: str) -> float:
        return 1.0 / (1.0 + 10 ** ((self.ratings[b] - self.ratings[a]) / 400.0))

    def update(self, winner: str, loser: str, draw: bool = False) -> None:
        ea = self.expected(winner, loser)
        eb = self.expected(loser, winner)
        sa = 0.5 if draw else 1.0
        sb = 0.5 if draw else 0.0
        self.ratings[winner] += self.k * (sa - ea)
        self.ratings[loser]  += self.k * (sb - eb)

    def apply_stats(self, stats: TournamentStats) -> None:
        for g in stats.games:
            if g.result == 1:
                self.update(g.red_agent, g.black_agent)
            elif g.result == -1:
                self.update(g.black_agent, g.red_agent)
            else:
                self.update(g.red_agent, g.black_agent, draw=True)

    def sorted_table(self) -> list[tuple[str, float]]:
        return sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Wrapper DQN con nombre y modo greedy
# ---------------------------------------------------------------------------

class _NamedDQN:
    def __init__(self, agent: Any, label: str) -> None:
        self._agent = agent
        self.name   = label

    def choose_action(self, state):
        return self._agent.act(state, greedy=True)


# ---------------------------------------------------------------------------
# Metricas adicionales por confrontacion
# ---------------------------------------------------------------------------

@dataclass
class MatchupMetrics:
    agent_a:       str
    agent_b:       str
    a_wins:        int
    b_wins:        int
    draws:         int
    total:         int
    avg_half_moves: float
    win_rate_a:    float


def _matchup_metrics(stats: TournamentStats) -> MatchupMetrics:
    avg_hm = (
        sum(g.half_moves for g in stats.games) / len(stats.games)
        if stats.games else 0.0
    )
    return MatchupMetrics(
        agent_a=stats.agent_a_name,
        agent_b=stats.agent_b_name,
        a_wins=stats.agent_a_wins,
        b_wins=stats.agent_b_wins,
        draws=stats.draws,
        total=stats.total,
        avg_half_moves=round(avg_hm, 1),
        win_rate_a=round(stats.agent_a_wins / stats.total, 4) if stats.total else 0.0,
    )


# ---------------------------------------------------------------------------
# Guardar resultados
# ---------------------------------------------------------------------------

def _save_matchups_csv(matchups: list[MatchupMetrics], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "agent_a", "agent_b", "a_wins", "b_wins", "draws",
            "total", "win_rate_a", "avg_half_moves",
        ])
        writer.writeheader()
        for m in matchups:
            writer.writerow({
                "agent_a":        m.agent_a,
                "agent_b":        m.agent_b,
                "a_wins":         m.a_wins,
                "b_wins":         m.b_wins,
                "draws":          m.draws,
                "total":          m.total,
                "win_rate_a":     m.win_rate_a,
                "avg_half_moves": m.avg_half_moves,
            })


def _save_elo_csv(elo: EloTracker, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "agent", "elo"])
        writer.writeheader()
        for rank, (agent, rating) in enumerate(elo.sorted_table(), 1):
            writer.writerow({"rank": rank, "agent": agent, "elo": round(rating, 1)})


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

def _save_figures(elo: EloTracker, matchups: list[MatchupMetrics], out_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  matplotlib no disponible, se omiten las figuras.")
        return

    os.makedirs(out_dir, exist_ok=True)

    # --- Figura 1: ELO por agente ---
    table   = elo.sorted_table()
    agents  = [a for a, _ in table]
    ratings = [r for _, r in table]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(agents[::-1], ratings[::-1], color="#4C72B0")
    ax.set_xlabel("ELO")
    ax.set_title("Clasificacion ELO — Torneo Final")
    for bar, val in zip(bars, ratings[::-1]):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}", va="center", fontsize=9)
    ax.set_xlim(min(ratings) - 80, max(ratings) + 80)
    plt.tight_layout()
    elo_path = os.path.join(out_dir, "torneo_elo.png")
    fig.savefig(elo_path, dpi=150)
    plt.close(fig)
    print(f"  Figura guardada: {elo_path}")

    # --- Figura 2: Heatmap de win-rate ---
    agent_set = list(dict.fromkeys(
        [m.agent_a for m in matchups] + [m.agent_b for m in matchups]
    ))
    n = len(agent_set)
    idx = {a: i for i, a in enumerate(agent_set)}
    matrix = np.full((n, n), float("nan"))

    for m in matchups:
        i, j = idx[m.agent_a], idx[m.agent_b]
        matrix[i][j] = m.win_rate_a
        matrix[j][i] = 1.0 - m.win_rate_a

    fig, ax = plt.subplots(figsize=(6, 5))
    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, vmin=0, vmax=1, cmap="RdYlGn")
    plt.colorbar(im, ax=ax, label="Win-rate (fila vs columna)")
    ax.set_xticks(range(n)); ax.set_xticklabels(agent_set, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(agent_set, fontsize=8)
    ax.set_title("Win-rate — Torneo Final")
    for i in range(n):
        for j in range(n):
            if not np.isnan(matrix[i][j]):
                ax.text(j, i, f"{matrix[i][j]:.2f}", ha="center", va="center", fontsize=8)
    plt.tight_layout()
    hm_path = os.path.join(out_dir, "torneo_winrate.png")
    fig.savefig(hm_path, dpi=150)
    plt.close(fig)
    print(f"  Figura guardada: {hm_path}")


# ---------------------------------------------------------------------------
# Tabla de consola
# ---------------------------------------------------------------------------

def _print_elo_table(elo: EloTracker) -> None:
    sep = "=" * 46
    print(f"\n{sep}")
    print(f"  Clasificacion ELO final")
    print(sep)
    print(f"  {'Pos':>3}  {'Agente':<24}  {'ELO':>7}")
    print(sep)
    for rank, (agent, rating) in enumerate(elo.sorted_table(), 1):
        print(f"  {rank:>3}  {agent:<24}  {rating:>7.1f}")
    print(sep)


def _print_matchup_table(matchups: list[MatchupMetrics]) -> None:
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  Resultados por confrontacion")
    print(sep)
    print(f"  {'Agente A':<22}  {'Agente B':<22}  {'WR-A':>6}  {'t/sem':>6}")
    print(sep)
    for m in matchups:
        print(
            f"  {m.agent_a:<22}  {m.agent_b:<22}  "
            f"{m.win_rate_a * 100:>5.1f}%  {m.avg_half_moves:>6.1f}"
        )
    print(sep)


# ---------------------------------------------------------------------------
# Runner principal
# ---------------------------------------------------------------------------

def run_final_tournament(
    checkpoint:  str,
    depths:      list[int]  = None,
    n_games:     int        = 20,
    out_dir:     str        = "results/final",
    verbose:     bool       = True,
) -> tuple[EloTracker, list[MatchupMetrics]]:
    if depths is None:
        depths = [3, 4, 5, 6]

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(
            f"No se encontro el checkpoint: {checkpoint}\n"
            "Ejecuta selfplay.py primero para generar el modelo entrenado."
        )

    from agents.dqn import DQNAgent
    dqn_base = DQNAgent()
    dqn_base.load(checkpoint)
    dqn = _NamedDQN(dqn_base, f"DQN(ep{dqn_base.learn_steps})")

    minimaxes = [MinimaxAgent(depth=d, player=1) for d in depths]
    all_agents: list[Any] = [dqn] + minimaxes
    agent_names = [a.name for a in all_agents]

    elo      = EloTracker(agent_names)
    matchups: list[MatchupMetrics] = []

    pairs = list(combinations(all_agents, 2))
    total_pairs = len(pairs)

    if verbose:
        print(f"Torneo final — {len(all_agents)} agentes, "
              f"{total_pairs} confrontaciones x {n_games} partidas\n")

    for idx, (a, b) in enumerate(pairs, 1):
        if verbose:
            print(f"  [{idx}/{total_pairs}]  {a.name}  vs  {b.name}")

        stats = run_tournament(a, b, n_games=n_games, verbose=False)
        elo.apply_stats(stats)
        matchups.append(_matchup_metrics(stats))

    _print_elo_table(elo)
    _print_matchup_table(matchups)

    os.makedirs(out_dir, exist_ok=True)
    _save_matchups_csv(matchups, os.path.join(out_dir, "matchups.csv"))
    _save_elo_csv(elo,           os.path.join(out_dir, "elo.csv"))
    _save_figures(elo, matchups, out_dir)

    if verbose:
        print(f"\nResultados guardados en: {out_dir}/")

    return elo, matchups


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Torneo final y ELO relativo (issue #16)"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="models/checkpoint_final.pt",
        help="Ruta al checkpoint del DQN",
    )
    parser.add_argument(
        "--depths", type=int, nargs="+", default=[3, 4, 5, 6],
        help="Profundidades de Minimax (default: 3 4 5 6)",
    )
    parser.add_argument(
        "--games", type=int, default=20,
        help="Partidas por confrontacion (default: 20)",
    )
    parser.add_argument(
        "--out", type=str, default="results/final",
        help="Directorio de salida (default: results/final)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
    )
    args = parser.parse_args()

    run_final_tournament(
        checkpoint=args.checkpoint,
        depths=args.depths,
        n_games=args.games,
        out_dir=args.out,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
