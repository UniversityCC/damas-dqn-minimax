"""Evaluación COST-MATCHED del agente híbrido (DQN+búsqueda) vs Minimax.

Mide, por configuración, no solo el win-rate sino el COSTO de decisión, para una
comparación justa ("a presupuesto de cómputo comparable", no "quien busca más hondo gana"):
  - win-rate (W/L/D)        sobre N partidas con aperturas aleatorias y semillas pareadas;
  - nodos/jugada (PRIMARIA) : determinista, independiente del recolector de basura;
  - ms/jugada (secundaria)  : se cronometra con gc.disable() y se reporta la mediana.

Sirve también para el experimento de CONTROL: misma búsqueda con eval=DQN vs eval=heurística
a igual profundidad, para aislar el aporte del evaluador aprendido.

Uso:
  python src/eval/hybrid_eval.py --eval dqn --checkpoint models/swa.pt --depth 4 --opp-depths 5 6
  python src/eval/hybrid_eval.py --eval heuristic --depth 4 --opp-depths 5 6
"""
from __future__ import annotations

import argparse
import gc
import random
import statistics
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _c in (_HERE.parent, _HERE):
    if (_c / "damas").is_dir():
        sys.path.insert(0, str(_c))
        break

from damas.engine import initial_state, legal_moves, step, is_terminal, result
from agents.minimax import MinimaxAgent
from agents.hybrid import HybridAgent, dqn_value_fn, heuristic_value_fn


def play_game(hybrid: HybridAgent, mm: MinimaxAgent, hybrid_color: int,
              opening_plies: int, rng: random.Random, max_half: int = 300):
    """Una partida híbrido vs Minimax con apertura aleatoria. Devuelve (resultado, nodos, ms)."""
    s = initial_state()
    for _ in range(opening_plies):                 # apertura aleatoria (variedad)
        if is_terminal(s):
            break
        s = step(s, rng.choice(legal_moves(s)))
    hybrid._cache.clear()                          # caché fresca por partida
    nodes, times, half = [], [], 0
    while not is_terminal(s) and half < max_half:
        if s["turn"] == hybrid_color:
            gc.disable()                           # evita que pausas del GC falseen el tiempo
            t0 = time.perf_counter()
            a = hybrid.choose_action(s)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            gc.enable()
            nodes.append(hybrid.nodes)
            times.append(dt_ms)
        else:
            a = mm.choose_action(s)
        if a is None:
            break
        s = step(s, a)
        half += 1
    r = result(s) if is_terminal(s) else 0
    outcome = "W" if r == hybrid_color else ("D" if r in (0, None) else "L")
    return outcome, nodes, times


def evaluate_vs_depth(hybrid: HybridAgent, opp_depth: int, games: int = 30,
                      opening_plies: int = 4, seed_base: int = 2000,
                      max_half: int = 300) -> dict:
    """Enfrenta el híbrido a Minimax(opp_depth) sobre ``games`` partidas pareadas."""
    mm = MinimaxAgent(depth=opp_depth)
    w = l = d = 0
    all_nodes: list[int] = []
    all_times: list[float] = []
    for g in range(games):
        rng = random.Random(seed_base + g)
        color = 1 if g % 2 == 0 else -1
        outcome, nodes, times = play_game(hybrid, mm, color, opening_plies, rng, max_half)
        w += outcome == "W"; l += outcome == "L"; d += outcome == "D"
        all_nodes += nodes; all_times += times
    return {
        "opp_depth": opp_depth, "games": games,
        "wins": w, "losses": l, "draws": d,
        "win_rate": w / games, "no_loss_rate": (w + d) / games,
        "nodes_per_move": statistics.mean(all_nodes) if all_nodes else 0.0,
        "ms_per_move_median": statistics.median(all_times) if all_times else 0.0,
        "ms_per_move_mean": statistics.mean(all_times) if all_times else 0.0,
    }


def _build_eval_fn(args):
    if args.eval == "dqn":
        from agents.dqn import DQNAgent
        agent = DQNAgent()
        agent.load(args.checkpoint)
        return dqn_value_fn(agent.online), f"DQN({Path(args.checkpoint).stem})"
    return heuristic_value_fn(), "heurística"


def main() -> None:
    p = argparse.ArgumentParser(description="Eval cost-matched del híbrido vs Minimax")
    p.add_argument("--eval", choices=["dqn", "heuristic"], required=True)
    p.add_argument("--checkpoint", default=None, help="requerido si --eval dqn")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--depth", type=int, help="profundidad de búsqueda fija k")
    g.add_argument("--time-budget", type=float, help="segundos por jugada (iterative deepening)")
    p.add_argument("--opp-depths", type=int, nargs="+", default=[3, 4, 5, 6])
    p.add_argument("--games", type=int, default=30)
    p.add_argument("--opening-plies", type=int, default=4)
    p.add_argument("--no-ordering", dest="ordering", action="store_false")
    p.add_argument("--no-cache", dest="cache", action="store_false")
    p.add_argument("--out", default=None, help="CSV de salida")
    args = p.parse_args()

    eval_fn, eval_name = _build_eval_fn(args)
    hybrid = HybridAgent(eval_fn=eval_fn, depth=args.depth, time_budget=args.time_budget,
                         use_ordering=args.ordering, use_cache=args.cache)
    tag = f"t={args.time_budget}s" if args.time_budget else f"k={args.depth}"
    print(f"Híbrido [{eval_name}, {tag}] vs Minimax (cost-matched, {args.games} partidas):")
    rows = []
    for opp in args.opp_depths:
        r = evaluate_vs_depth(hybrid, opp, args.games, args.opening_plies)
        print(f"  d={opp}: win {100*r['win_rate']:.0f}% (no-pierde {100*r['no_loss_rate']:.0f}%) "
              f"| W/L/D={r['wins']}/{r['losses']}/{r['draws']} "
              f"| nodos/jugada={r['nodes_per_move']:.0f} | ms/jugada(mediana)={r['ms_per_move_median']:.1f}")
        r.update({"eval": eval_name, "config": tag})
        rows.append(r)

    if args.out:
        import csv
        import os
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wcsv.writeheader(); wcsv.writerows(rows)
        print(f"CSV: {args.out}")


if __name__ == "__main__":
    main()
