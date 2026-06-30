"""Experimento: ¿más entrenamiento -> evaluador DQN más fuerte? (calidad del evaluador)

Aísla el aporte del *evaluador aprendido* de la *profundidad de búsqueda*: fija la
profundidad del híbrido en k y usa cada snapshot del entrenamiento (checkpoint_first,
checkpoint_epNNNNNN, checkpoint_final) como evaluador de hojas, enfrentándolo a Minimax(d)
con la evaluación cost-matched (aperturas aleatorias, semillas pareadas).

Lectura:
  - win% sube con learn_steps  -> entrenar fortalece el evaluador.
  - win% plano                 -> el evaluador tocó techo (la palanca es la búsqueda, no el
                                  entrenamiento).

Uso:
  python src/eval/evaluator_quality.py --checkpoint-dir models/snapshots \
      --depth 3 --opp-depths 4 5 --games 14 --out results/evaluator_quality.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _c in (_HERE.parent, _HERE):
    if (_c / "damas").is_dir():
        sys.path.insert(0, str(_c))
        break

from agents.dqn import DQNAgent
from agents.hybrid import HybridAgent, dqn_value_fn
from eval.hybrid_eval import evaluate_vs_depth


def _episode_order(path: Path) -> int:
    """Clave de orden temporal: first=0, epNNNNNN=N, final al final."""
    name = path.stem
    if "first" in name:
        return 0
    if "final" in name:
        return 10**9
    m = re.search(r"ep(\d+)", name)
    return int(m.group(1)) if m else 10**8


def run(checkpoint_dir: str, depth: int, opp_depths: list[int], games: int,
        opening_plies: int = 4, seed_base: int = 3000) -> list[dict]:
    ckpts = sorted(Path(checkpoint_dir).glob("checkpoint_*.pt"), key=_episode_order)
    if not ckpts:
        raise SystemExit(f"No hay checkpoints en {checkpoint_dir}")
    print(f"Checkpoints: {len(ckpts)} | k={depth} | {games} partidas | vs d={opp_depths}\n")
    rows: list[dict] = []
    for ck in ckpts:
        agent = DQNAgent()
        agent.load(str(ck))
        ep = _episode_order(ck)
        ep_label = "first" if ep == 0 else ("final" if ep >= 10**8 else f"ep{ep}")
        hybrid = HybridAgent(eval_fn=dqn_value_fn(agent.online), depth=depth)
        rec = {"checkpoint": ck.name, "ep_label": ep_label,
               "learn_steps": agent.learn_steps, "k": depth}
        line = f"{ck.name:26s} | learn_steps={agent.learn_steps:6d} |"
        for d in opp_depths:
            r = evaluate_vs_depth(hybrid, opp_depth=d, games=games,
                                  opening_plies=opening_plies, seed_base=seed_base)
            rec[f"win_d{d}"] = round(100 * r["win_rate"], 1)
            rec[f"noloss_d{d}"] = round(100 * r["no_loss_rate"], 1)
            rec[f"wld_d{d}"] = f"{r['wins']}/{r['losses']}/{r['draws']}"
            rec[f"nodes_d{d}"] = round(r["nodes_per_move"], 0)
            line += (f"  d{d}: W{rec[f'win_d{d}']:3.0f}% NP{rec[f'noloss_d{d}']:3.0f}%"
                     f" ({rec[f'wld_d{d}']})")
        print(line, flush=True)
        rows.append(rec)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Calidad del evaluador DQN vs cantidad de entrenamiento")
    p.add_argument("--checkpoint-dir", required=True,
                   help="carpeta con snapshots checkpoint_*.pt de UNA corrida")
    p.add_argument("--depth", type=int, default=3, help="profundidad FIJA del híbrido")
    p.add_argument("--opp-depths", type=int, nargs="+", default=[4, 5])
    p.add_argument("--games", type=int, default=14)
    p.add_argument("--opening-plies", type=int, default=4)
    p.add_argument("--out", default=None, help="CSV de salida")
    args = p.parse_args()

    rows = run(args.checkpoint_dir, args.depth, args.opp_depths, args.games, args.opening_plies)
    if args.out:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\nCSV: {args.out}")


if __name__ == "__main__":
    main()
