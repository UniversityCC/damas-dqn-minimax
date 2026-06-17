"""
Evaluacion preliminar DQN vs Minimax (issue #21).

Metricas capturadas por partida:
  - resultado (victoria/derrota/empate)
  - longitud en semijugadas
  - tiempo promedio por jugada del DQN
  - recompensa acumulada (+1 victoria, 0 empate, -1 derrota)

Ademas imprime 2-3 ejemplos cualitativos: momentos de captura, promocion
y el desenlace final de partidas representativas.

Uso
---
  python src/eval/preliminary.py --checkpoint models/checkpoint_final.pt
  python src/eval/preliminary.py --checkpoint models/checkpoint_final.pt --games 40 --depth 3
  python src/eval/preliminary.py --checkpoint models/checkpoint_final.pt --out results/preliminary.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from damas.engine import (
    initial_state, legal_moves, step, is_terminal, result,
    _JUMP_OVER, _promotion_row,
)
from agents.minimax import MinimaxAgent


# ---------------------------------------------------------------------------
# Estructuras de datos
# ---------------------------------------------------------------------------

@dataclass
class MoveRecord:
    half_move:  int
    turn:       int        # 1=rojo, -1=negro
    action:     tuple
    is_capture: bool
    is_promotion: bool
    elapsed_ms: float      # solo para el DQN; -1 para minimax


@dataclass
class GameRecord:
    game_id:       int
    result:        int          # 1=DQN gana, -1=DQN pierde, 0=empate
    half_moves:    int
    dqn_moves:     int
    dqn_total_ms:  float
    moves:         list[MoveRecord] = field(default_factory=list)

    @property
    def reward(self) -> float:
        return float(self.result)

    @property
    def avg_ms_per_move(self) -> float:
        return self.dqn_total_ms / self.dqn_moves if self.dqn_moves > 0 else 0.0


# ---------------------------------------------------------------------------
# Helpers de geometria
# ---------------------------------------------------------------------------

def _is_capture(action: tuple) -> bool:
    return len(action) >= 2 and (action[0], action[-1]) in _JUMP_OVER or (
        len(action) > 2
    )


def _check_capture(action: tuple) -> bool:
    for i in range(len(action) - 1):
        if (action[i], action[i + 1]) in _JUMP_OVER:
            return True
    return False


def _check_promotion(action: tuple, board: list, turn: int) -> bool:
    src = action[0]
    dst = action[-1]
    piece = board[src]
    if abs(piece) == 1 and piece * turn > 0:
        return dst in _promotion_row(turn)
    return False


# ---------------------------------------------------------------------------
# Jugar una partida con registro completo
# ---------------------------------------------------------------------------

def play_game_recorded(
    dqn_agent: Any,
    minimax_agent: MinimaxAgent,
    game_id: int,
    dqn_plays_red: bool = True,
    max_half_moves: int = 300,
) -> GameRecord:
    state = initial_state()
    record = GameRecord(
        game_id=game_id,
        result=0,
        half_moves=0,
        dqn_moves=0,
        dqn_total_ms=0.0,
    )

    for hm in range(max_half_moves):
        if is_terminal(state):
            break

        is_dqn_turn = (state["turn"] == 1) == dqn_plays_red

        board_before = list(state["board"])
        t0 = time.perf_counter()

        if is_dqn_turn:
            action = dqn_agent.act(state, greedy=True)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            record.dqn_moves += 1
            record.dqn_total_ms += elapsed_ms
        else:
            action = minimax_agent.choose_action(state)
            elapsed_ms = -1.0

        if action is None:
            break

        capture   = _check_capture(action)
        promotion = _check_promotion(action, board_before, state["turn"])

        record.moves.append(MoveRecord(
            half_move=hm + 1,
            turn=state["turn"],
            action=action,
            is_capture=capture,
            is_promotion=promotion,
            elapsed_ms=elapsed_ms,
        ))

        state = step(state, action)
        record.half_moves = hm + 1

    raw = result(state)
    raw = raw if raw is not None else 0
    # Convertir al punto de vista del DQN
    if dqn_plays_red:
        record.result = raw          # +1 si rojo (DQN) gana
    else:
        record.result = -raw         # DQN es negro: invertir
    return record


# ---------------------------------------------------------------------------
# Resumen de metricas
# ---------------------------------------------------------------------------

@dataclass
class PreliminaryStats:
    dqn_label:    str
    mm_label:     str
    games:        list[GameRecord] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.games)

    @property
    def wins(self) -> int:
        return sum(1 for g in self.games if g.result == 1)

    @property
    def losses(self) -> int:
        return sum(1 for g in self.games if g.result == -1)

    @property
    def draws(self) -> int:
        return sum(1 for g in self.games if g.result == 0)

    @property
    def win_rate(self) -> float:
        return self.wins / self.total if self.total else 0.0

    @property
    def avg_reward(self) -> float:
        return sum(g.reward for g in self.games) / self.total if self.total else 0.0

    @property
    def avg_half_moves(self) -> float:
        return sum(g.half_moves for g in self.games) / self.total if self.total else 0.0

    @property
    def avg_ms_per_move(self) -> float:
        total_ms    = sum(g.dqn_total_ms for g in self.games)
        total_moves = sum(g.dqn_moves    for g in self.games)
        return total_ms / total_moves if total_moves else 0.0


def _print_metrics(stats: PreliminaryStats) -> None:
    sep = "=" * 58
    n   = stats.total or 1
    print(f"\n{sep}")
    print(f"  Evaluacion preliminar: {stats.dqn_label} vs {stats.mm_label}")
    print(sep)
    print(f"  Partidas jugadas   : {stats.total}")
    print(f"  Victorias DQN      : {stats.wins}  ({stats.win_rate * 100:.1f}%)")
    print(f"  Derrotas DQN       : {stats.losses}  ({stats.losses / n * 100:.1f}%)")
    print(f"  Empates            : {stats.draws}  ({stats.draws / n * 100:.1f}%)")
    print(f"  Recompensa media   : {stats.avg_reward:+.3f}")
    print(f"  Longitud media     : {stats.avg_half_moves:.1f} semijugadas")
    print(f"  Tiempo/jugada DQN  : {stats.avg_ms_per_move:.2f} ms")
    print(sep)


# ---------------------------------------------------------------------------
# Ejemplos cualitativos
# ---------------------------------------------------------------------------

_PIECE_CHAR = {0: ".", 1: "r", -1: "b", 2: "R", -2: "B"}


def _board_str(board: list) -> str:
    lines = []
    sq = 0
    for row in range(8):
        cells = []
        for col in range(8):
            if (col % 2) == (row % 2):
                cells.append(" ")
            else:
                cells.append(_PIECE_CHAR.get(board[sq], "?"))
                sq += 1
        lines.append(f"  {''.join(cells)}")
    return "\n".join(lines)


def _print_qualitative(stats: PreliminaryStats, n_examples: int = 3) -> None:
    if not stats.games:
        return

    print("\n  Ejemplos cualitativos")
    print("  " + "-" * 54)

    # Seleccionar: una victoria (si existe), una derrota, una con mas capturas
    candidates: list[GameRecord] = []
    wins   = [g for g in stats.games if g.result == 1]
    losses = [g for g in stats.games if g.result == -1]
    most_captures = sorted(
        stats.games,
        key=lambda g: sum(1 for m in g.moves if m.is_capture),
        reverse=True,
    )

    if wins:
        candidates.append(wins[0])
    if losses:
        candidates.append(losses[0])
    if most_captures and most_captures[0] not in candidates:
        candidates.append(most_captures[0])

    candidates = candidates[:n_examples]

    for game in candidates:
        outcome = {1: "Victoria DQN", -1: "Derrota DQN", 0: "Empate"}[game.result]
        captures   = sum(1 for m in game.moves if m.is_capture)
        promotions = sum(1 for m in game.moves if m.is_promotion)

        print(f"\n  Partida {game.game_id} — {outcome}")
        print(f"  Duracion: {game.half_moves} semijugadas  |  "
              f"Capturas totales: {captures}  |  Promociones: {promotions}")

        # Mostrar momentos destacados del DQN
        highlights = [
            m for m in game.moves
            if (m.is_capture or m.is_promotion) and m.turn == (1 if True else -1)
        ]
        if highlights:
            print(f"  Jugadas destacadas del DQN ({len(highlights)}):")
            for h in highlights[:3]:
                tipo = []
                if h.is_capture:   tipo.append("captura")
                if h.is_promotion: tipo.append("promocion")
                print(f"    semijugada {h.half_move:>3}: {h.action}  [{', '.join(tipo)}]  "
                      f"({h.elapsed_ms:.1f} ms)")
        else:
            print("  Sin jugadas destacadas del DQN en esta partida.")


# ---------------------------------------------------------------------------
# Guardar CSV
# ---------------------------------------------------------------------------

def _save_csv(stats: PreliminaryStats, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "game_id", "result", "reward", "half_moves",
            "dqn_moves", "avg_ms_per_move",
        ])
        writer.writeheader()
        for g in stats.games:
            writer.writerow({
                "game_id":         g.game_id,
                "result":          g.result,
                "reward":          g.reward,
                "half_moves":      g.half_moves,
                "dqn_moves":       g.dqn_moves,
                "avg_ms_per_move": round(g.avg_ms_per_move, 3),
            })


# ---------------------------------------------------------------------------
# Runner principal
# ---------------------------------------------------------------------------

def run_preliminary(
    checkpoint:     str,
    depth:          int        = 3,
    n_games:        int        = 20,
    csv_path:       str | None = None,
    verbose:        bool       = True,
) -> PreliminaryStats:
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(
            f"No se encontro el checkpoint: {checkpoint}\n"
            "Ejecuta selfplay.py primero para generar el modelo entrenado."
        )

    from agents.dqn import DQNAgent
    dqn = DQNAgent()
    dqn.load(checkpoint)
    dqn_label = f"DQN(ep{dqn.learn_steps})"
    mm        = MinimaxAgent(depth=depth, player=1)

    stats = PreliminaryStats(dqn_label=dqn_label, mm_label=mm.name)

    if verbose:
        print(f"Checkpoint: {checkpoint}  ({dqn.learn_steps} pasos)")
        print(f"Oponente  : {mm.name}  |  Partidas: {n_games}\n")

    for i in range(1, n_games + 1):
        dqn_plays_red = (i % 2 == 1)
        rec = play_game_recorded(dqn, mm, game_id=i, dqn_plays_red=dqn_plays_red)
        stats.games.append(rec)

        if verbose:
            sym = {1: "W", -1: "L", 0: "D"}[rec.result]
            print(f"  [{i:>3}/{n_games}]  {'rojo' if dqn_plays_red else 'negro'}  "
                  f"{sym}  {rec.half_moves} t  {rec.avg_ms_per_move:.1f} ms/mov")

    _print_metrics(stats)
    _print_qualitative(stats)

    if csv_path:
        _save_csv(stats, csv_path)
        if verbose:
            print(f"\nResultados guardados en: {csv_path}")

    return stats


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluacion preliminar DQN vs Minimax (issue #21)"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="models/checkpoint_final.pt",
        help="Ruta al checkpoint del DQN",
    )
    parser.add_argument(
        "--depth", type=int, default=3,
        help="Profundidad del Minimax (default: 3)",
    )
    parser.add_argument(
        "--games", type=int, default=20,
        help="Numero de partidas (default: 20)",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Ruta del CSV de resultados (ej: results/preliminary.csv)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suprime el detalle por partida",
    )
    args = parser.parse_args()

    run_preliminary(
        checkpoint=args.checkpoint,
        depth=args.depth,
        n_games=args.games,
        csv_path=args.out,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
