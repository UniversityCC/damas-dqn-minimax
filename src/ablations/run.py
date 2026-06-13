"""
Ablation study: red objetivo y modelado de la recompensa (issue #14).

Mide el impacto de dos factores de forma independiente:
  1. Red objetivo (use_target=True vs False)
  2. Recompensa (solo terminal vs shaping: captura + promoción)

Uso
---
  python src/ablations/run.py
  python src/ablations/run.py --steps 5000 --eval-games 30 --out data/ablations.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from damas.engine import legal_moves, initial_state, step, is_terminal, result, encode
from agents.dqn import DQNAgent
from env.damas_env import DamasEnv


# ---------------------------------------------------------------------------
# Agente aleatorio para evaluación rápida
# ---------------------------------------------------------------------------

class _RandomAgent:
    def __init__(self, seed: int = 0) -> None:
        self.name = "Random"
        self._rng  = random.Random(seed)

    def act(self, state: dict, greedy: bool = False) -> tuple[int, ...] | None:
        moves = legal_moves(state)
        return self._rng.choice(moves) if moves else None


# ---------------------------------------------------------------------------
# Bucle de entrenamiento (auto-juego compacto)
# ---------------------------------------------------------------------------

def _train(
    agent: DQNAgent,
    env: DamasEnv,
    steps: int,
) -> list[float]:
    """Entrena el agente por *steps* pasos con auto-juego. Devuelve lista de losses."""
    losses: list[float] = []
    obs, _ = env.reset()
    state = env.state

    for _ in range(steps):
        action = agent.act(state)
        idx    = env.tuple_to_action(action)
        obs2, reward, terminated, _, _ = env.step(idx)
        next_state = env.state
        agent.remember(state, action, reward, next_state, terminated)
        loss = agent.learn()
        if loss is not None:
            losses.append(loss)
        if terminated:
            obs, _ = env.reset()
            state = env.state
        else:
            state = next_state

    return losses


# ---------------------------------------------------------------------------
# Evaluación greedy vs agente aleatorio
# ---------------------------------------------------------------------------

def _evaluate(agent: DQNAgent, n_games: int, seed: int = 42) -> float:
    """Win-rate del agente DQN (jugando de rojo) contra un agente aleatorio."""
    rng = random.Random(seed)
    wins = 0
    for _ in range(n_games):
        state = initial_state()
        for _ in range(300):
            if is_terminal(state):
                break
            if state["turn"] == 1:
                action = agent.act(state, greedy=True)
            else:
                moves = legal_moves(state)
                action = rng.choice(moves) if moves else None
            if action is None:
                break
            state = step(state, action)
        r = result(state)
        if r == 1:
            wins += 1
    return wins / n_games


# ---------------------------------------------------------------------------
# Configuraciones de ablación
# ---------------------------------------------------------------------------

@dataclass
class AblationConfig:
    name:           str
    use_target:     bool
    capture_reward: float
    king_reward:    float


CONFIGS: list[AblationConfig] = [
    AblationConfig("target+shaping",    use_target=True,  capture_reward=0.3, king_reward=0.5),
    AblationConfig("target+terminal",   use_target=True,  capture_reward=0.0, king_reward=0.0),
    AblationConfig("no-target+shaping", use_target=False, capture_reward=0.3, king_reward=0.5),
    AblationConfig("no-target+terminal",use_target=False, capture_reward=0.0, king_reward=0.0),
]


# ---------------------------------------------------------------------------
# Runner principal
# ---------------------------------------------------------------------------

@dataclass
class AblationResult:
    name:           str
    use_target:     bool
    capture_reward: float
    king_reward:    float
    win_rate:       float
    avg_loss:       float
    learn_steps:    int


def run_ablations(
    steps: int       = 3_000,
    eval_games: int  = 20,
    csv_path: str | None = None,
    verbose: bool    = True,
) -> list[AblationResult]:
    results: list[AblationResult] = []

    for cfg in CONFIGS:
        if verbose:
            print(f"\n  [{cfg.name}]  entrenando {steps} pasos...", flush=True)

        env = DamasEnv(
            capture_reward=cfg.capture_reward,
            king_reward=cfg.king_reward,
        )
        agent = DQNAgent(
            use_target=cfg.use_target,
            eps_decay_steps=steps,
            batch_size=32,
            buffer_capacity=5_000,
            target_update_freq=max(1, steps // 10),
        )

        losses = _train(agent, env, steps)
        avg_loss = sum(losses) / len(losses) if losses else float("nan")
        win_rate = _evaluate(agent, eval_games)

        res = AblationResult(
            name=cfg.name,
            use_target=cfg.use_target,
            capture_reward=cfg.capture_reward,
            king_reward=cfg.king_reward,
            win_rate=win_rate,
            avg_loss=avg_loss,
            learn_steps=agent.learn_steps,
        )
        results.append(res)

        if verbose:
            print(
                f"    win_rate={win_rate:.1%}  avg_loss={avg_loss:.4f}"
                f"  learn_steps={agent.learn_steps}"
            )

    if csv_path:
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "name", "use_target", "capture_reward",
                    "king_reward", "win_rate", "avg_loss", "learn_steps",
                ],
            )
            writer.writeheader()
            for r in results:
                writer.writerow({
                    "name":           r.name,
                    "use_target":     r.use_target,
                    "capture_reward": r.capture_reward,
                    "king_reward":    r.king_reward,
                    "win_rate":       round(r.win_rate, 4),
                    "avg_loss":       round(r.avg_loss, 6) if r.avg_loss == r.avg_loss else "nan",
                    "learn_steps":    r.learn_steps,
                })
        if verbose:
            print(f"\nResultados guardados en: {csv_path}")

    _print_table(results)
    return results


def _print_table(results: list[AblationResult]) -> None:
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  {'Configuración':<26} {'target':>6}  {'shaping':>7}  {'win%':>6}  {'avg_loss':>9}")
    print(sep)
    for r in results:
        shaping = r.capture_reward > 0 or r.king_reward > 0
        print(
            f"  {r.name:<26} {'sí' if r.use_target else 'no':>6}  "
            f"{'sí' if shaping else 'no':>7}  "
            f"{r.win_rate * 100:>5.1f}%  {r.avg_loss:>9.4f}"
        )
    print(sep)


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablation study: red objetivo y recompensa (issue #14)"
    )
    parser.add_argument(
        "--steps",      type=int, default=3_000,
        help="Pasos de entrenamiento por configuración (default: 3000)",
    )
    parser.add_argument(
        "--eval-games", type=int, default=20,
        help="Partidas de evaluación vs agente aleatorio (default: 20)",
    )
    parser.add_argument(
        "--out",        type=str, default=None,
        help="Ruta del CSV de resultados (default: sin guardar)",
    )
    args = parser.parse_args()

    print(f"Ablation study — {len(CONFIGS)} configs × {args.steps} pasos\n")
    run_ablations(steps=args.steps, eval_games=args.eval_games, csv_path=args.out)


if __name__ == "__main__":
    main()
