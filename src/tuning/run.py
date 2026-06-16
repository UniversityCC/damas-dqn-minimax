"""
Ajuste de hiperparámetros y comparación arquitectónica del DQN (issue #13).

Hace dos estudios y produce una sola tabla "configuración vs desempeño":

  1. Barrido de hiperparámetros (one-factor-at-a-time desde un baseline):
       lr, gamma, tamaño del buffer y frecuencia de actualización de la red objetivo.
  2. Comparación de 2-3 arquitecturas de la red Q:
       (256, 256), (512, 512) y (512, 512, 1024).

Cada configuración entrena por auto-juego un número corto de pasos y se mide su
win-rate (greedy) contra un agente aleatorio y la pérdida media. La idea NO es
entrenar el modelo final (eso es el #15), sino encontrar de forma barata la
configuración más prometedora para el entrenamiento largo.

Uso
---
  python src/tuning/run.py
  python src/tuning/run.py --steps 3000 --eval-games 30 --out data/tuning.csv
  python src/tuning/run.py --only arch          # solo comparación de arquitecturas
  python src/tuning/run.py --only hparams        # solo barrido de hiperparámetros
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from damas.engine import legal_moves, initial_state, step, is_terminal, result
from agents.dqn import DQNAgent
from env.damas_env import DamasEnv


# ---------------------------------------------------------------------------
# Baseline: punto de partida del barrido. Cada config varía UN factor.
# ---------------------------------------------------------------------------

BASE_LR        = 1e-3
BASE_GAMMA     = 0.99
BASE_BUFFER    = 5_000
BASE_HIDDEN    = (512, 512)
# target_update_freq se fija como una fracción de los pasos (igual que en #14);
# en el baseline equivale a steps // 10. Aquí lo expresamos como divisor.
BASE_TARGET_DIV = 10


@dataclass
class TuneConfig:
    """Una configuración a evaluar. ``group`` separa los dos estudios."""
    group:       str
    name:        str
    lr:          float = BASE_LR
    gamma:       float = BASE_GAMMA
    buffer:      int = BASE_BUFFER
    target_div:  int = BASE_TARGET_DIV
    hidden:      tuple[int, ...] = BASE_HIDDEN


def hparam_configs() -> list[TuneConfig]:
    """Barrido one-factor-at-a-time de lr, gamma, buffer y frecuencia de target."""
    return [
        TuneConfig("hparams", "baseline"),
        # learning rate
        TuneConfig("hparams", "lr=5e-4",        lr=5e-4),
        TuneConfig("hparams", "lr=1e-4",        lr=1e-4),
        # factor de descuento
        TuneConfig("hparams", "gamma=0.95",     gamma=0.95),
        TuneConfig("hparams", "gamma=0.90",     gamma=0.90),
        # tamaño del replay buffer
        TuneConfig("hparams", "buffer=20k",     buffer=20_000),
        # frecuencia de actualización de la red objetivo (target_div alto = update lento)
        TuneConfig("hparams", "target=rápido",  target_div=40),
        TuneConfig("hparams", "target=lento",   target_div=4),
    ]


def arch_configs() -> list[TuneConfig]:
    """Comparación de 2-3 arquitecturas de la red Q (resto = baseline)."""
    return [
        TuneConfig("arch", "mlp-256x2",         hidden=(256, 256)),
        TuneConfig("arch", "mlp-512x2",         hidden=(512, 512)),
        TuneConfig("arch", "mlp-512x2-1024",    hidden=(512, 512, 1024)),
    ]


# ---------------------------------------------------------------------------
# Entrenamiento por auto-juego (mismo patrón que la ablación #14)
# ---------------------------------------------------------------------------

def _train(agent: DQNAgent, env: DamasEnv, steps: int) -> list[float]:
    """Entrena el agente *steps* pasos por auto-juego. Devuelve la lista de losses."""
    losses: list[float] = []
    env.reset()
    state = env.state

    for _ in range(steps):
        action = agent.act(state)
        idx    = env.tuple_to_action(action)
        _, reward, terminated, _, _ = env.step(idx)
        next_state = env.state
        agent.remember(state, action, float(reward), next_state, terminated)
        loss = agent.learn()
        if loss is not None:
            losses.append(loss)
        if terminated:
            env.reset()
            state = env.state
        else:
            state = next_state

    return losses


# ---------------------------------------------------------------------------
# Evaluación greedy vs agente aleatorio
# ---------------------------------------------------------------------------

def _evaluate(agent: DQNAgent, n_games: int, seed: int = 42) -> float:
    """Win-rate del DQN (jugando de rojo, greedy) contra un agente aleatorio."""
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
        if result(state) == 1:
            wins += 1
    return wins / n_games if n_games else 0.0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

@dataclass
class TuneResult:
    group:       str
    name:        str
    lr:          float
    gamma:       float
    buffer:      int
    target_div:  int
    hidden:      str
    win_rate:    float
    avg_loss:    float
    learn_steps: int


def _run_config(cfg: TuneConfig, steps: int, eval_games: int) -> TuneResult:
    env = DamasEnv()
    agent = DQNAgent(
        lr=cfg.lr,
        gamma=cfg.gamma,
        buffer_capacity=cfg.buffer,
        hidden=cfg.hidden,
        batch_size=32,
        eps_decay_steps=steps,
        target_update_freq=max(1, steps // cfg.target_div),
    )
    losses = _train(agent, env, steps)
    avg_loss = sum(losses) / len(losses) if losses else float("nan")
    win_rate = _evaluate(agent, eval_games)
    return TuneResult(
        group=cfg.group,
        name=cfg.name,
        lr=cfg.lr,
        gamma=cfg.gamma,
        buffer=cfg.buffer,
        target_div=cfg.target_div,
        hidden="x".join(str(h) for h in cfg.hidden),
        win_rate=win_rate,
        avg_loss=avg_loss,
        learn_steps=agent.learn_steps,
    )


def run_tuning(
    steps: int = 3_000,
    eval_games: int = 20,
    only: str | None = None,
    csv_path: str | None = None,
    verbose: bool = True,
) -> list[TuneResult]:
    """Ejecuta el barrido y/o la comparación arquitectónica.

    ``only``: ``"hparams"``, ``"arch"`` o ``None`` (ambos).
    """
    configs: list[TuneConfig] = []
    if only in (None, "hparams"):
        configs += hparam_configs()
    if only in (None, "arch"):
        configs += arch_configs()

    results: list[TuneResult] = []
    for cfg in configs:
        if verbose:
            print(f"\n  [{cfg.group}/{cfg.name}]  entrenando {steps} pasos...", flush=True)
        res = _run_config(cfg, steps, eval_games)
        results.append(res)
        if verbose:
            print(f"    win_rate={res.win_rate:.1%}  avg_loss={res.avg_loss:.4f}"
                  f"  learn_steps={res.learn_steps}")

    if csv_path:
        _write_csv(results, csv_path)
        if verbose:
            print(f"\nResultados guardados en: {csv_path}")

    if verbose:
        _print_table(results)
        _print_best(results)
    return results


# ---------------------------------------------------------------------------
# Salidas
# ---------------------------------------------------------------------------

_FIELDS = ["group", "name", "lr", "gamma", "buffer", "target_div",
           "hidden", "win_rate", "avg_loss", "learn_steps"]


def _write_csv(results: list[TuneResult], csv_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "group":       r.group,
                "name":        r.name,
                "lr":          r.lr,
                "gamma":       r.gamma,
                "buffer":      r.buffer,
                "target_div":  r.target_div,
                "hidden":      r.hidden,
                "win_rate":    round(r.win_rate, 4),
                "avg_loss":    round(r.avg_loss, 6) if r.avg_loss == r.avg_loss else "nan",
                "learn_steps": r.learn_steps,
            })


def _print_table(results: list[TuneResult]) -> None:
    sep = "=" * 78
    print(f"\n{sep}")
    print(f"  {'grupo':<8} {'config':<16} {'lr':>7} {'gamma':>6} {'buffer':>7} "
          f"{'arq':>14} {'win%':>6}")
    print(sep)
    for r in results:
        print(f"  {r.group:<8} {r.name:<16} {r.lr:>7.0e} {r.gamma:>6.2f} "
              f"{r.buffer:>7d} {r.hidden:>14} {r.win_rate * 100:>5.1f}%")
    print(sep)


def _print_best(results: list[TuneResult]) -> None:
    if not results:
        return
    best = max(results, key=lambda r: r.win_rate)
    print(f"\n>> Mejor configuración: [{best.group}/{best.name}]  "
          f"win_rate={best.win_rate:.1%}")
    print(f"   lr={best.lr}  gamma={best.gamma}  buffer={best.buffer}  "
          f"target_div={best.target_div}  arq={best.hidden}")


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ajuste de hiperparámetros y comparación arquitectónica (issue #13)"
    )
    parser.add_argument("--steps", type=int, default=3_000,
                        help="Pasos de entrenamiento por configuración (default: 3000)")
    parser.add_argument("--eval-games", type=int, default=20,
                        help="Partidas de evaluación vs agente aleatorio (default: 20)")
    parser.add_argument("--only", choices=["hparams", "arch"], default=None,
                        help="Ejecutar solo un estudio (default: ambos)")
    parser.add_argument("--out", type=str, default=None,
                        help="Ruta del CSV de resultados (default: sin guardar)")
    args = parser.parse_args()

    n = len(hparam_configs()) * (args.only in (None, "hparams")) + \
        len(arch_configs()) * (args.only in (None, "arch"))
    print(f"Tuning DQN — {n} configs × {args.steps} pasos\n")
    run_tuning(steps=args.steps, eval_games=args.eval_games,
               only=args.only, csv_path=args.out)


if __name__ == "__main__":
    main()
