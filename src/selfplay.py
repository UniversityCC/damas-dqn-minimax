from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections import deque
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup: permite ejecutar desde la raíz o desde src/
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
for _candidate in (_HERE, _HERE / "src"):
    if (_candidate / "damas").is_dir():
        sys.path.insert(0, str(_candidate))
        break

from damas.engine import initial_state, legal_moves, step, is_terminal, result
from agents.dqn import DQNAgent

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("selfplay")


# ---------------------------------------------------------------------------
# Jugar una partida completa entre el agente y sí mismo
# ---------------------------------------------------------------------------

def play_episode(agent: DQNAgent, max_steps: int = 300) -> dict:
    """Juega una partida completa por self-play.

    Ambos jugadores usan la MISMA red online con la política ε-greedy actual.
    Las transiciones se guardan desde el punto de vista del jugador en turno,
    con recompensa negamax: el jugador que recibe la transición ve +1 al ganar
    y la target negamax invierte el Q del siguiente estado (ya implementado
    en DQNAgent.learn).

    Devuelve un dict con métricas de la partida.
    """
    state = initial_state()
    steps = 0
    transitions_added = 0

    while not is_terminal(state) and steps < max_steps:
        current_turn = state["turn"]

        # Selección de acción ε-greedy
        action = agent.act(state)

        # Paso en el motor
        next_state = step(state, action)
        done = is_terminal(next_state)

        # Recompensa desde la perspectiva del jugador actual
        if done:
            res = result(next_state)
            if res is None or res == 0:
                reward = 0.0          # empate
            elif res == current_turn:
                reward = 1.0          # gana el jugador actual
            else:
                reward = -1.0         # pierde el jugador actual
        else:
            reward = 0.0

        agent.remember(state, action, reward, next_state, done)
        transitions_added += 1

        state = next_state
        steps += 1

    # Resultado final para las métricas
    final_result = result(state) if is_terminal(state) else None
    return {
        "steps": steps,
        "transitions": transitions_added,
        "result": final_result,          # 1, -1, 0, o None (truncada)
        "truncated": steps >= max_steps,
    }


# ---------------------------------------------------------------------------
# Verificación de gradientes
# ---------------------------------------------------------------------------

def check_gradients(agent: DQNAgent) -> dict:
    """Comprueba que todos los parámetros de la red online tienen gradientes."""
    total_params = 0
    params_with_grad = 0
    max_grad_norm = 0.0

    for name, p in agent.online.named_parameters():
        total_params += 1
        if p.grad is not None:
            params_with_grad += 1
            grad_norm = float(p.grad.norm().item())
            max_grad_norm = max(max_grad_norm, grad_norm)

    return {
        "total_params": total_params,
        "params_with_grad": params_with_grad,
        "all_grads_flow": params_with_grad == total_params,
        "max_grad_norm": max_grad_norm,
    }


# ---------------------------------------------------------------------------
# Bucle principal de entrenamiento
# ---------------------------------------------------------------------------

def train(
    episodes: int = 1_000,
    max_steps_per_episode: int = 300,
    learn_every: int = 4,            # pasos de aprendizaje por transición generada
    log_every: int = 50,             # frecuencia de log en episodios
    checkpoint_dir: str = "models",
    checkpoint_every: int = 200,     # guardar checkpoint cada N episodios
    resume_checkpoint: str | None = None,
    device: str = "cpu",
    # Hiperparámetros del agente (defaults razonables para un arranque rápido)
    gamma: float = 0.99,
    lr: float = 1e-3,
    batch_size: int = 64,
    buffer_capacity: int = 50_000,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay_steps: int = 50_000,
    target_update_freq: int = 1_000,
) -> DQNAgent:

    os.makedirs(checkpoint_dir, exist_ok=True)

    agent = DQNAgent(
        gamma=gamma,
        lr=lr,
        batch_size=batch_size,
        buffer_capacity=buffer_capacity,
        eps_start=eps_start,
        eps_end=eps_end,
        eps_decay_steps=eps_decay_steps,
        target_update_freq=target_update_freq,
        device=device,
    )

    if resume_checkpoint:
        log.info("Cargando checkpoint: %s", resume_checkpoint)
        agent.load(resume_checkpoint)

    # Ventanas deslizantes para estadísticas
    loss_window: deque[float] = deque(maxlen=log_every)
    result_window: deque[int | None] = deque(maxlen=log_every)

    first_checkpoint_saved = False
    total_transitions = 0
    learn_calls = 0
    t0 = time.time()

    log.info(
        "Iniciando self-play: %d episodios | batch=%d | buffer=%d | device=%s",
        episodes, batch_size, buffer_capacity, device,
    )

    for ep in range(1, episodes + 1):
        # --- Jugar episodio ---
        ep_info = play_episode(agent, max_steps=max_steps_per_episode)
        total_transitions += ep_info["transitions"]
        result_window.append(ep_info["result"])

        # --- Aprender cada `learn_every` transiciones generadas ---
        learn_calls_this_ep = max(1, ep_info["transitions"] // learn_every)
        for _ in range(learn_calls_this_ep):
            loss = agent.learn()
            if loss is not None:
                loss_window.append(loss)
                learn_calls += 1

        # --- Guardar primer checkpoint cuando el buffer esté lleno ---
        if not first_checkpoint_saved and len(agent.buffer) >= batch_size:
            path = os.path.join(checkpoint_dir, "checkpoint_first.pt")
            agent.save(path)
            first_checkpoint_saved = True
            log.info("★ Primer checkpoint guardado en '%s' (buffer=%d)",
                     path, len(agent.buffer))

            # Verificar gradientes tras el primer learn
            grad_info = check_gradients(agent)
            if grad_info["all_grads_flow"]:
                log.info(
                    "✓ Gradientes OK: %d/%d parámetros | grad_norm_max=%.4f",
                    grad_info["params_with_grad"],
                    grad_info["total_params"],
                    grad_info["max_grad_norm"],
                )
            else:
                log.warning(
                    "✗ Gradientes NO fluyen en %d/%d parámetros",
                    grad_info["total_params"] - grad_info["params_with_grad"],
                    grad_info["total_params"],
                )

        # --- Log periódico ---
        if ep % log_every == 0:
            elapsed = time.time() - t0
            avg_loss = sum(loss_window) / len(loss_window) if loss_window else float("nan")

            wins   = sum(1 for r in result_window if r ==  1)
            losses = sum(1 for r in result_window if r == -1)
            draws  = sum(1 for r in result_window if r ==  0)
            trunc  = sum(1 for r in result_window if r is None)

            log.info(
                "Ep %5d/%d | ε=%.3f | loss=%.5f | "
                "W/L/D/T=%d/%d/%d/%d | buf=%d | learn_steps=%d | %.1fs",
                ep, episodes,
                agent.epsilon,
                avg_loss,
                wins, losses, draws, trunc,
                len(agent.buffer),
                agent.learn_steps,
                elapsed,
            )

            # Verificar tendencia de la pérdida cada `log_every` episodios
            if len(loss_window) >= log_every // 2:
                first_half  = list(loss_window)[: len(loss_window) // 2]
                second_half = list(loss_window)[len(loss_window) // 2 :]
                avg_first   = sum(first_half)  / len(first_half)
                avg_second  = sum(second_half) / len(second_half)
                trend = "↓ bajando" if avg_second < avg_first else "↑ subiendo / estable"
                log.info("  Loss trend: %.5f → %.5f  %s", avg_first, avg_second, trend)

        # --- Checkpoint periódico ---
        if ep % checkpoint_every == 0:
            path = os.path.join(checkpoint_dir, f"checkpoint_ep{ep:06d}.pt")
            agent.save(path)
            log.info("Checkpoint guardado: %s", path)

    # --- Checkpoint final ---
    final_path = os.path.join(checkpoint_dir, "checkpoint_final.pt")
    agent.save(final_path)
    total_time = time.time() - t0
    log.info(
        "Entrenamiento completado en %.1fs | %d episodios | %d transiciones | "
        "%d pasos de aprendizaje | checkpoint final: %s",
        total_time, episodes, total_transitions, agent.learn_steps, final_path,
    )
    return agent


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Self-play DQN training loop para Damas"
    )
    p.add_argument("--episodes",          type=int,   default=1_000)
    p.add_argument("--max-steps",         type=int,   default=300,
                   dest="max_steps_per_episode")
    p.add_argument("--learn-every",       type=int,   default=4)
    p.add_argument("--log-every",         type=int,   default=50)
    p.add_argument("--checkpoint-dir",    type=str,   default="models")
    p.add_argument("--checkpoint-every",  type=int,   default=200)
    p.add_argument("--checkpoint",        type=str,   default=None,
                   dest="resume_checkpoint",
                   help="Ruta a un checkpoint para continuar entrenamiento")
    p.add_argument("--device",            type=str,   default="cpu")
    p.add_argument("--gamma",             type=float, default=0.99)
    p.add_argument("--lr",                type=float, default=1e-3)
    p.add_argument("--batch-size",        type=int,   default=64)
    p.add_argument("--buffer-capacity",   type=int,   default=50_000)
    p.add_argument("--eps-start",         type=float, default=1.0)
    p.add_argument("--eps-end",           type=float, default=0.05)
    p.add_argument("--eps-decay-steps",   type=int,   default=50_000)
    p.add_argument("--target-update-freq",type=int,   default=1_000)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        episodes=args.episodes,
        max_steps_per_episode=args.max_steps_per_episode,
        learn_every=args.learn_every,
        log_every=args.log_every,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        resume_checkpoint=args.resume_checkpoint,
        device=args.device,
        gamma=args.gamma,
        lr=args.lr,
        batch_size=args.batch_size,
        buffer_capacity=args.buffer_capacity,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay_steps=args.eps_decay_steps,
        target_update_freq=args.target_update_freq,
    )
# Desde src/
# python selfplay.py --episodes 1000 --device cpu

# Con CUDA 
# python selfplay.py --episodes 5000 --device cuda --batch-size 128

# Continuar desde un checkpoint
# python selfplay.py --checkpoint checkpoints/checkpoint_ep000200.pt