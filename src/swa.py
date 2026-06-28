"""Promediado de pesos (Stochastic Weight Averaging) de checkpoints DQN.

Combina varios checkpoints en un único modelo promediando sus pesos. Útil porque
el entrenamiento por auto-juego oscila y las fortalezas quedan repartidas en
checkpoints distintos: el promedio de varios buenos suele ser mejor y más estable
que cualquiera individual (issue #42).

Importante: promediar pesos solo tiene sentido entre checkpoints de la MISMA
trayectoria/linaje (mismo "valle"); promediar uno malo arrastra al conjunto.

Uso:
    python src/swa.py --checkpoints a.pt b.pt c.pt --out swa.pt
"""
from __future__ import annotations

import argparse

import torch


def average_state_dicts(state_dicts: list[dict]) -> dict:
    """Promedia una lista de state_dict (todos con las mismas claves)."""
    if not state_dicts:
        raise ValueError("Se requiere al menos un state_dict")
    keys = state_dicts[0].keys()
    return {k: sum(sd[k] for sd in state_dicts) / len(state_dicts) for k in keys}


def average_checkpoints(paths: list[str], map_location: str = "cpu") -> dict:
    """Carga varios checkpoints y devuelve uno nuevo con los pesos promediados.

    Promedia la red ``online``; la red ``target`` se sincroniza con ese promedio.
    Conserva el optimizer del último checkpoint y el mayor ``learn_steps`` (solo
    para poder reanudar; para inferencia/evaluación únicamente importa ``online``).
    El checkpoint resultante es compatible con ``DQNAgent.load``.
    """
    if not paths:
        raise ValueError("Se requiere al menos un checkpoint")
    ckpts = [torch.load(p, map_location=map_location) for p in paths]
    online_avg = average_state_dicts([c["online"] for c in ckpts])
    return {
        "online": online_avg,
        "target": {k: v.clone() for k, v in online_avg.items()},
        "optimizer": ckpts[-1]["optimizer"],
        "learn_steps": max(c["learn_steps"] for c in ckpts),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Promedia (SWA) varios checkpoints DQN en uno solo"
    )
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="Rutas de los checkpoints a promediar (2 o más)")
    p.add_argument("--out", required=True,
                   help="Ruta de salida del checkpoint promediado")
    args = p.parse_args()
    avg = average_checkpoints(args.checkpoints)
    torch.save(avg, args.out)
    print(f"SWA de {len(args.checkpoints)} checkpoints -> {args.out}")


if __name__ == "__main__":
    main()
