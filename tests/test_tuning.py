"""
Tests del ajuste de hiperparámetros y comparación arquitectónica (issue #13).

Cubre:
  - QNetwork con arquitecturas variables y retrocompatibilidad de la red por defecto.
  - DQNAgent construido con una arquitectura personalizada.
  - El runner de tuning (configs, ejecución corta, selección del mejor y CSV).
"""
import csv

import pytest

torch = pytest.importorskip("torch")

from damas.engine import initial_state, legal_moves
from model.q_network import QNetwork, INPUT_SIZE
from model.action_space import NUM_ACTIONS
from agents.dqn import DQNAgent
from tuning.run import (
    TuneConfig, hparam_configs, arch_configs, run_tuning, _run_config,
)


# ---------------------------------------------------------------------------
# QNetwork: arquitecturas variables
# ---------------------------------------------------------------------------

def test_default_arch_is_512x2():
    """Sin argumentos la red es (512, 512): tres capas Linear como antes."""
    net = QNetwork()
    assert net.hidden == (512, 512)
    linears = [m for m in net.net if isinstance(m, torch.nn.Linear)]
    assert len(linears) == 3
    assert linears[0].in_features == INPUT_SIZE
    assert linears[-1].out_features == NUM_ACTIONS


def test_int_hidden_means_two_layers():
    """Un ``int`` se interpreta como dos capas ocultas de ese ancho (retrocompat)."""
    net = QNetwork(hidden=256)
    assert net.hidden == (256, 256)


def test_sequence_hidden_builds_one_layer_per_element():
    """Una secuencia produce una capa oculta por elemento."""
    net = QNetwork(hidden=(512, 512, 1024))
    linears = [m for m in net.net if isinstance(m, torch.nn.Linear)]
    assert len(linears) == 4  # 3 ocultas + salida
    assert linears[1].out_features == 512
    assert linears[2].out_features == 1024


def test_default_statedict_is_backward_compatible():
    """El state_dict de la red por defecto conserva las claves net.0/2/4."""
    keys = set(QNetwork().state_dict().keys())
    for k in ("net.0.weight", "net.2.weight", "net.4.weight"):
        assert k in keys


def test_forward_shapes_for_each_arch():
    """Cada arquitectura mapea (batch, 160) → (batch, 1024)."""
    x = torch.zeros(4, INPUT_SIZE)
    for hidden in [(256, 256), (512, 512), (512, 512, 1024)]:
        out = QNetwork(hidden=hidden)(x)
        assert out.shape == (4, NUM_ACTIONS)


# ---------------------------------------------------------------------------
# DQNAgent con arquitectura personalizada
# ---------------------------------------------------------------------------

def test_agent_uses_requested_arch():
    """El agente construye online y target con la arquitectura pedida."""
    agent = DQNAgent(hidden=(256, 256))
    assert agent.online.hidden == (256, 256)
    assert agent.target.hidden == (256, 256)


def test_agent_custom_arch_acts_legally():
    agent = DQNAgent(hidden=(512, 512, 1024))
    state = initial_state()
    assert agent.act(state, greedy=True) in legal_moves(state)


# ---------------------------------------------------------------------------
# Configuraciones del barrido
# ---------------------------------------------------------------------------

def test_hparam_sweep_is_one_factor_at_a_time():
    """Cada config del barrido difiere del baseline en a lo sumo un factor."""
    configs = hparam_configs()
    base = configs[0]
    assert base.name == "baseline"
    for cfg in configs[1:]:
        diffs = sum([
            cfg.lr != base.lr,
            cfg.gamma != base.gamma,
            cfg.buffer != base.buffer,
            cfg.target_div != base.target_div,
            cfg.hidden != base.hidden,
        ])
        assert diffs == 1, f"{cfg.name} cambia {diffs} factores, debe cambiar 1"


def test_arch_comparison_has_2_to_3_archs():
    archs = arch_configs()
    assert 2 <= len(archs) <= 3
    assert all(c.group == "arch" for c in archs)


# ---------------------------------------------------------------------------
# Runner (ejecución muy corta)
# ---------------------------------------------------------------------------

def test_run_config_returns_finite_metrics():
    cfg = TuneConfig("arch", "mlp-256x2", hidden=(256, 256))
    res = _run_config(cfg, steps=60, eval_games=2)
    assert 0.0 <= res.win_rate <= 1.0
    assert res.learn_steps >= 0
    assert res.hidden == "256x256"


def test_run_tuning_only_arch_and_csv(tmp_path):
    out = tmp_path / "tuning.csv"
    results = run_tuning(steps=60, eval_games=2, only="arch",
                         csv_path=str(out), verbose=False)
    assert len(results) == len(arch_configs())
    assert out.exists()
    with open(out, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(results)
    assert {"group", "name", "win_rate", "hidden"}.issubset(rows[0].keys())


def test_run_tuning_only_hparams_count():
    results = run_tuning(steps=40, eval_games=1, only="hparams", verbose=False)
    assert len(results) == len(hparam_configs())
    assert all(r.group == "hparams" for r in results)
