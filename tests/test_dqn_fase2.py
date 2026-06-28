"""Tests de la Fase 2 del DQN (issue #42): soft target update + currículum.

Cubre:
  - Polyak / soft update de la red objetivo (y copia dura por defecto).
  - Currículum: play_episode usa el oponente en su turno y guarda TODAS las transiciones.
"""
import pytest

torch = pytest.importorskip("torch")

from damas.engine import initial_state, legal_moves
from agents.dqn import DQNAgent
from selfplay import play_episode


# --------------------------------------------------------------------------- #
# Soft target update (Polyak)
# --------------------------------------------------------------------------- #

def _fill(net, value):
    with torch.no_grad():
        for p in net.parameters():
            p.fill_(value)


def test_soft_update_mueve_target_hacia_online():
    agent = DQNAgent(soft_tau=0.5)
    _fill(agent.online, 1.0)
    _fill(agent.target, 0.0)
    agent.update_target()                         # target ← 0.5*online + 0.5*target = 0.5
    for p in agent.target.parameters():
        assert torch.allclose(p, torch.full_like(p, 0.5))


def test_hard_update_copia_exacto_por_defecto():
    agent = DQNAgent(soft_tau=None)
    _fill(agent.online, 2.0)
    _fill(agent.target, 0.0)
    agent.update_target()                         # copia dura -> target == online
    for p in agent.target.parameters():
        assert torch.allclose(p, torch.full_like(p, 2.0))


def test_soft_update_no_iguala_de_golpe():
    """Con τ pequeño la target se acerca poco a poco, no de golpe."""
    agent = DQNAgent(soft_tau=0.01)
    _fill(agent.online, 1.0)
    _fill(agent.target, 0.0)
    agent.update_target()
    for p in agent.target.parameters():
        assert torch.allclose(p, torch.full_like(p, 0.01))   # solo 1% del camino


# --------------------------------------------------------------------------- #
# Currículum vs oponente externo
# --------------------------------------------------------------------------- #

class _RecordingOpponent:
    def __init__(self):
        self.calls = 0

    def choose_action(self, state):
        self.calls += 1
        return legal_moves(state)[0]


def test_curriculum_usa_oponente_y_guarda_todas_las_transiciones():
    agent = DQNAgent(buffer_capacity=2000)
    opp = _RecordingOpponent()
    before = len(agent.buffer)
    info = play_episode(agent, max_steps=60, opponent=opp, agent_color=1)
    assert opp.calls > 0                                   # el oponente jugó sus turnos
    assert info["transitions"] > 0
    # se guardan TODAS las transiciones (también las del oponente)
    assert len(agent.buffer) - before == info["transitions"]


def test_sin_oponente_es_autojuego():
    """Sin opponent, play_episode no falla y guarda transiciones (auto-juego)."""
    agent = DQNAgent(buffer_capacity=2000)
    info = play_episode(agent, max_steps=60)
    assert info["transitions"] > 0
    assert len(agent.buffer) == info["transitions"]
