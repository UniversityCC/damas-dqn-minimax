"""Test de la evaluación estocástica (issue #42)."""
import pytest

torch = pytest.importorskip("torch")

from agents.dqn import DQNAgent
from eval.stochastic import evaluate_stochastic


def test_evaluate_stochastic_cuenta_todas_las_partidas():
    agent = DQNAgent()                                  # red sin entrenar; solo se mide el conteo
    r = evaluate_stochastic(agent, depth=3, games=4, eps=0.1, max_half=120)
    assert r["games"] == 4
    assert r["wins"] + r["losses"] + r["draws"] == 4   # cada partida cae en exactamente una
    assert 0.0 <= r["win_rate"] <= 1.0
    assert r["no_loss_rate"] >= r["win_rate"]           # no-pierde incluye empates
