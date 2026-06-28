"""Tests del promediado de pesos (SWA) — issue #42."""
import pytest

torch = pytest.importorskip("torch")

from damas.engine import initial_state, legal_moves
from agents.dqn import DQNAgent
from swa import average_state_dicts, average_checkpoints


def test_average_state_dicts_promedia_elemento_a_elemento():
    a = {"w": torch.tensor([0.0, 2.0])}
    b = {"w": torch.tensor([4.0, 0.0])}
    avg = average_state_dicts([a, b])
    assert torch.allclose(avg["w"], torch.tensor([2.0, 1.0]))


def test_average_state_dicts_vacio_falla():
    with pytest.raises(ValueError):
        average_state_dicts([])


def test_average_checkpoints_promedia_y_es_cargable(tmp_path):
    a1, a2 = DQNAgent(), DQNAgent()
    with torch.no_grad():
        for p in a1.online.parameters():
            p.fill_(0.0)
        for p in a2.online.parameters():
            p.fill_(2.0)
    a1.target.load_state_dict(a1.online.state_dict())
    a2.target.load_state_dict(a2.online.state_dict())
    p1, p2 = tmp_path / "a.pt", tmp_path / "b.pt"
    a1.save(str(p1)); a2.save(str(p2))

    avg = average_checkpoints([str(p1), str(p2)])
    for v in avg["online"].values():                      # promedio de 0 y 2 = 1
        assert torch.allclose(v, torch.ones_like(v))

    out = tmp_path / "swa.pt"
    torch.save(avg, str(out))
    agent = DQNAgent()
    agent.load(str(out))                                  # debe cargar sin desajustes
    action = agent.act(initial_state(), greedy=True)
    assert action in legal_moves(initial_state())


def test_average_checkpoints_requiere_al_menos_uno():
    with pytest.raises(ValueError):
        average_checkpoints([])
