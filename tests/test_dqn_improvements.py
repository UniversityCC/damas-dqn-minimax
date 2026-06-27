"""Tests de las mejoras del DQN — Fase 1 (issue #40).

Cubre:
  - Reward shaping (captura y coronación) desde la perspectiva del que mueve.
  - Double DQN: la acción se selecciona con la red online y se evalúa con la objetivo.
  - Compatibilidad de checkpoints: cargar pesos previos con double_dqn activado.
"""
import pytest

torch = pytest.importorskip("torch")

from damas.engine import (
    initial_state, legal_moves, step, is_terminal,
    empty_state_for_test, _JUMP_OVER, _promotion_row,
)
from model.action_space import NUM_ACTIONS
from agents.dqn import DQNAgent
from selfplay import _shaping_reward


# --------------------------------------------------------------------------- #
# Reward shaping
# --------------------------------------------------------------------------- #

def test_shaping_capture_cuenta_piezas():
    (src, land), _mid = next(iter(_JUMP_OVER.items()))
    state = empty_state_for_test(turn=1)
    state["board"][src] = 1                       # peón propio (no corona)
    r = _shaping_reward(state, (src, land), capture_reward=0.1, king_reward=0.0)
    assert r == pytest.approx(0.1)                # un salto = una pieza capturada


def test_shaping_premia_coronacion_de_peon():
    turn = 1
    prom = sorted(_promotion_row(turn))[0]
    state = empty_state_for_test(turn=turn)
    state["board"][prom - 4] = 1                  # peón que llega a la fila de coronación
    r = _shaping_reward(state, (prom - 4, prom), capture_reward=0.0, king_reward=0.5)
    assert r == pytest.approx(0.5)


def test_shaping_no_premia_si_ya_es_dama():
    turn = 1
    prom = sorted(_promotion_row(turn))[0]
    state = empty_state_for_test(turn=turn)
    state["board"][prom - 4] = 2                  # ya es dama -> no debe premiar
    assert _shaping_reward(state, (prom - 4, prom), 0.0, 0.5) == pytest.approx(0.0)


def test_shaping_desactivado_es_cero():
    (src, land), _mid = next(iter(_JUMP_OVER.items()))
    state = empty_state_for_test(turn=1)
    state["board"][src] = 1
    assert _shaping_reward(state, (src, land), 0.0, 0.0) == 0.0


# --------------------------------------------------------------------------- #
# Double DQN
# --------------------------------------------------------------------------- #

class _StubNet(torch.nn.Module):
    """Devuelve un vector de Q fijo, ignorando la entrada (para tests deterministas)."""
    def __init__(self, vec: torch.Tensor):
        super().__init__()
        self.vec = vec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vec.repeat(x.shape[0], 1)


def _mask_for(legal_indices):
    mask = torch.zeros((1, NUM_ACTIONS), dtype=torch.bool)
    for i in legal_indices:
        mask[0, i] = True
    return mask


def test_double_dqn_selecciona_online_evalua_target():
    agent = DQNAgent(double_dqn=True)
    q_online = torch.full((1, NUM_ACTIONS), -1.0); q_online[0, 5] = 10.0   # online prefiere 5
    q_target = torch.zeros((1, NUM_ACTIONS)); q_target[0, 5] = 3.0; q_target[0, 7] = 99.0
    agent.online = _StubNet(q_online)
    agent.target = _StubNet(q_target)
    best = agent._bootstrap_values(torch.zeros((1, 160)), _mask_for([5, 7]))
    # online elige la acción 5; la target la evalúa en 3.0 (NO su máximo 99 en 7)
    assert best.item() == pytest.approx(3.0)


def test_single_dqn_usa_max_de_target():
    agent = DQNAgent(double_dqn=False)
    q_online = torch.full((1, NUM_ACTIONS), -1.0); q_online[0, 5] = 10.0
    q_target = torch.zeros((1, NUM_ACTIONS)); q_target[0, 5] = 3.0; q_target[0, 7] = 99.0
    agent.online = _StubNet(q_online)
    agent.target = _StubNet(q_target)
    best = agent._bootstrap_values(torch.zeros((1, 160)), _mask_for([5, 7]))
    # sin Double DQN: máximo legal de la red objetivo (acción 7)
    assert best.item() == pytest.approx(99.0)


def test_bootstrap_cero_si_no_hay_jugadas_legales():
    agent = DQNAgent(double_dqn=True)
    best = agent._bootstrap_values(torch.zeros((1, 160)), _mask_for([]))
    assert best.item() == pytest.approx(0.0)


def test_learn_corre_en_ambos_modos():
    for dd in (True, False):
        agent = DQNAgent(batch_size=8, buffer_capacity=200, double_dqn=dd)
        s = initial_state()
        for _ in range(30):
            a = legal_moves(s)[0]
            ns = step(s, a)
            agent.remember(s, a, 0.0, ns, is_terminal(ns))
            s = initial_state() if is_terminal(ns) else ns
        loss = agent.learn()
        assert isinstance(loss, float)


# --------------------------------------------------------------------------- #
# Compatibilidad de checkpoints
# --------------------------------------------------------------------------- #

def test_checkpoint_carga_con_double_dqn(tmp_path):
    """Un checkpoint guardado sin el flag debe cargar con double_dqn=True (misma red)."""
    a1 = DQNAgent(double_dqn=False)
    path = tmp_path / "ck.pt"
    a1.save(str(path))

    a2 = DQNAgent(double_dqn=True)
    a2.load(str(path))                                   # no debe lanzar por dimensiones
    action = a2.act(initial_state(), greedy=True)
    assert action in legal_moves(initial_state())
