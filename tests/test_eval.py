"""
Tests para src/eval/run.py (issue #15).

Usa un agente aleatorio como sustituto del DQN para que los tests
no dependan de torch ni de un checkpoint real.
"""
import csv
import os
import random
import tempfile

import pytest

from damas.engine import legal_moves
from tournament.tournament import run_tournament
from eval.run import DepthResult, _print_matrix, _save_csv, _NamedDQN


# ---------------------------------------------------------------------------
# Agente aleatorio reutilizable
# ---------------------------------------------------------------------------

class _FakeDQN:
    """Simula DQNAgent sin torch: act() devuelve un movimiento aleatorio."""

    def __init__(self, seed: int = 0) -> None:
        self.learn_steps = 9999
        self._rng = random.Random(seed)

    def act(self, state, greedy: bool = False):
        moves = legal_moves(state)
        return self._rng.choice(moves) if moves else None

    def load(self, path: str) -> None:
        pass


# ---------------------------------------------------------------------------
# Tests de DepthResult
# ---------------------------------------------------------------------------

def test_depth_result_win_rate_range():
    r = DepthResult(depth=3, dqn_wins=7, mm_wins=11, draws=2, total=20, win_rate=0.35)
    assert 0.0 <= r.win_rate <= 1.0


def test_depth_result_counts_consistent():
    r = DepthResult(depth=4, dqn_wins=8, mm_wins=9, draws=3, total=20, win_rate=0.4)
    assert r.dqn_wins + r.mm_wins + r.draws == r.total


# ---------------------------------------------------------------------------
# Tests de _NamedDQN
# ---------------------------------------------------------------------------

def test_named_dqn_label():
    fake = _FakeDQN()
    wrapper = _NamedDQN(fake, label="DQN(test)")
    assert wrapper.name == "DQN(test)"


def test_named_dqn_choose_action_legal():
    from damas.engine import initial_state
    fake    = _FakeDQN(seed=7)
    wrapper = _NamedDQN(fake, label="DQN")
    state   = initial_state()
    action  = wrapper.choose_action(state)
    assert action in legal_moves(state)


# ---------------------------------------------------------------------------
# Tests de salidas
# ---------------------------------------------------------------------------

def test_print_matrix_no_crash(capsys):
    results = [
        DepthResult(3, 10, 8, 2, 20, 0.50),
        DepthResult(4, 6,  12, 2, 20, 0.30),
    ]
    _print_matrix(results, "DQN(test)")
    out = capsys.readouterr().out
    assert "3" in out
    assert "4" in out
    assert "DQN(test)" in out


def test_save_csv_structure():
    results = [
        DepthResult(3, 10, 8, 2, 20, 0.50),
        DepthResult(4, 6,  12, 2, 20, 0.30),
        DepthResult(5, 4,  14, 2, 20, 0.20),
        DepthResult(6, 2,  16, 2, 20, 0.10),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "eval.csv")
        _save_csv(results, path, "DQN(test)")
        assert os.path.exists(path)
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 4
        assert set(rows[0].keys()) == {
            "dqn_label", "depth", "dqn_wins", "mm_wins",
            "draws", "total", "win_rate",
        }
        assert rows[0]["depth"] == "3"


def test_save_csv_win_rate_values():
    results = [DepthResult(3, 10, 8, 2, 20, 0.50)]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "eval.csv")
        _save_csv(results, path, "DQN")
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        assert float(rows[0]["win_rate"]) == pytest.approx(0.50)


# ---------------------------------------------------------------------------
# Test de FileNotFoundError
# ---------------------------------------------------------------------------

def test_run_eval_missing_checkpoint():
    from eval.run import run_eval
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        run_eval(checkpoint="nonexistent/model.pt", n_games=2)
