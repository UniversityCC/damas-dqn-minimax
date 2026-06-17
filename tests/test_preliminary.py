"""
Tests para src/eval/preliminary.py (issue #21).

Usa un agente aleatorio en lugar del DQN para que los tests
no dependan de torch ni de un checkpoint real.
"""
import csv
import os
import random
import tempfile

import pytest

from damas.engine import legal_moves, initial_state
from eval.preliminary import (
    MoveRecord, GameRecord, PreliminaryStats,
    _check_capture, _check_promotion,
    _print_metrics, _print_qualitative, _save_csv,
)


# ---------------------------------------------------------------------------
# Agente falso (no requiere torch)
# ---------------------------------------------------------------------------

class _FakeDQN:
    def __init__(self, seed: int = 0) -> None:
        self.learn_steps = 1000
        self._rng = random.Random(seed)

    def act(self, state, greedy: bool = False):
        moves = legal_moves(state)
        return self._rng.choice(moves) if moves else None


# ---------------------------------------------------------------------------
# Tests de helpers geometricos
# ---------------------------------------------------------------------------

def test_check_capture_simple_move():
    # Movimiento simple (2 casillas): no es captura
    from damas.engine import NEIGHBORS, _JUMP_OVER
    sq = 8
    nb = NEIGHBORS[sq]["dr"]   # 13
    assert not _check_capture((sq, nb))


def test_check_capture_jump():
    # Salto conocido: 8 -> 17 pasando por 13
    from damas.engine import _JUMP_OVER
    assert (8, 17) in _JUMP_OVER
    assert _check_capture((8, 17))


def test_check_promotion_red_row7():
    board = [0] * 32
    board[24] = 1   # pieza roja en fila 6
    assert _check_promotion((24, 28), board, turn=1)


def test_check_promotion_not_king():
    board = [0] * 32
    board[24] = 2   # ya es dama → no cuenta como promocion
    assert not _check_promotion((24, 28), board, turn=1)


# ---------------------------------------------------------------------------
# Tests de GameRecord
# ---------------------------------------------------------------------------

def test_game_record_reward():
    g = GameRecord(1, result=1,  half_moves=40, dqn_moves=20, dqn_total_ms=100.0)
    assert g.reward == 1.0
    g2 = GameRecord(2, result=-1, half_moves=30, dqn_moves=15, dqn_total_ms=60.0)
    assert g2.reward == -1.0


def test_game_record_avg_ms():
    g = GameRecord(1, result=0, half_moves=20, dqn_moves=10, dqn_total_ms=50.0)
    assert g.avg_ms_per_move == pytest.approx(5.0)


def test_game_record_avg_ms_zero_moves():
    g = GameRecord(1, result=0, half_moves=0, dqn_moves=0, dqn_total_ms=0.0)
    assert g.avg_ms_per_move == 0.0


# ---------------------------------------------------------------------------
# Tests de PreliminaryStats
# ---------------------------------------------------------------------------

def _make_stats(results: list[int]) -> PreliminaryStats:
    stats = PreliminaryStats(dqn_label="DQN", mm_label="Minimax(d=3)")
    for i, r in enumerate(results, 1):
        stats.games.append(
            GameRecord(i, result=r, half_moves=50, dqn_moves=25, dqn_total_ms=125.0)
        )
    return stats


def test_stats_win_rate():
    stats = _make_stats([1, 1, -1, 0])
    assert stats.wins == 2
    assert stats.losses == 1
    assert stats.draws == 1
    assert stats.win_rate == pytest.approx(0.5)


def test_stats_avg_reward():
    stats = _make_stats([1, -1, 0])
    assert stats.avg_reward == pytest.approx(0.0)


def test_stats_avg_half_moves():
    stats = _make_stats([1, -1])
    assert stats.avg_half_moves == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# Tests de salidas
# ---------------------------------------------------------------------------

def test_print_metrics_no_crash(capsys):
    stats = _make_stats([1, -1, 0, 1])
    _print_metrics(stats)
    out = capsys.readouterr().out
    assert "DQN" in out
    assert "Victorias" in out


def test_print_qualitative_no_crash(capsys):
    stats = _make_stats([1, -1])
    stats.games[0].moves.append(
        MoveRecord(1, 1, (8, 17), is_capture=True, is_promotion=False, elapsed_ms=5.0)
    )
    _print_qualitative(stats, n_examples=2)
    capsys.readouterr()


def test_save_csv_columns():
    stats = _make_stats([1, -1, 0])
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "pre.csv")
        _save_csv(stats, path)
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3
        assert set(rows[0].keys()) == {
            "game_id", "result", "reward", "half_moves",
            "dqn_moves", "avg_ms_per_move",
        }


# ---------------------------------------------------------------------------
# Test de FileNotFoundError
# ---------------------------------------------------------------------------

def test_run_preliminary_missing_checkpoint():
    from eval.preliminary import run_preliminary
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        run_preliminary(checkpoint="no/existe.pt", n_games=2)
