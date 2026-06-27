"""
Tests para src/eval/final_tournament.py (issue #16).

Usa agentes aleatorios para no depender de torch ni checkpoints.
"""
import csv
import math
import os
import random
import tempfile

import pytest

from damas.engine import legal_moves, initial_state
from eval.final_tournament import (
    EloTracker, MatchupMetrics,
    _save_matchups_csv, _save_elo_csv,
    _matchup_metrics,
)
from tournament.tournament import run_tournament


# ---------------------------------------------------------------------------
# Agentes de prueba
# ---------------------------------------------------------------------------

class _RandAgent:
    def __init__(self, name: str, seed: int = 0) -> None:
        self.name = name
        self._rng = random.Random(seed)

    def choose_action(self, state):
        moves = legal_moves(state)
        return self._rng.choice(moves) if moves else None


# ---------------------------------------------------------------------------
# Tests de EloTracker
# ---------------------------------------------------------------------------

def test_elo_initial_ratings():
    elo = EloTracker(["A", "B", "C"], base=1500)
    assert elo.ratings["A"] == 1500
    assert elo.ratings["B"] == 1500


def test_elo_winner_gains_points():
    elo = EloTracker(["A", "B"], base=1500, k=32)
    before_a = elo.ratings["A"]
    before_b = elo.ratings["B"]
    elo.update("A", "B")
    assert elo.ratings["A"] > before_a
    assert elo.ratings["B"] < before_b


def test_elo_draw_symmetric():
    elo = EloTracker(["A", "B"], base=1500, k=32)
    elo.update("A", "B", draw=True)
    assert math.isclose(elo.ratings["A"], elo.ratings["B"], abs_tol=1e-6)


def test_elo_sum_preserved():
    elo = EloTracker(["A", "B"], base=1500, k=32)
    total_before = sum(elo.ratings.values())
    elo.update("A", "B")
    total_after = sum(elo.ratings.values())
    assert math.isclose(total_before, total_after, abs_tol=1e-6)


def test_elo_sorted_table_descending():
    elo = EloTracker(["A", "B", "C"], base=1500, k=32)
    elo.update("A", "B")
    elo.update("A", "C")
    table = elo.sorted_table()
    ratings = [r for _, r in table]
    assert ratings == sorted(ratings, reverse=True)


def test_elo_apply_stats():
    elo = EloTracker(["X", "Y"], base=1500, k=32)
    a = _RandAgent("X", seed=1)
    b = _RandAgent("Y", seed=2)
    stats = run_tournament(a, b, n_games=4, verbose=False)
    elo.apply_stats(stats)
    total = sum(elo.ratings.values())
    assert math.isclose(total, 3000.0, abs_tol=1e-4)


# ---------------------------------------------------------------------------
# Tests de MatchupMetrics
# ---------------------------------------------------------------------------

def test_matchup_metrics_from_stats():
    a = _RandAgent("A", seed=3)
    b = _RandAgent("B", seed=4)
    stats = run_tournament(a, b, n_games=6, verbose=False)
    m = _matchup_metrics(stats)
    assert m.a_wins + m.b_wins + m.draws == m.total
    assert m.total == 6
    assert 0.0 <= m.win_rate_a <= 1.0
    assert m.avg_half_moves > 0


# ---------------------------------------------------------------------------
# Tests de salidas CSV
# ---------------------------------------------------------------------------

def test_save_matchups_csv():
    matchups = [
        MatchupMetrics("A", "B", 3, 2, 1, 6, 45.0, 0.5),
        MatchupMetrics("A", "C", 4, 2, 0, 6, 50.0, 0.667),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "matchups.csv")
        _save_matchups_csv(matchups, path)
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert set(rows[0].keys()) == {
            "agent_a", "agent_b", "a_wins", "b_wins",
            "draws", "total", "win_rate_a", "avg_half_moves",
        }


def test_save_elo_csv():
    elo = EloTracker(["A", "B", "C"], base=1500, k=32)
    elo.update("A", "B")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "elo.csv")
        _save_elo_csv(elo, path)
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3
        assert set(rows[0].keys()) == {"rank", "agent", "elo"}
        ratings = [float(r["elo"]) for r in rows]
        assert ratings == sorted(ratings, reverse=True)


# ---------------------------------------------------------------------------
# Test de FileNotFoundError
# ---------------------------------------------------------------------------

def test_run_final_tournament_missing_checkpoint():
    from eval.final_tournament import run_final_tournament
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        run_final_tournament(checkpoint="no/existe.pt", n_games=2)
