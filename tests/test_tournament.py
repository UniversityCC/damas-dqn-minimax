"""
Tests for src/tournament/tournament.py.

Usa un agente aleatorio para que los tests sean rápidos y deterministas.
"""
import csv
import os
import random
import tempfile

from damas import legal_moves
from tournament.tournament import play_one_game, run_tournament, TournamentStats


class _RandomAgent:
    """Agente que elige movimientos al azar. Rápido para tests."""
    def __init__(self, name: str = "Random", seed: int | None = None) -> None:
        self.name = name
        self._rng  = random.Random(seed)

    def choose_action(self, state):
        moves = legal_moves(state)
        return self._rng.choice(moves) if moves else None


# ---------------------------------------------------------------------------

def test_play_one_game_returns_valid_result():
    a = _RandomAgent("A", seed=0)
    b = _RandomAgent("B", seed=1)
    res, half_moves = play_one_game(a, b)
    assert res in (-1, 0, 1)
    assert half_moves >= 1


def test_play_one_game_positive_half_moves():
    a = _RandomAgent(seed=42)
    b = _RandomAgent(seed=99)
    _, half_moves = play_one_game(a, b)
    assert half_moves > 0


def test_run_tournament_game_count():
    a = _RandomAgent("A", seed=0)
    b = _RandomAgent("B", seed=1)
    stats = run_tournament(a, b, n_games=4, verbose=False)
    assert stats.total == 4


def test_run_tournament_wins_plus_draws_equals_total():
    a = _RandomAgent("A", seed=7)
    b = _RandomAgent("B", seed=8)
    stats = run_tournament(a, b, n_games=6, verbose=False)
    assert stats.agent_a_wins + stats.agent_b_wins + stats.draws == stats.total


def test_run_tournament_color_alternation():
    """En partidas impares agent_a juega de rojo; en pares de negro."""
    a = _RandomAgent("A", seed=3)
    b = _RandomAgent("B", seed=4)
    stats = run_tournament(a, b, n_games=4, verbose=False)
    for g in stats.games:
        if g.game % 2 == 1:
            assert g.red_agent   == "A"
            assert g.black_agent == "B"
        else:
            assert g.red_agent   == "B"
            assert g.black_agent == "A"


def test_run_tournament_saves_csv():
    a = _RandomAgent("A", seed=5)
    b = _RandomAgent("B", seed=6)
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "results.csv")
        run_tournament(a, b, n_games=3, csv_path=csv_path, verbose=False)
        assert os.path.exists(csv_path)
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3
        assert set(rows[0].keys()) == {"game", "red_agent", "black_agent", "result", "half_moves"}


def test_tournament_stats_summary(capsys):
    a = _RandomAgent("Alfa", seed=1)
    b = _RandomAgent("Beta", seed=2)
    stats = run_tournament(a, b, n_games=4, verbose=False)
    stats.print_summary()
    captured = capsys.readouterr().out
    assert "Alfa" in captured
    assert "Beta" in captured
