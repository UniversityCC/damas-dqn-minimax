"""
Tests for agents/heuristic.py and MinimaxAgent decision-making.

Board index reference (new engine):
  even rows (0,2,4,6): indices use cols 0,2,4,6
  odd  rows (1,3,5,7): indices use cols 1,3,5,7
  row = idx // 4

Piece values: 0=empty  1=red  -1=black  2=red-king  -2=black-king

Capture geometry verified with NEIGHBORS:
  NEIGHBORS[8]["dr"]  = 13  (mid)
  NEIGHBORS[13]["dr"] = 17  (land)  → red@8 jumps black@13, lands at 17

Promotion geometry verified:
  NEIGHBORS[24]["dl"] = 28  row-7 square → red@24 can promote
  NEIGHBORS[24]["dr"] = 29  row-7 square → red@24 can promote
"""
import pytest
from damas import step, initial_state
from agents.heuristic import evaluate
from agents.minimax import MinimaxAgent


def _state(board: list, turn: int = 1) -> dict:
    return {"board": board, "turn": turn, "no_capture_count": 0}


# ---------------------------------------------------------------------------
# evaluate() tests
# ---------------------------------------------------------------------------

def test_evaluate_red_advantage():
    """More red pieces → positive score."""
    board = [0] * 32
    board[5]  =  1   # red
    board[10] =  1   # red
    board[15] =  1   # red
    board[25] = -1   # single black
    assert evaluate(_state(board)) > 0


def test_evaluate_black_advantage():
    """More black pieces → negative score."""
    board = [0] * 32
    board[5]  =  1   # single red
    board[20] = -1   # black
    board[25] = -1   # black
    board[26] = -1   # black
    assert evaluate(_state(board)) < 0


def test_evaluate_balanced():
    """
    1 red piece vs 1 black piece at equivalent positions → score near 0.
    idx 5  = (row 1, col 3)   red advances toward row 7
    idx 26 = (row 6, col 4)   black advances toward row 0
    Both at the same relative depth → net positional ≈ 0; mobility = 2 each.
    """
    board = [0] * 32
    board[5]  =  1
    board[26] = -1
    val = evaluate(_state(board))
    assert abs(val) < 2.0


def test_evaluate_king_beats_piece():
    """A red king should produce a higher score than a plain red piece."""
    board_king  = [0] * 32
    board_piece = [0] * 32
    board_king[5]  = 2    # red king
    board_piece[5] = 1    # red piece
    # Add matching black piece so mobility doesn't dominate
    board_king[20]  = -1
    board_piece[20] = -1
    assert evaluate(_state(board_king)) > evaluate(_state(board_piece))


# ---------------------------------------------------------------------------
# MinimaxAgent decision tests
# ---------------------------------------------------------------------------

def test_minimax_captures_when_available():
    """
    Captures are mandatory in Damas.  When a capture exists, the agent must
    take it — there is no other legal move.

    Setup (verified against NEIGHBORS):
      red  @ 8  → can jump "dr": over black@13 to land@17
      red  @ 0  → only simple moves (row 0, no enemies in range)
      black@ 13 → the target piece

    legal_moves returns [(8, 17)] only (captures override simples).
    """
    board = [0] * 32
    board[8]  =  1   # red – has a capture
    board[13] = -1   # black – will be captured
    board[0]  =  1   # red – only simple moves available
    state = _state(board)

    agent = MinimaxAgent(depth=3, player=1)
    action = agent.choose_action(state)

    next_state = step(state, action)
    assert next_state["board"][13] == 0, "Black piece at 13 should have been captured"


def test_minimax_promotes_to_king():
    """
    The agent should prefer promoting a piece over making a regular move because
    a king (value 3) is worth more than a piece (value 1).

    Setup (verified against NEIGHBORS):
      red  @ 24 (row 6) → can move only to 28 or 29 (both row 7 = promotion)
      red  @ 9  (row 2) → has four regular non-promotion moves
      black@ 20         → not reachable as a capture target from either red piece

    No captures are available; the agent must choose between 6 simple moves.
    Promotion gives +2 material immediately, so the agent should pick sq 24.
    """
    board = [0] * 32
    board[24] =  1   # red – about to promote
    board[9]  =  1   # red – regular moves only
    board[20] = -1   # black – keeps the game non-trivial
    state = _state(board)

    agent = MinimaxAgent(depth=3, player=1)
    action = agent.choose_action(state)

    assert action[0] == 24, "Agent should move the piece that can promote"
    assert action[-1] // 4 == 7, "Piece should end on the promotion row (row 7)"

    next_state = step(state, action)
    assert next_state["board"][action[-1]] == 2, "Piece should have become a king (value 2)"
