from __future__ import annotations
from damas import legal_moves, State

MATERIAL_PIECE  = 1.0
MATERIAL_KING   = 3.0
WEIGHT_POS      = 0.1
WEIGHT_MOBILITY = 0.5

# Playable indices that sit at or near the board centre
CENTER_SQUARES = frozenset({13, 14, 17, 18})
NEAR_CENTER    = frozenset({9, 10, 11, 12, 19, 20, 21, 22})


def evaluate(state: State) -> float:
    """
    Heuristic evaluation from red's perspective (positive = good for red).

    Components
    ----------
    material  : weight 1.0 per piece, 3.0 per king
    positional: weight 0.1 — centre bonus + advancement for regular pieces
    mobility  : weight 0.5 — (red legal moves) - (black legal moves)
    """
    board = state["board"]

    # --- Material ---
    material = 0.0
    for piece in board:
        if piece == 1:
            material += MATERIAL_PIECE
        elif piece == -1:
            material -= MATERIAL_PIECE
        elif piece == 2:
            material += MATERIAL_KING
        elif piece == -2:
            material -= MATERIAL_KING

    # --- Positional ---
    positional = 0.0
    for sq, piece in enumerate(board):
        if piece == 0:
            continue
        sign = 1.0 if piece > 0 else -1.0
        r = sq // 4  # row 0-7; equals _idx_to_rc(sq)[0]

        if sq in CENTER_SQUARES:
            positional += sign * 1.0
        elif sq in NEAR_CENTER:
            positional += sign * 0.5

        # Advancement bonus: regular pieces gain value as they approach promotion
        if piece == 1:          # red advances toward row 7
            positional += r / 7.0
        elif piece == -1:       # black advances toward row 0
            positional -= (7 - r) / 7.0

    # --- Mobility ---
    red_moves   = len(legal_moves({**state, "turn":  1}))
    black_moves = len(legal_moves({**state, "turn": -1}))
    mobility = float(red_moves - black_moves)

    return material + WEIGHT_POS * positional + WEIGHT_MOBILITY * mobility
