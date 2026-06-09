from __future__ import annotations
from typing import Any

State  = dict[str, Any]
Action = tuple[int, ...]
 
def _build_neighbors() -> list[dict[str, int | None]]:
    """Para cada casilla devuelve sus 4 vecinos diagonales (None = fuera)."""
    neighbors: list[dict[str, int | None]] = []
    for sq in range(32):
        row = sq // 4
        col_in_row = sq % 4          # 0..3, índice dentro de la fila
        even_row = (row % 2 == 0)
 
        if row == 0:
            ul = ur = None
        elif even_row:              
            ul = sq - 4             
            ur = sq - 3
        else:                      
            ul = sq - 5
            ur = sq - 4
 
        if row == 7:
            dl = dr = None
        elif even_row:
            dl = sq + 4
            dr = sq + 5
        else:
            dl = sq + 3
            dr = sq + 4
 
        if col_in_row == 0:
            if even_row:
                pass         
            else:
                ul = None
                dl = None
        if col_in_row == 3:
            if even_row:
                ur = None
                dr = None
 
        neighbors.append({"ul": ul, "ur": ur, "dl": dl, "dr": dr})
    return neighbors
 
NEIGHBORS = _build_neighbors()

_JUMP_OVER: dict[tuple[int, int], int] = {}
for _sq in range(32):
    for _d, _mid in NEIGHBORS[_sq].items():
        if _mid is None:
            continue
        _land = NEIGHBORS[_mid].get(_d)
        if _land is not None:
            _JUMP_OVER[(_sq, _land)] = _mid
 
def _promotion_row(turn: int) -> set[int]:
    """Casillas de la última fila para el jugador dado."""
    if turn == 1:   
        return {28, 29, 30, 31}
    else:           
        return {0, 1, 2, 3}

def _move_dirs(piece: int) -> list[str]:
    """Direcciones de movimiento (sin captura) para una pieza."""
    if abs(piece) == 2:                    # dama
        return ["ul", "ur", "dl", "dr"]
    return ["dl", "dr"] if piece > 0 else ["ul", "ur"]
 
 
def _capture_dirs(_piece: int) -> list[str]:
    """Las damas y las piezas normales capturan en las 4 diagonales."""
    return ["ul", "ur", "dl", "dr"]

def _capture_sequences(
    sq: int,
    piece: int,
    board: list[int],
    captured: frozenset[int],
    path: tuple[int, ...],
) -> list[tuple[int, ...]]:
    turn      = 1 if piece > 0 else -1
    prom_row  = _promotion_row(turn)
    sequences = []
 
    for d in _capture_dirs(piece):
        mid = NEIGHBORS[sq].get(d)
        if mid is None:
            continue
        land = NEIGHBORS[mid].get(d)
        if land is None:
            continue
 
        enemy = board[mid]
        if enemy == 0 or (enemy > 0) == (piece > 0) or mid in captured:
            continue
        if board[land] != 0 and land not in captured:
            if land not in captured:
                continue
        new_captured = captured | {mid}
 
        landed_piece = piece
        just_promoted = False
        if abs(piece) == 1 and land in prom_row:
            landed_piece   = 2 * turn
            just_promoted  = True
 
        new_path = path + (land,)
 
        # Intenta continuar la cadena 
        if not just_promoted:
            continuations = _capture_sequences(
                land, landed_piece, board, new_captured, new_path
            )
        else:
            continuations = []
 
        if continuations:
            sequences.extend(continuations)
        else:
            sequences.append(new_path)
 
    return sequences
 
def legal_moves(state: State) -> list[Action]:
    """
    Devuelve todos los movimientos legales y si existe al menos una captura posible,
    se devuelven solo capturas 
    """
    board = state["board"]
    turn  = state["turn"]
 
    captures: list[Action] = []
    simple:   list[Action] = []
 
    for sq in range(32):
        piece = board[sq]
        if piece == 0 or (piece > 0) != (turn > 0):
            continue
 
        # --- capturas desde esta casilla ---
        seqs = _capture_sequences(sq, piece, board, frozenset(), (sq,))
        captures.extend(seqs)
 
        # --- movimientos simples ---
        for d in _move_dirs(piece):
            dest = NEIGHBORS[sq].get(d)
            if dest is not None and board[dest] == 0:
                simple.append((sq, dest))
 
    return captures if captures else simple

def step(state: State, action: Action) -> State:

    legal = legal_moves(state)
    if action not in legal:
        raise ValueError(f"Acción ilegal: {action}")
 
    board = list(state["board"])
    turn  = state["turn"]
    src = action[0]
    piece = board[src]
    board[src] = 0

    is_capture = len(action) >= 3 or (
        # captura de un solo salto mientras la distancia en el tablero es 2
        len(action) == 2 and (action[1], action[0]) in _JUMP_OVER or
        (action[0], action[1]) in _JUMP_OVER
    )

    is_capture = len(action) >= 2 and all(
        (action[i], action[i + 1]) in _JUMP_OVER
        for i in range(len(action) - 1)
    )
    # Elimina piezas capturadas
    for i in range(len(action) - 1):
        mid = _JUMP_OVER.get((action[i], action[i + 1]))
        if mid is not None:
            board[mid] = 0
    dst = action[-1]
    if dst in _promotion_row(turn) and abs(piece) == 1:
        piece = 2 * turn   # 2 o -2
 
    board[dst] = piece
    no_capture_count = 0 if is_capture else state["no_capture_count"] + 1
    pos_key   = _position_key(board, -turn)
    history   = dict(state.get("position_history", {}))
    history[pos_key] = history.get(pos_key, 0) + 1
    return {
        "board": board,
        "turn": -turn,
        "no_capture_count": no_capture_count,
        "position_history": history,
    }
 

def is_terminal(state: State) -> bool:
    if state["no_capture_count"] >= 80:
        return True
    if _threefold_repetition(state):
        return True
    return len(legal_moves(state)) == 0
 
 
def result(state: State) -> int | None:
    if not is_terminal(state):
        return None
    if state["no_capture_count"] >= 80:
        return 0
    if _threefold_repetition(state):
        return 0
    return -state["turn"]

def _position_key(board: list[int], turn: int) -> tuple:
    return (tuple(board), turn)
 
 
def _threefold_repetition(state: State) -> bool:
    history = state.get("position_history", {})
    return any(v >= 3 for v in history.values())

def encode(state: State) -> list[float]:
    board = state["board"]
    turn  = state["turn"]
    channels = [
        [1.0 if v ==  1 else 0.0 for v in board],  # piezas rojas 
        [1.0 if v ==  2 else 0.0 for v in board],  # damas rojas
        [1.0 if v == -1 else 0.0 for v in board],  # piezas negras 
        [1.0 if v == -2 else 0.0 for v in board],  # damas negras
        [float(turn)] * 32,                        # turno (1 o -1)
    ]
    return [x for ch in channels for x in ch]

def initial_state() -> State:
    board = [0] * 32
    for i in range(12):
        board[i] = 1
    for i in range(20, 32):
        board[i] = -1
    return {"board": board, "turn": 1, "no_capture_count": 0}

def empty_state_for_test(turn: int = 1) -> State:
    return {
        "board":            [0] * 32,
        "turn":             turn,
        "no_capture_count": 0,
        "position_history": {},
    }
