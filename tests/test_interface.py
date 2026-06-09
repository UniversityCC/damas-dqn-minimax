"""Verifica el contrato público del motor de Damas (issue #1).

No prueba la lógica del juego (eso llega con los issues #4-#6), solo que la
interfaz se importa correctamente y que el estado inicial cumple lo documentado
en src/damas/README.md.
"""
import damas


def test_public_api_exists():
    for fn in ("initial_state", "legal_moves", "step", "is_terminal", "result", "encode"):
        assert hasattr(damas, fn), f"falta la función pública '{fn}' en la interfaz del motor"


def test_initial_state_shape():
    s = damas.initial_state()
    assert set(s) == {"board", "turn", "no_capture_count"}
    assert len(s["board"]) == 32
    assert s["turn"] == 1                 # rojo mueve primero
    assert s["no_capture_count"] == 0


def test_initial_state_piece_layout():
    board = damas.initial_state()["board"]
    assert all(board[i] == 1 for i in range(0, 12)), "rojas en casillas 0-11"
    assert all(board[i] == 0 for i in range(12, 20)), "franja central vacía"
    assert all(board[i] == -1 for i in range(20, 32)), "negras en casillas 20-31"
