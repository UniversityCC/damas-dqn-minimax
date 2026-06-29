"""Tests del agente híbrido (negamax + alfa-beta con evaluador enchufable)."""
import pytest

from damas.engine import (
    initial_state, legal_moves, step, is_terminal, empty_state_for_test,
)
from agents.minimax import MinimaxAgent
from agents.hybrid import HybridAgent, heuristic_value_fn


def _positions(n: int = 8) -> list:
    """Batería de posiciones jugando un movimiento determinista desde el inicio."""
    states, s = [], initial_state()
    for _ in range(n):
        if is_terminal(s):
            s = initial_state()
        states.append(s)
        moves = legal_moves(s)
        s = step(s, moves[len(moves) // 2])
    return states


# --------------------------------------------------------------------------- #
# PRUEBA CLAVE: el negamax del híbrido equivale al Minimax del repo
# --------------------------------------------------------------------------- #

def test_hibrido_heuristica_equivale_a_minimax_repo():
    """Con la heurística como evaluador, el híbrido debe elegir el MISMO movimiento que
    MinimaxAgent a igual profundidad → confirma que el negamax es correcto."""
    mm = MinimaxAgent(depth=3)
    hy = HybridAgent(eval_fn=heuristic_value_fn(), depth=3,
                     use_ordering=False, use_cache=False)
    for s in _positions(8):
        if is_terminal(s):
            continue
        assert hy.choose_action(s) == mm.choose_action(s)


# --------------------------------------------------------------------------- #
# Correctitud y mecánica
# --------------------------------------------------------------------------- #

def test_term_value_mover_sin_jugadas_pierde():
    s = empty_state_for_test(turn=1)
    s["board"][0] = -1                       # solo una pieza negra; rojo (turn=1) sin jugadas
    assert is_terminal(s)
    assert HybridAgent._term(s) < 0          # el mover perdió


def test_term_value_empate_es_cero():
    s = empty_state_for_test(turn=1)
    s["no_capture_count"] = 80               # empate por regla de 80
    assert is_terminal(s)
    assert HybridAgent._term(s) == 0.0


def test_cache_evalua_una_vez_por_estado():
    calls = {"n": 0}
    def counting(_s):
        calls["n"] += 1
        return 1.0
    hy = HybridAgent(eval_fn=counting, depth=1, use_cache=True)
    s = initial_state()
    hy._leaf(s); hy._leaf(s)
    assert calls["n"] == 1                    # la caché evita la segunda evaluación


def test_ordering_no_cambia_el_valor():
    """La ordenación solo poda: el valor negamax es idéntico con y sin ordenación."""
    s = initial_state()
    on = HybridAgent(eval_fn=heuristic_value_fn(), depth=3, use_ordering=True, use_cache=False)
    off = HybridAgent(eval_fn=heuristic_value_fn(), depth=3, use_ordering=False, use_cache=False)
    _, v_on = on._search(s, 3)
    _, v_off = off._search(s, 3)
    assert abs(v_on - v_off) < 1e-9


def test_conteo_de_nodos_se_reinicia_por_jugada():
    hy = HybridAgent(eval_fn=heuristic_value_fn(), depth=2, use_cache=False)
    hy.choose_action(initial_state())
    n1 = hy.nodes
    hy.choose_action(initial_state())
    n2 = hy.nodes
    assert n1 > 0 and n2 > 0 and n1 == n2     # misma posición -> mismo nº de nodos (reiniciado)


def test_hibrido_heuristica_juega_legal():
    hy = HybridAgent(eval_fn=heuristic_value_fn(), depth=3)
    a = hy.choose_action(initial_state())
    assert a in legal_moves(initial_state())


def test_iterative_deepening_por_tiempo_juega_legal():
    hy = HybridAgent(eval_fn=heuristic_value_fn(), time_budget=0.05)
    a = hy.choose_action(initial_state())
    assert a in legal_moves(initial_state())


# --------------------------------------------------------------------------- #
# Evaluador DQN (requiere torch)
# --------------------------------------------------------------------------- #

def test_dqn_value_fn_y_hibrido_juegan_legal():
    pytest.importorskip("torch")
    from agents.dqn import DQNAgent
    from agents.hybrid import dqn_value_fn
    agent = DQNAgent()
    fn = dqn_value_fn(agent.online)
    assert isinstance(fn(initial_state()), float)
    hy = HybridAgent(eval_fn=fn, depth=2)
    assert hy.choose_action(initial_state()) in legal_moves(initial_state())


# --------------------------------------------------------------------------- #
# Módulo de evaluación cost-matched
# --------------------------------------------------------------------------- #

def test_hybrid_eval_cuenta_partidas_y_costo():
    from eval.hybrid_eval import evaluate_vs_depth
    hy = HybridAgent(eval_fn=heuristic_value_fn(), depth=2, use_cache=False)
    r = evaluate_vs_depth(hy, opp_depth=3, games=4, opening_plies=2, max_half=80)
    assert r["wins"] + r["losses"] + r["draws"] == 4
    assert r["nodes_per_move"] > 0               # se midió el costo en nodos
