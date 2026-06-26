"""Demo Streamlit: humano vs agente DQN cargado desde checkpoint.

Uso local:
    streamlit run demo/app.py

Uso Docker:
    docker compose --profile demo up demo

La aplicación busca checkpoints ``.pt`` en ``models/`` y ``src/models/`` para
jugar una partida de damas contra el DQN en modo greedy. El humano puede jugar
con rojas (mueve primero) o negras; el agente responde automáticamente cuando
es su turno.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Iterable

import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agents.dqn import DQNAgent  # noqa: E402
from damas.engine import Action, State, _JUMP_OVER, initial_state, is_terminal, legal_moves, result, step  # noqa: E402


PIECE_LABELS = {
    0: "",
    1: "🔴",
    2: "🔴♛",
    -1: "⚫",
    -2: "⚫♛",
}


def square_to_rc(square: int) -> tuple[int, int]:
    """Convierte una casilla jugable 0..31 a coordenadas visuales 8x8."""
    row = square // 4
    col_in_row = square % 4
    col = 1 + 2 * col_in_row if row % 2 == 0 else 2 * col_in_row
    return row, col


def rc_to_square(row: int, col: int) -> int | None:
    """Convierte coordenadas 8x8 a casilla jugable 0..31, o None si es clara."""
    if row < 0 or row > 7 or col < 0 or col > 7:
        return None
    playable = (row + col) % 2 == 1
    if not playable:
        return None
    col_in_row = (col - 1) // 2 if row % 2 == 0 else col // 2
    return row * 4 + col_in_row


def action_text(action: Action) -> str:
    """Representación breve y legible de una acción del motor."""
    parts = [str(action[0])]
    for src, dst in zip(action, action[1:]):
        parts.append("×" if (src, dst) in _JUMP_OVER else "→")
        parts.append(str(dst))
    return " ".join(parts)


def checkpoint_options() -> list[Path]:
    """Lista checkpoints disponibles en models/ y src/models/.

    El proyecto documenta ``models/`` como carpeta estándar, pero algunas
    corridas de entrenamiento guardan checkpoints en ``src/models/``. La demo
    soporta ambas ubicaciones y prioriza los archivos más recientes.
    """
    candidates: list[Path] = []
    for models_dir in (ROOT / "models", ROOT / "src" / "models"):
        if models_dir.exists():
            candidates.extend(models_dir.glob("*.pt"))
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)


@st.cache_resource(show_spinner="Cargando checkpoint del DQN...")
def load_agent(checkpoint: str, device: str = "cpu") -> DQNAgent:
    """Carga un DQNAgent desde checkpoint y lo deja en modo evaluación."""
    agent = DQNAgent(device=device)
    agent.load(checkpoint)
    agent.online.eval()
    agent.target.eval()
    return agent


def new_game(human_player: int) -> None:
    """Inicializa estado de partida en la sesión de Streamlit."""
    st.session_state.game_state = initial_state()
    st.session_state.human_player = human_player
    st.session_state.selected_square = None
    st.session_state.message = "Partida nueva. Rojas mueven primero."
    st.session_state.move_log = []


def ensure_session_defaults() -> None:
    if "game_state" not in st.session_state:
        new_game(human_player=1)
    if "move_log" not in st.session_state:
        st.session_state.move_log = []


def legal_moves_from(square: int, state: State) -> list[Action]:
    return [move for move in legal_moves(state) if move[0] == square]


def apply_move(action: Action, actor: str) -> None:
    """Aplica una acción y registra el movimiento."""
    legal = legal_moves(st.session_state.game_state)
    if action not in legal:
        st.session_state.message = (
            f"Movimiento ilegal bloqueado para {actor}: {action_text(action)}. "
            f"Legales: {', '.join(action_text(m) for m in legal)}"
        )
        return

    before_turn = st.session_state.game_state["turn"]
    before_board = list(st.session_state.game_state["board"])
    st.session_state.game_state = step(st.session_state.game_state, action)
    after_board = st.session_state.game_state["board"]
    color = "Rojas" if before_turn == 1 else "Negras"
    moved_piece = PIECE_LABELS.get(before_board[action[0]], str(before_board[action[0]]))
    captured = [mid for src, dst in zip(action, action[1:]) if (mid := _JUMP_OVER.get((src, dst))) is not None]
    capture_text = f" | capturó {captured}" if captured else ""
    st.session_state.move_log.append(
        f"{actor} ({color}) {moved_piece}: {action_text(action)}{capture_text}"
    )
    st.session_state.last_action_debug = {
        "actor": actor,
        "action": action,
        "before_nonzero": {i: p for i, p in enumerate(before_board) if p},
        "after_nonzero": {i: p for i, p in enumerate(after_board) if p},
    }
    st.session_state.selected_square = None


def maybe_agent_turn(agent: DQNAgent | None) -> None:
    """Si corresponde, ejecuta automáticamente el movimiento greedy del DQN."""
    state = st.session_state.game_state
    if agent is None or is_terminal(state):
        return
    if state["turn"] == st.session_state.human_player:
        return
    try:
        action = agent.act(state, greedy=True)
        apply_move(action, "DQN")
        st.session_state.message = f"El DQN jugó: {action_text(action)}"
    except Exception as exc:  # pragma: no cover - visible en UI
        st.session_state.message = f"Error al mover el agente: {exc}"


def handle_click(square: int) -> None:
    """Gestiona selección de pieza/destino para el jugador humano."""
    state = st.session_state.game_state
    if is_terminal(state):
        return
    if state["turn"] != st.session_state.human_player:
        st.session_state.message = "Es turno del agente."
        return

    board = state["board"]
    selected = st.session_state.selected_square
    human_piece = board[square] != 0 and (board[square] > 0) == (st.session_state.human_player > 0)

    if selected is None:
        if human_piece and legal_moves_from(square, state):
            st.session_state.selected_square = square
            st.session_state.message = f"Seleccionaste la casilla {square}. Elige destino."
        else:
            st.session_state.message = "Selecciona una pieza tuya con movimientos legales."
        return

    if selected == square:
        st.session_state.selected_square = None
        st.session_state.message = "Selección cancelada."
        return

    if human_piece and legal_moves_from(square, state):
        st.session_state.selected_square = square
        st.session_state.message = f"Nueva pieza seleccionada: {square}."
        return

    candidates = [move for move in legal_moves_from(selected, state) if move[-1] == square]
    if len(candidates) == 1:
        apply_move(candidates[0], "Humano")
        st.session_state.message = f"Jugaste: {action_text(candidates[0])}"
    elif len(candidates) > 1:
        st.session_state.pending_candidates = candidates
        st.session_state.message = "Hay varias capturas con el mismo destino; elige una en el panel lateral."
    else:
        st.session_state.message = "Destino inválido para la pieza seleccionada."


def status_text(state: State) -> str:
    if is_terminal(state):
        winner = result(state)
        if winner == 0:
            return "🏁 Partida terminada: empate."
        return f"🏁 Partida terminada: ganan {'rojas' if winner == 1 else 'negras'}."
    return f"Turno actual: {'🔴 rojas' if state['turn'] == 1 else '⚫ negras'}"


def render_board(highlights: Iterable[int]) -> None:
    """Dibuja el tablero como una cuadrícula de botones Streamlit."""
    state = st.session_state.game_state
    board = state["board"]
    selected = st.session_state.selected_square
    highlight_set = set(highlights)

    for row in range(8):
        cols = st.columns(8, gap="small")
        for col in range(8):
            square = rc_to_square(row, col)
            with cols[col]:
                if square is None:
                    st.button(" ", key=f"light-{row}-{col}", disabled=True, use_container_width=True)
                    continue

                piece = PIECE_LABELS[board[square]]
                marker = ""
                if square == selected:
                    marker = "▣"
                elif square in highlight_set:
                    marker = "●"
                label = f"{piece or marker}\n{square}" if piece else f"{marker}\n{square}"
                if st.button(label, key=f"sq-{square}", use_container_width=True):
                    handle_click(square)
                    st.rerun()


def main() -> None:
    st.set_page_config(page_title="Damas DQN vs Humano", page_icon="♟️", layout="wide")
    st.markdown(
        """
        <style>
        [data-testid="stHeader"] {
            height: 0rem;
            min-height: 0rem;
            background: transparent;
        }

        [data-testid="stAppViewContainer"] > .main .block-container {
            padding-top: 0.75rem;
            padding-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    ensure_session_defaults()

    st.title("♟️ Damas: Humano vs DQN")
    st.caption("Interfaz Streamlit para jugar contra un agente DQN cargado desde checkpoint.")

    checkpoints = checkpoint_options()
    agent: DQNAgent | None = None

    with st.sidebar:
        st.header("Configuración")
        human_side_label = st.radio("Jugar como", ["Rojas (mueves primero)", "Negras"], index=0)
        chosen_human = 1 if human_side_label.startswith("Rojas") else -1

        if checkpoints:
            selected_ckpt = st.selectbox(
                "Checkpoint",
                options=[str(p.relative_to(ROOT)) for p in checkpoints],
            )
            ckpt_path = str(ROOT / selected_ckpt)
            try:
                agent = load_agent(ckpt_path)
                st.success(f"DQN cargado: `{selected_ckpt}`")
                st.caption(f"Pasos de aprendizaje: {agent.learn_steps}")
            except Exception as exc:
                st.error(f"No se pudo cargar el checkpoint: {exc}")
                agent = None
        else:
            st.warning("No hay checkpoints `.pt` en `models/` ni en `src/models/`.")
            st.info("Copia, por ejemplo, `models/checkpoint_final.pt` o `src/models/checkpoint_final.pt` para jugar contra el DQN entrenado.")

        if st.button("Nueva partida", use_container_width=True):
            new_game(chosen_human)
            st.rerun()

        if chosen_human != st.session_state.human_player:
            st.info("Pulsa **Nueva partida** para aplicar el cambio de color.")

        st.divider()
        st.subheader("Movimientos legales")
        state = st.session_state.game_state
        if state["turn"] == st.session_state.human_player and not is_terminal(state):
            selected = st.session_state.selected_square
            moves = legal_moves_from(selected, state) if selected is not None else legal_moves(state)
            for move in moves[:30]:
                st.code(action_text(move), language=None)
            if len(moves) > 30:
                st.caption(f"... y {len(moves) - 30} más")
        else:
            st.caption("Disponibles durante tu turno.")

        pending = st.session_state.get("pending_candidates", [])
        if pending:
            st.subheader("Resolver captura")
            choice = st.selectbox("Secuencia", list(range(len(pending))), format_func=lambda i: action_text(pending[i]))
            if st.button("Aplicar secuencia", use_container_width=True):
                apply_move(pending[choice], "Humano")
                st.session_state.pending_candidates = []
                st.rerun()

    maybe_agent_turn(agent)

    left, right = st.columns([2, 1])
    with left:
        state = st.session_state.game_state
        selected = st.session_state.selected_square
        highlights = [move[-1] for move in legal_moves_from(selected, state)] if selected is not None else []
        st.subheader(status_text(state))
        st.info(st.session_state.message)
        render_board(highlights)

    with right:
        st.subheader("Resumen")
        state = st.session_state.game_state
        board = state["board"]
        st.metric("Piezas rojas", sum(1 for p in board if p > 0))
        st.metric("Piezas negras", sum(1 for p in board if p < 0))
        st.metric("Sin capturas", state["no_capture_count"])
        st.divider()
        st.subheader("Historial")
        if st.session_state.move_log:
            for item in reversed(st.session_state.move_log[-20:]):
                st.write(item)
        else:
            st.caption("Aún no hay movimientos.")

        last_action = st.session_state.get("last_action_debug")
        if last_action:
            st.divider()
            st.subheader("Última acción real")
            st.write(f"Actor: **{last_action['actor']}**")
            st.code(action_text(tuple(last_action["action"])), language=None)

        with st.expander("Estado crudo para depuración"):
            st.json(copy.deepcopy(state))
            if last_action:
                st.json(copy.deepcopy(last_action))


if __name__ == "__main__":
    main()