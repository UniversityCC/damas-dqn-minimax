"""
Enfrenta dos MinimaxAgent con distintas profundidades y muestra el resultado.
Uso: python play.py
"""
import sys
from pathlib import Path

# Añade src/ al path para que 'damas' y 'agents' sean importables
sys.path.insert(0, str(Path(__file__).parent / "src"))

from damas import initial_state, step, is_terminal, result
from agents.minimax import MinimaxAgent


def render_board(board: list) -> None:
    symbols = {0: ".", 1: "r", -1: "b", 2: "R", -2: "B"}
    grid = [["." for _ in range(8)] for _ in range(8)]
    for idx in range(32):
        r = idx // 4
        c = (idx % 4) * 2 + (1 if r % 2 == 1 else 0)
        grid[r][c] = symbols[board[idx]]
    print("  0 1 2 3 4 5 6 7")
    for r, row in enumerate(grid):
        print(f"{r} {' '.join(row)}")
    print()


def play_game(agent1: MinimaxAgent, agent2: MinimaxAgent, max_half_moves: int = 300) -> tuple:
    state = initial_state()
    for half_move in range(max_half_moves):
        if is_terminal(state):
            break
        agent = agent1 if state["turn"] == 1 else agent2
        action = agent.choose_action(state)
        if action is None:
            break
        state = step(state, action)
        if (half_move + 1) % 20 == 0:
            r_count = sum(1 for p in state["board"] if p in (1, 2))
            b_count = sum(1 for p in state["board"] if p in (-1, -2))
            print(f"  [turno {half_move + 1}]  rojo={r_count}  negro={b_count}")
    return state, half_move + 1


def main() -> None:
    depth1, depth2 = 3, 4
    agent1 = MinimaxAgent(depth=depth1, player=1)
    agent2 = MinimaxAgent(depth=depth2, player=-1)

    print(f"=== Minimax(depth={depth1}, rojo) vs Minimax(depth={depth2}, negro) ===\n")
    print("Tablero inicial:")
    render_board(initial_state()["board"])

    final_state, total_moves = play_game(agent1, agent2)

    print("\nTablero final:")
    render_board(final_state["board"])

    r = result(final_state)
    print(f"Partida terminada en {total_moves} half-moves.")
    if r == 1:
        print("Resultado: ROJO gana")
    elif r == -1:
        print("Resultado: NEGRO gana")
    elif r == 0:
        print("Resultado: EMPATE")
    else:
        print("Resultado: límite de half-moves alcanzado (sin ganador decidido)")


if __name__ == "__main__":
    main()
