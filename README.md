# damas-dqn-minimax
Agente de Damas: DQN por auto-juego vs. Minimax con poda alfa-beta. Proyecto Final de Inteligencia Artificial — UNI FC CC.

Agentes de aprendizaje por refuerzo para damas (8×8).

[![CI](https://github.com/akobo05/damas-dqn-minimax/actions/workflows/ci.yml/badge.svg)](https://github.com/akobo05/damas-dqn-minimax/actions)

## Instalación

```bash
git clone https://github.com/akobo05/damas-dqn-minimax.git
cd damas-dqn-minimax
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Verificar

```bash
pytest
```

## Estructura

```
src/
  damas/        # motor del juego
  env/          # wrapper Gymnasium
  agents/       # minimax.py, dqn.py
  model/        # red Q
  tournament/   # torneos + métricas
data/           # partidas de auto-juego generadas
notebooks/      # experimentos y visualizaciones
models/         # checkpoints entrenados (.pt)
demo/           # Pygame/Streamlit
tests/
```
