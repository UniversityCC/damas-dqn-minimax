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

## Docker (corre en cualquier máquina, sin GPU)

Alternativa al `venv`: una imagen **CPU-only** que funciona igual en Windows/Mac/Linux
sin instalar nada más que Docker. El entrenamiento del DQN con GPU se hace aparte en
Colab/Kaggle; este contenedor cubre motor, Minimax, evaluación, tests y la demo.

```bash
# Correr la batería de tests dentro del contenedor
docker compose run --rm tests

# Levantar la demo (cuando exista demo/app.py, issue #18) en http://localhost:8501
docker compose --profile demo up demo
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
