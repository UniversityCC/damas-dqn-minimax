# damas-dqn-minimax

Agente de **Damas 8×8** que combina un motor propio de reglas, un agente **Minimax con poda alfa-beta** y un agente **DQN entrenable por auto-juego**. Proyecto Final de Inteligencia Artificial — UNI FC CC.

[![CI](https://github.com/UniversityCC/damas-dqn-minimax/actions/workflows/ci.yml/badge.svg)](https://github.com/UniversityCC/damas-dqn-minimax/actions)

## Release v1.0

- **Release:** [`v1.0`](https://github.com/UniversityCC/damas-dqn-minimax/releases/tag/v1.0)
- **Paper/informe:** [`Informe/main.pdf`](Informe/main.pdf)
- **Demo:** [`demo/app.py`](demo/app.py) — ejecutar con `streamlit run demo/app.py` o con Docker mediante `docker compose --profile demo up demo`.

## Contenido del proyecto

- Motor de Damas sobre 32 casillas jugables: movimientos legales, capturas obligatorias, multicapturas, coronación y detección de finales.
- Agentes: `MinimaxAgent` y `DQNAgent`.
- Entrenamiento DQN por auto-juego con checkpoints `.pt`.
- Evaluaciones, torneos, tuning y ablaciones reproducibles.
- Demo web en Streamlit para jugar Humano vs DQN.
- Informe/paper en [`Informe/main.pdf`](Informe/main.pdf).

## Requisitos

- Python **3.11** recomendado.
- `pip` y `venv`.
- Opcional: Docker Desktop / Docker Compose para ejecutar sin instalar dependencias locales.
- Opcional: GPU CUDA para entrenamientos largos; por defecto todo puede correr en CPU.

## Instalación local

### 1. Clonar el repositorio

```bash
git clone https://github.com/UniversityCC/damas-dqn-minimax.git
cd damas-dqn-minimax
```

### 2. Crear y activar entorno virtual

En Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

En Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

En Windows CMD:

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

### 3. Instalar dependencias

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> Nota: `requirements.txt` instala PyTorch, Streamlit, Gymnasium, PyGame, JupyterLab, PyTest y librerías de análisis/visualización.

## Ejecución rápida

### Ejecutar pruebas

```bash
pytest
```

También se puede ejecutar una prueba específica:

```bash
pytest tests/test_engine.py
```

### Ejecutar una partida Minimax vs Minimax por consola

```bash
python play.py
```

### Levantar la demo web Humano vs DQN

```bash
streamlit run demo/app.py
```

Luego abrir:

```text
http://localhost:8501
```

La demo busca checkpoints `.pt` en:

- `models/`
- `src/models/`

El repositorio versiona `models/.gitkeep`, pero ignora los checkpoints `.pt` para evitar subir archivos grandes. Si no hay un checkpoint disponible, la interfaz se abre pero no podrá cargar un agente DQN entrenado.

## Entrenamiento DQN por auto-juego

Entrenamiento corto de prueba en CPU:

```bash
python src/selfplay.py --episodes 10 --log-every 5 --checkpoint-every 10 --device cpu
```

Entrenamiento más largo, guardando curva de entrenamiento:

```bash
python src/selfplay.py \
  --episodes 1000 \
  --device cpu \
  --checkpoint-dir models \
  --checkpoint-every 200 \
  --log-csv results/training_log.csv
```

Continuar desde un checkpoint:

```bash
python src/selfplay.py --checkpoint models/checkpoint_final.pt --episodes 500 --device cpu
```

Con GPU, si el entorno tiene CUDA disponible:

```bash
python src/selfplay.py --episodes 5000 --device cuda --batch-size 128
```

## Evaluación y experimentos

### Torneo Minimax vs Minimax

```bash
python src/tournament/run.py --games 20 --depth-a 3 --depth-b 4 --out data/tournament_results.csv
```

### Evaluación DQN vs Minimax por profundidad

```bash
python src/eval/run.py --checkpoint models/checkpoint_final.pt --games 20 --depths 3 4 5 6 --out results/eval_depth.csv
```

### Evaluación preliminar DQN vs Minimax

```bash
python src/eval/preliminary.py --checkpoint models/checkpoint_final.pt --games 20 --depth 3 --out results/preliminary.csv
```

### Torneo final y métricas ELO

```bash
python src/eval/final_tournament.py --checkpoint models/checkpoint_final.pt --games 20 --out results/final
```

### Tuning de hiperparámetros

```bash
python src/tuning/run.py --steps 3000 --eval-games 20 --out results/tuning.csv
```

### Ablaciones

```bash
python src/ablations/run.py --steps 3000 --eval-games 20 --out results/ablations.csv
```

### Notebook de experimentos

```bash
jupyter lab notebooks/experimentos.ipynb
```

## Docker

El proyecto incluye una imagen **CPU-only** para ejecutar pruebas y demo en Windows, macOS o Linux sin preparar un entorno Python local.

### Ejecutar tests con Docker

```bash
docker compose run --rm tests
```

### Levantar demo con Docker

```bash
docker compose --profile demo up demo
```

Abrir:

```text
http://localhost:8501
```

## Artefactos incluidos

- Paper/informe: [`Informe/main.pdf`](Informe/main.pdf)
- Fuente LaTeX del informe: [`Informe/main.tex`](Informe/main.tex)
- Demo Streamlit: [`demo/app.py`](demo/app.py)
- Resultados CSV y figuras finales: [`results/`](results/)
- Figuras del informe: [`Informe/figures/`](Informe/figures/)
- Carpeta local para checkpoints: [`models/`](models/) (`models/.gitkeep` versionado; archivos `.pt` ignorados por Git)

> Los checkpoints `.pt` se generan o copian localmente en `models/` para entrenamiento, demo y evaluación, pero no se versionan porque pueden ser archivos grandes.

## Estructura del repositorio

```text
.
├── play.py                    # partida Minimax vs Minimax por consola
├── demo/app.py                # demo Streamlit Humano vs DQN
├── Dockerfile                 # imagen CPU-only
├── docker-compose.yml         # servicios tests y demo
├── Informe/                   # paper, fuente LaTeX y figuras
├── notebooks/                 # experimentos reproducibles
├── results/                   # métricas, CSV y figuras generadas
├── src/
│   ├── damas/                 # motor del juego
│   ├── agents/                # DQN, Minimax, heurística, replay buffer
│   ├── model/                 # espacio de acciones y red Q
│   ├── env/                   # wrapper Gymnasium
│   ├── tournament/            # lógica de torneos
│   ├── eval/                  # evaluaciones y torneo final
│   ├── tuning/                # búsqueda de hiperparámetros
│   ├── ablations/             # estudios de ablación
│   └── selfplay.py            # entrenamiento DQN por auto-juego
└── tests/                     # pruebas automatizadas
```

## Flujo recomendado para reproducir

1. Instalar dependencias con `pip install -r requirements.txt`.
2. Ejecutar `pytest` para verificar el entorno.
3. Probar `python play.py` para validar el motor y Minimax.
4. Ejecutar `streamlit run demo/app.py` para abrir la demo.
5. Opcionalmente entrenar con `python src/selfplay.py ...` y evaluar con los scripts de `src/eval/`.

## Solución de problemas comunes

- **`ModuleNotFoundError` al ejecutar scripts:** ejecutar los comandos desde la raíz del repositorio y mantener activo el entorno virtual.
- **No carga el DQN en la demo:** generar o copiar un checkpoint `.pt` en `models/` o `src/models/`.
- **PyTorch pesado en instalación local:** usar Docker si solo se quiere correr tests/demo en CPU.
- **PowerShell bloquea activación del venv:** ejecutar `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` y volver a activar `.\.venv\Scripts\Activate.ps1`.

## Licencia y créditos

Proyecto académico desarrollado para el curso de Inteligencia Artificial — UNI FC CC.
