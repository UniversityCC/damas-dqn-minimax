# Imagen CPU-only: corre en cualquier máquina (Windows/Mac/Linux) SIN GPU.
# El entrenamiento del DQN que necesita GPU se hace aparte en Colab/Kaggle;
# este contenedor sirve para motor, Minimax, evaluación, tests y la demo.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Librerías de sistema mínimas: libgomp1 (runtime de torch),
# libgl1 y libglib2.0-0 (necesarias para que pygame importe sin fallar).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# torch en su build de CPU (más liviano y sin CUDA). Al fijar la misma versión
# 2.3.0, el paso siguiente con requirements.txt lo da por satisfecho.
RUN pip install --no-cache-dir torch==2.3.0 \
        --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Por defecto corre la batería de tests. La demo se levanta con el perfil "demo".
CMD ["pytest"]
