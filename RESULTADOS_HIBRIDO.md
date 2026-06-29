# Híbrido DQN + búsqueda: cruzar el muro d=5/d=6 y la palanca real

Este documento reúne la investigación sobre por qué el DQN no vence a Minimax de
profundidad alta y qué lo resuelve. Tres resultados, todos reproducibles con los scripts
del repo.

## 1. El muro estructural

El DQN entrenado (Fase 2: Double DQN, reward shaping, soft update, currículo vs d=3) vence
a Minimax de baja profundidad pero choca contra un muro:

| Oponente | DQN puro (greedy `argmax Q`) |
|----------|------------------------------|
| Minimax(d=3) | ~80 % victorias |
| Minimax(d=4) | ~63 % victorias |
| Minimax(d=5) | **0 %** |
| Minimax(d=6) | **0 %** |

**Causa:** el DQN decide en **una sola pasada, sin búsqueda hacia adelante**, mientras que
Minimax(d=5/6) hace *lookahead* de 5–6 plies. Es un límite **estructural**: ningún ajuste
de entrenamiento le da profundidad de cálculo.

## 2. El híbrido cruza el muro — pero solo a profundidad comparable

`HybridAgent` (`src/agents/hybrid.py`) envuelve el evaluador del DQN
(`V(s)=max_a Q(s,a)`) dentro de una búsqueda **negamax + poda alfa-beta** de profundidad k.
La red aporta la intuición posicional en las **hojas**; la búsqueda aporta el *lookahead*.

Evaluación **cost-matched** (`src/eval/hybrid_eval.py`): aperturas aleatorias, semillas
pareadas, y se reporta win-rate **junto con el costo** (nodos/jugada y ms/jugada), para una
comparación justa "a cómputo comparable" y no "quien busca más hondo gana". Como **control**
se corre el mismo buscador con la heurística a mano en vez del DQN, a igual k.

Win-rate del híbrido vs Minimax (np = no-pierde):

| k | DQN vs d5 | DQN vs d6 | Heur vs d5 | Heur vs d6 | ms/jugada DQN | ms/jugada Heur |
|---|-----------|-----------|------------|------------|---------------|----------------|
| 2 | 0 %       | 0 % (np30)| –          | –          | 3             | –              |
| 3 | 0 %       | 10 %      | 10 %       | 10 %       | 9             | 3              |
| 4 | 10 %(np20)| 10 %      | 0 %        | 10 %       | 22            | 10             |
| 5 | 25 %(np62)| 38 %(np62)| 50 %       | 20 %       | 53–76         | 29–38          |
| 6 | 33 %(np50)| 17 %(np83)| 30 %       | 30 %       | 66–186        | 49–87          |

**Lecturas:**
1. **El muro se cruza solo a profundidad de búsqueda comparable (k≥5).** A k≤4 el híbrido va
   0–10 %; a k=5 salta a 25–38 % de victorias y ~60 % de no-perder vs d=5/d=6.
2. **El evaluador DQN NO supera a la heurística a mano** a igual k (van parejos, con ruido) y
   **cuesta 2–4× más** (inferencia de red en hojas). No es *cost-effective*.
3. **El factor dominante es la profundidad de búsqueda, no el evaluador aprendido.**

## 3. ¿Más entrenamiento fortalece al evaluador? (calidad vs cantidad)

Experimento que **aísla la calidad del evaluador de la profundidad**: se reentrena con la
receta de Fase 2 guardando un snapshot cada 500 episodios, y cada snapshot se enchufa como
evaluador del híbrido a **profundidad FIJA k=3**, enfrentado a Minimax(d=4) y Minimax(d=5)
(14 partidas, aperturas aleatorias). Script: `src/eval/evaluator_quality.py`.

![Calidad del evaluador vs entrenamiento](results/evaluator_quality.png)

Datos: `results/evaluator_quality.csv`.

| learn_steps | vs d=4 (W% / NP%) | vs d=5 (W% / NP%) |
|-------------|-------------------|-------------------|
| 17 (sin entrenar) | 0 / 0   | 0 / 0 |
| 7 906   | 14 / 29 | 0 / 0 |
| 15 954  | 14 / 57 | 0 / 0 |
| 24 153  | 21 / 64 | 0 / 7 |
| 32 570  | 14 / 79 | 7 / 7 |
| **40 768 (ep2500)** | **36 / 86** | 0 / 7 |
| 48 940  | 21 / 64 | 0 / 7 |
| 57 428  | 14 / 71 | 0 / 0 |
| 65 439 (final) | 29 / 64 | 0 / 0 |

**Lecturas:**
1. **El entrenamiento SÍ fortalece el evaluador** contra oponentes alcanzables: sin entrenar
   pierde 14/14 vs d=4; entrenado llega a **86 % de no-perder**. La fuerza viene del
   entrenamiento, medido de forma directa.
2. **Pero satura y luego se estanca/retrocede:** el pico vs d=4 está en ~40k pasos (**ep2500**)
   y después baja a 64–71 %. Más entrenamiento del mismo tipo no ayuda — consistente con el
   olvido/colapso de distribución del auto-juego.
3. **Contra el muro d=5 la curva es plana en CERO**, sin importar la cantidad de entrenamiento.
   A profundidad k=3, ni el mejor checkpoint cruza d=5. **La palanca para d=5/d=6 es la
   profundidad de búsqueda, no el entrenamiento.**
4. **El checkpoint final no es el mejor:** ep2500 (86 % NP vs d=4) supera al final (64 %).
   Conviene seleccionar el checkpoint por evaluación, no usar siempre el último.

## Conclusión

Para vencer a Minimax a **toda** profundidad, la palanca demostrada es la **búsqueda en
inferencia** (híbrido a k≥5), no más entrenamiento. El entrenamiento fortalece el evaluador
solo hasta un techo (cercano al de una heurística manual) y no mueve el muro d=5/d=6. La
formulación honesta del resultado:

> *El RL puro pierde frente a búsqueda profunda; un evaluador aprendido + búsqueda ligera la
> alcanza solo a profundidad comparable, y ese evaluador no supera a una heurística a mano
> (y cuesta más). La profundidad de búsqueda —no la cantidad de entrenamiento— es el factor
> dominante.*

## Reproducir

```bash
# 1) Entrenar guardando snapshots cada 500 episodios (receta Fase 2)
OMP_NUM_THREADS=4 .venv/bin/python src/selfplay.py --episodes 4000 --device cpu \
  --gamma 0.90 --lr 0.001 --buffer-capacity 200000 --eps-decay-steps 60000 \
  --capture-reward 0.05 --king-reward 0.2 --soft-tau 0.005 \
  --opponent-minimax-frac 0.5 --opponent-minimax-depth 3 \
  --checkpoint-dir models/snapshots --checkpoint-every 500

# 2) Curva calidad-del-evaluador (búsqueda fija k=3 vs d=4 y d=5)
OMP_NUM_THREADS=4 .venv/bin/python src/eval/evaluator_quality.py \
  --checkpoint-dir models/snapshots --depth 3 --opp-depths 4 5 --games 14 \
  --out results/evaluator_quality.csv

# 3) Híbrido cost-matched vs Minimax profundo
OMP_NUM_THREADS=4 .venv/bin/python src/eval/hybrid_eval.py \
  --eval dqn --checkpoint models/checkpoint_swa_fase2.pt --depth 5 --opp-depths 5 6 --games 10
```
