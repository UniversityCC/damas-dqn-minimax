# Interfaz pública del motor de damas

Esta sección define la interfaz estable que deben utilizar todos los módulos del proyecto. Los agentes de aprendizaje por refuerzo, los jugadores basados en Minimax, los torneos y la interfaz gráfica no deben manipular el tablero directamente; en su lugar, deben interactuar con el motor exclusivamente mediante las funciones públicas descritas aquí.

## Representación del estado

El estado de una partida se representa como un diccionario de Python con tres campos:

```python
state = {
    "board":            list[int],  # 32 enteros
    "turn":             int,        # 1 = ROJO, -1 = NEGRO
    "no_capture_count": int,        # half-moves sin captura
}
```

El campo `board` contiene únicamente las 32 casillas jugables del tablero. Los valores permitidos son:

| Valor | Significado |
|-------|-------------|
| `0` | Casilla vacía |
| `1` | Pieza normal roja |
| `-1` | Pieza normal negra |
| `2` | Dama roja |
| `-2` | Dama negra |

Las casillas se numeran de izquierda a derecha y de arriba hacia abajo, considerando solo las diagonales oscuras:

```
Fila 0:  00  01  02  03
Fila 1:  04  05  06  07
Fila 2:  08  09  10  11
Fila 3:  12  13  14  15
Fila 4:  16  17  18  19
Fila 5:  20  21  22  23
Fila 6:  24  25  26  27
Fila 7:  28  29  30  31
```

Las piezas rojas comienzan en las casillas `0–11` y las negras en las casillas `20–31`. El turno inicial corresponde al jugador rojo.

## Acciones

Una acción se representa como una tupla de enteros que indica la secuencia de casillas recorridas por una pieza:

```python
(5, 9)           # movimiento simple
(5, 14, 9)       # captura simple
(5, 14, 9, 18)   # captura múltiple
```

Esta representación permite tratar de forma uniforme los movimientos simples, las capturas y las cadenas de capturas obligatorias.

## Funciones públicas

Los módulos externos deben importar la interfaz del motor de la siguiente manera:

```python
from damas import initial_state, legal_moves, step, is_terminal, result, encode
```

**`initial_state()`** devuelve el estado inicial estándar de una partida nueva.

**`legal_moves(state)`** devuelve la lista de acciones legales para el jugador en turno. Si existe al menos una captura disponible, solo se devuelven capturas, respetando la regla de captura obligatoria. Si no hay movimientos legales, devuelve una lista vacía.

**`step(state, action)`** aplica una acción legal y devuelve un nuevo estado. La función no modifica el estado recibido. Si la acción no es legal, debe lanzar `ValueError`.

**`is_terminal(state)`** indica si la partida ha terminado. Una posición es terminal cuando el jugador en turno no tiene movimientos legales o cuando `no_capture_count >= 80`, lo que corresponde a 40 movimientos completos sin captura.

**`result(state)`** devuelve el resultado final: `1` si gana rojo, `-1` si gana negro, `0` si hay empate y `None` si la partida sigue en curso.

**`encode(state)`** convierte el estado en un vector de 160 valores de tipo `float`. El vector está compuesto por cinco canales de 32 casillas: piezas rojas normales, damas rojas, piezas negras normales, damas negras y turno actual.