# Proyecto #2 - Análisis y Diseño de Algoritmos

## Optimal Game Strategy in a Row of Coins

Este proyecto compara dos enfoques para el problema **Optimal Game Strategy in a Row of Coins**:

- **Programación Dinámica (DP)**: solución exacta, optima.
- **Greedy**: aproximación rapida, elige siempre la moneda mayor entre extremos.

La comparación se realiza en terminos de:

- Tiempo de ejecución
- Calidad de la solución greedy respecto al optimo
- Casos donde greedy coincide con DP
- Casos donde greedy falla

## Requisitos

- Python 3.10+
- Paquetes:
  - `numpy`
  - `matplotlib`

Instalación rápida:

```bash
pip install numpy matplotlib
```

## Cómo ejecutar

Desde la carpeta del proyecto:

```bash
python main.py
```

El script ejecuta, en orden:

1. Casos de prueba manuales
2. Demo de tabla DP para un contraejemplo
3. Analisis empírico para multiples tamaños
4. Exportación de resultados a CSV
5. Generación de gráficas

## Complejidad 

- DP:
  - Tiempo: `O(n^2)`
  - Espacio: `O(n^2)`
- Greedy (simulación rápida):
  - Tiempo: `O(n)`
  - Espacio: `O(1)` o `O(n)` si se guarda historial de picks.

## Ejemplo fragmento del codigo

El archivo `main.py` ya incluye ejecución automatica. Si deseas reutilizar funciones:

```python
from main import dp_coin_game, greedy_coin_game, greedy_score_vs_optimal_opponent

coins = [8, 15, 3, 7]
dp_score, _ = dp_coin_game(coins)
greedy_fast, _ = greedy_coin_game(coins)
greedy_vs_opt = greedy_score_vs_optimal_opponent(coins)

print(dp_score, greedy_fast, greedy_vs_opt)
```
