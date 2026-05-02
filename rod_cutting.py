
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import Optional


# ─────────────────────────────────────────────
#  1. PROGRAMACIÓN DINÁMICA — BOTTOM-UP
# ─────────────────────────────────────────────

def dp_bottom_up(prices: list[int], n: int) -> tuple[int, list[int]]:
    """
    Resuelve Rod Cutting con DP bottom-up (tabulación).

    Parámetros
    ----------
    prices : list[int]
        prices[i] = precio de una varilla de longitud i  (índice 0 no se usa)
    n      : int
        Longitud total de la varilla

    Retorna
    -------
    (valor_optimo, lista_de_cortes)

    Complejidad temporal : O(n^2)
    Complejidad espacial : O(n)
    """
    r = [0] * (n + 1)   # r[j] = máximo ingreso para varilla de longitud j
    s = [0] * (n + 1)   # s[j] = primer corte óptimo para longitud j

    for j in range(1, n + 1):
        best = float('-inf')
        for i in range(1, j + 1):
            if i <= len(prices) - 1 and prices[i] + r[j - i] > best:
                best = prices[i] + r[j - i]
                s[j] = i
        r[j] = best

    # Reconstruir cortes desde la tabla s
    cuts = []
    remaining = n
    while remaining > 0:
        cuts.append(s[remaining])
        remaining -= s[remaining]

    return r[n], cuts


# ─────────────────────────────────────────────
#  2. PROGRAMACIÓN DINÁMICA — TOP-DOWN (MEMOIZACIÓN)
# ─────────────────────────────────────────────

def dp_top_down(prices: list[int], n: int,
                memo: Optional[dict] = None) -> tuple[int, list[int]]:
    """
    Resuelve Rod Cutting con DP top-down + memoización.

    Complejidad temporal : O(n^2)
    Complejidad espacial : O(n)  (memo + pila de recursión)
    """
    if memo is None:
        memo = {}

    def _solve(length: int) -> int:
        if length == 0:
            return 0
        if length in memo:
            return memo[length]
        best = float('-inf')
        for i in range(1, min(length, len(prices) - 1) + 1):
            val = prices[i] + _solve(length - i)
            if val > best:
                best = val
        memo[length] = best
        return best

    optimal_value = _solve(n)

    # Reconstruir cortes (greedy sobre memo ya calculado)
    cuts = []
    remaining = n
    while remaining > 0:
        best_cut = 1
        best_val = float('-inf')
        for i in range(1, min(remaining, len(prices) - 1) + 1):
            val = prices[i] + memo.get(remaining - i, 0)
            if val > best_val:
                best_val = val
                best_cut = i
        cuts.append(best_cut)
        remaining -= best_cut

    return optimal_value, cuts


# ─────────────────────────────────────────────
#  3. ALGORITMO GREEDY
# ─────────────────────────────────────────────

def greedy_rod_cutting(prices: list[int], n: int) -> tuple[int, list[int]]:
    """
    Resuelve Rod Cutting con heurística greedy:
    en cada paso elige la longitud con mayor ratio precio/longitud.

    NO garantiza solución óptima global.

    Complejidad temporal : O(n log n)  — dominada por el ordenamiento
    Complejidad espacial : O(n)
    """
    # Calcular densidad de valor por unidad
    densities = []
    for length in range(1, len(prices)):
        if prices[length] > 0:
            densities.append((length, prices[length], prices[length] / length))

    # Ordenar de mayor a menor ratio
    densities.sort(key=lambda x: x[2], reverse=True)

    remaining = n
    total_value = 0
    cuts = []

    for length, price, _ in densities:
        while length <= remaining:
            cuts.append(length)
            total_value += price
            remaining -= length

        if remaining == 0:
            break

    return total_value, cuts


# ─────────────────────────────────────────────
#  UTILIDADES
# ─────────────────────────────────────────────

def generate_prices(n: int, seed: int = 42) -> list[int]:
    """Genera tabla de precios aleatoria de tamaño n+1."""
    rng = random.Random(seed)
    prices = [0]
    for i in range(1, n + 1):
        prices.append(rng.randint(1, i * 3))
    return prices


def measure_time(func, prices: list[int], n: int, reps: int = 5) -> float:
    """Mide tiempo promedio de ejecución en segundos."""
    times = []
    for _ in range(reps):
        p_copy = prices[:]
        t0 = time.perf_counter()
        func(p_copy, n)
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times)


# ─────────────────────────────────────────────
#  DEMOSTRACIÓN RÁPIDA
# ─────────────────────────────────────────────

def demo():
    # Tabla clásica del CLRS (longitudes 1-10)
    prices = [0, 1, 5, 8, 9, 10, 17, 17, 20, 24, 30]
    n = 10

    print("=" * 55)
    print("  ROD CUTTING PROBLEM — Demo con n=10 (precios CLRS)")
    print("=" * 55)
    print(f"  Precios: {prices[1:]}")
    print(f"  n = {n}\n")

    val_dp, cuts_dp = dp_bottom_up(prices, n)
    print(f"  [DP Bottom-Up]   valor = {val_dp}   cortes = {cuts_dp}")

    val_td, cuts_td = dp_top_down(prices, n)
    print(f"  [DP Top-Down]    valor = {val_td}   cortes = {cuts_td}")

    val_gr, cuts_gr = greedy_rod_cutting(prices, n)
    quality = (val_gr / val_dp * 100) if val_dp > 0 else 100
    print(f"  [Greedy]         valor = {val_gr}   cortes = {cuts_gr}")
    print(f"\n  Calidad greedy: {quality:.1f}% del óptimo")
    print("=" * 55)


# ─────────────────────────────────────────────
#  ANÁLISIS EMPÍRICO
# ─────────────────────────────────────────────

def empirical_analysis():
    """
    Mide tiempos de ejecución para distintos tamaños de entrada
    y genera diagrama de dispersión + regresión polinomial.
    """
    sizes = list(range(5, 201, 5))   # 5, 10, 15, ..., 200
    dp_times   = []
    gr_times   = []
    qualities  = []

    print("\nEjecutando análisis empírico...")
    print(f"{'n':>6}  {'DP (ms)':>10}  {'Greedy (ms)':>12}  {'Calidad (%)':>12}")
    print("-" * 46)

    for n in sizes:
        prices = generate_prices(n)

        t_dp = measure_time(dp_bottom_up, prices, n) * 1000
        t_gr = measure_time(greedy_rod_cutting, prices, n) * 1000

        val_dp, _ = dp_bottom_up(prices, n)
        val_gr, _ = greedy_rod_cutting(prices, n)
        q = (val_gr / val_dp * 100) if val_dp > 0 else 100.0

        dp_times.append(t_dp)
        gr_times.append(t_gr)
        qualities.append(q)

        if n % 20 == 0:
            print(f"{n:>6}  {t_dp:>10.4f}  {t_gr:>12.4f}  {q:>11.1f}%")

    return sizes, dp_times, gr_times, qualities


# ─────────────────────────────────────────────
#  GRÁFICAS
# ─────────────────────────────────────────────

def plot_results(sizes, dp_times, gr_times, qualities):
    sizes_arr = np.array(sizes)

    # Regresiones polinomiales
    deg_dp = 2   # O(n^2) esperado
    deg_gr = 1   # O(n log n) ≈ lineal en práctica para estos tamaños

    coef_dp = np.polyfit(sizes_arr, dp_times, deg_dp)
    coef_gr = np.polyfit(sizes_arr, gr_times, deg_gr)

    x_smooth = np.linspace(sizes_arr[0], sizes_arr[-1], 400)
    fit_dp   = np.polyval(coef_dp, x_smooth)
    fit_gr   = np.polyval(coef_gr, x_smooth)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Rod Cutting Problem — Análisis Empírico", fontsize=14, fontweight='bold')

    # ── Panel izquierdo: tiempos de ejecución ──
    ax1 = axes[0]
    ax1.scatter(sizes, dp_times, color='#185FA5', alpha=0.7, s=30, zorder=3, label='DP (muestras)')
    ax1.scatter(sizes, gr_times, color='#0F6E56', alpha=0.7, s=30, zorder=3, label='Greedy (muestras)')
    ax1.plot(x_smooth, fit_dp, color='#185FA5', linewidth=2,
             label=f'DP regresión (grado {deg_dp})')
    ax1.plot(x_smooth, fit_gr, color='#0F6E56', linewidth=2, linestyle='--',
             label=f'Greedy regresión (grado {deg_gr})')

    ax1.set_xlabel("Longitud de la varilla (n)", fontsize=11)
    ax1.set_ylabel("Tiempo de ejecución (ms)", fontsize=11)
    ax1.set_title("Tiempo de ejecución vs tamaño de entrada", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))

    # Ecuaciones de regresión en el gráfico
    poly_dp_str = _poly_str(coef_dp, 'n')
    poly_gr_str = _poly_str(coef_gr, 'n')
    ax1.text(0.03, 0.97, f"DP:     {poly_dp_str}", transform=ax1.transAxes,
             fontsize=8, color='#185FA5', va='top', family='monospace')
    ax1.text(0.03, 0.91, f"Greedy: {poly_gr_str}", transform=ax1.transAxes,
             fontsize=8, color='#0F6E56', va='top', family='monospace')

    # ── Panel derecho: calidad de la solución greedy ──
    ax2 = axes[1]
    ax2.scatter(sizes, qualities, color='#BA7517', alpha=0.7, s=30, zorder=3)
    ax2.axhline(100, color='#A32D2D', linewidth=1.5, linestyle='--', label='Óptimo (100%)')
    ax2.fill_between(sizes, qualities, 100, alpha=0.12, color='#A32D2D')

    coef_q = np.polyfit(sizes_arr, qualities, 1)
    fit_q  = np.polyval(coef_q, x_smooth)
    ax2.plot(x_smooth, fit_q, color='#BA7517', linewidth=2,
             label='Tendencia calidad')

    ax2.set_xlabel("Longitud de la varilla (n)", fontsize=11)
    ax2.set_ylabel("Calidad (% del óptimo)", fontsize=11)
    ax2.set_title("Calidad de la solución greedy", fontsize=11)
    ax2.set_ylim(50, 105)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = "rod_cutting_analysis.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nGráfica guardada en: {out_path}")
    plt.show()


def _poly_str(coef, var='x') -> str:
    """Formatea coeficientes de np.polyfit como string legible."""
    degree = len(coef) - 1
    terms = []
    for i, c in enumerate(coef):
        exp = degree - i
        if abs(c) < 1e-9:
            continue
        sign = '+' if c >= 0 else '-'
        val  = abs(c)
        if exp == 0:
            terms.append(f"{sign}{val:.4f}")
        elif exp == 1:
            terms.append(f"{sign}{val:.6f}{var}")
        else:
            terms.append(f"{sign}{val:.8f}{var}^{exp}")
    s = ' '.join(terms).lstrip('+').strip()
    return s if s else '0'


# ─────────────────────────────────────────────
#  ENTRADAS DE PRUEBA (para el reporte)
# ─────────────────────────────────────────────

TEST_CASES = [
    # (n, prices, descripcion)
    (4,  [0, 1, 5, 8, 9],                         "CLRS pequeño n=4"),
    (8,  [0, 1, 5, 8, 9, 10, 17, 17, 20],         "CLRS estándar n=8"),
    (10, [0, 1, 5, 8, 9, 10, 17, 17, 20, 24, 30], "CLRS completo n=10"),
    (15, generate_prices(15, seed=1),              "Aleatorio n=15"),
    (30, generate_prices(30, seed=2),              "Aleatorio n=30"),
    (50, generate_prices(50, seed=3),              "Aleatorio n=50"),
    (100, generate_prices(100, seed=4),            "Aleatorio n=100"),
    (150, generate_prices(150, seed=5),            "Aleatorio n=150"),
    (200, generate_prices(200, seed=6),            "Aleatorio n=200"),
]


def run_test_cases():
    print("\n" + "=" * 70)
    print("  ENTRADAS DE PRUEBA")
    print("=" * 70)
    print(f"{'Descripción':<25} {'n':>5} {'DP valor':>10} {'Greedy valor':>13} {'Calidad':>9}")
    print("-" * 70)
    for n, prices, desc in TEST_CASES:
        p = prices[:n+1]
        val_dp, _ = dp_bottom_up(p, n)
        val_gr, _ = greedy_rod_cutting(p, n)
        q = (val_gr / val_dp * 100) if val_dp > 0 else 100.0
        print(f"{desc:<25} {n:>5} {val_dp:>10} {val_gr:>13} {q:>8.1f}%")
    print("=" * 70)


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    demo()
    run_test_cases()
    sizes, dp_times, gr_times, qualities = empirical_analysis()
    plot_results(sizes, dp_times, gr_times, qualities)
