import random
import time
import csv
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

# ALGORITMOS

def dp_coin_game(coins: list[int]) -> tuple[int, list[list[int]]]:
    """
    DP para Optimal Game Strategy

    Recurrencia:
        dp[i][j] = max(
            coins[i] + min(dp[i+2][j], dp[i+1][j-1]),
            coins[j] + min(dp[i+1][j-1], dp[i][j-2])
        )

    Casos base:
        dp[i][i]   = coins[i]
        dp[i][i+1] = max(coins[i], coins[i+1])

    coins : lista de valores de las monedas.

    Retorna
    -------
    (ganancia_optima, tabla_dp)
    """
    n = len(coins)
    if n == 0:
        return 0, []

    # tabla n x n
    dp = [[0] * n for _ in range(n)]

    # base de 1
    for i in range(n):
        dp[i][i] = coins[i]

    # base de 2
    for i in range(n - 1):
        dp[i][i + 1] = max(coins[i], coins[i + 1])

    # intervalos mayores
    for length in range(3, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1

            # toma izquierda
            left_val = dp[i + 2][j] if i + 2 <= j else 0
            left_val = min(left_val, dp[i + 1][j - 1] if i + 1 <= j - 1 else 0)
            pick_left = coins[i] + left_val

            # toma derecha
            right_val = dp[i + 1][j - 1] if i + 1 <= j - 1 else 0
            right_val = min(right_val, dp[i][j - 2] if i <= j - 2 else 0)
            pick_right = coins[j] + right_val

            dp[i][j] = max(pick_left, pick_right)

    return dp[0][n - 1], dp


def greedy_coin_game(coins: list[int]) -> tuple[int, list[int]]:
    """
    Greedy para Optimal Game Strategy.

    Toma el extremo mayor en cada turno.

    coins : lista de valores de las monedas.

    Retorna
    -------
    (ganancia_greedy, historial_de_picks)
    """
    left, right = 0, len(coins) - 1
    player1_score = 0
    player2_score = 0
    turn = 1
    picks = []

    while left <= right:
        if coins[left] >= coins[right]:
            chosen = coins[left]
            picks.append((turn, 'L', chosen))
            left += 1
        else:
            chosen = coins[right]
            picks.append((turn, 'R', chosen))
            right -= 1

        if turn == 1:
            player1_score += chosen
            turn = 2
        else:
            player2_score += chosen
            turn = 1

    return player1_score, picks


def greedy_score_vs_optimal_opponent(coins: list[int]) -> int:
    """
    Puntaje de J1 con greedy y respuesta optima de J2

    Se usa para medir calidad contra DP
    """
    n = len(coins)
    if n == 0:
        return 0

    # g1 con turno de J1
    # g2 con turno de J2
    g1 = [[0] * n for _ in range(n)]
    g2 = [[0] * n for _ in range(n)]

    # base
    for i in range(n):
        g1[i][i] = coins[i]
        g2[i][i] = 0

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1

            # turno J1
            if coins[i] >= coins[j]:
                g1[i][j] = coins[i] + g2[i + 1][j]
            else:
                g1[i][j] = coins[j] + g2[i][j - 1]

            # turno J2
            g2[i][j] = min(g1[i + 1][j], g1[i][j - 1])

    return g1[0][n - 1]

# UTILIDADES

def quality_percentage(greedy_score: int, dp_score: int) -> float:
    """Calidad = greedy sobre optimo por 100"""
    if dp_score == 0:
        return 100.0
    return (greedy_score / dp_score) * 100.0


def timed_dp(coins):
    """DP y tiempo"""
    start = time.perf_counter()
    score, _ = dp_coin_game(coins)
    elapsed = time.perf_counter() - start
    return score, elapsed


def timed_greedy(coins):
    """Greedy y tiempo"""
    start = time.perf_counter()
    score, _ = greedy_coin_game(coins)
    elapsed = time.perf_counter() - start
    return score, elapsed


def generate_random_coins(n: int, min_val: int = 1, max_val: int = 100) -> list[int]:
    """Genera monedas aleatorias"""
    return [random.randint(min_val, max_val) for _ in range(n)]


# CASOS DE PRUEBA MANUALES

def run_manual_tests():
    """
    Ejecuta pruebas manuales
    """
    test_cases = [
        {
            "name": "Contraejemplo clasico greedy falla",
            "coins": [8, 15, 3, 7],
            "expected_dp": 22,
            "note": "DP toma 7 y luego 15 = 22. Greedy toma 8 y luego 7 = 15"
        },
        {
            "name": "Greedy falla visible",
            "coins": [2, 3, 15, 7],
            "expected_dp": 17,
            "note": "DP garantiza 17. Greedy toma 7 y queda con 10"
        },
        {
            "name": "Caso clasico 4 monedas",
            "coins": [20, 30, 2, 2],
            "expected_dp": 32,
            "note": "DP toma 20 y luego 2 o 30 y 2 = 32"
        },
        {
            "name": "Contraejemplo del proyecto",
            "coins": [3, 9, 1, 2],
            "expected_dp": 11,
            "note": "DP toma 2 y luego 9 = 11. Greedy toma 3 y queda 5"
        },
        {
            "name": "Monedas iguales",
            "coins": [5, 5, 5, 5],
            "expected_dp": 10,
            "note": "Todos iguales y ambos obtienen 10"
        },
        {
            "name": "Una sola moneda",
            "coins": [42],
            "expected_dp": 42,
            "note": "J1 toma la unica moneda"
        },
        {
            "name": "Dos monedas",
            "coins": [10, 20],
            "expected_dp": 20,
            "note": "J1 toma la mayor 20"
        },
        {
            "name": "Caso grande manual",
            "coins": [6, 9, 1, 2, 16, 8],
            "expected_dp": None,    # calculado por DP
            "note": "6 monedas y greedy difiere de DP"
        },
    ]

    print("=" * 70)
    print("  CASOS DE PRUEBA MANUALES")
    print("=" * 70)

    all_passed = True
    for tc in test_cases:
        coins = tc["coins"]
        dp_score, _ = dp_coin_game(coins)
        greedy_fast_score, picks = greedy_coin_game(coins)
        greedy_score = greedy_score_vs_optimal_opponent(coins)
        quality = quality_percentage(greedy_score, dp_score)

        passed = (tc["expected_dp"] is None or dp_score == tc["expected_dp"])
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False

        print(f"\n  [{status}] {tc['name']}")
        print(f"    Monedas   : {coins}")
        print(f"    DP        : {dp_score}   (esperado: {tc['expected_dp']})")
        print(f"    Greedy*   : {greedy_score}   (calidad: {quality:.1f}%)")
        print(f"    Greedy rapido ambos greedy: {greedy_fast_score}")
        print(f"    Nota      : {tc['note']}")
        # Mostrar picks del jugador 1
        p1_picks = [f"{v}({d})" for (p, d, v) in picks if p == 1]
        print(f"    Picks J1  : {p1_picks}")

    print()
    print("  Resultado global:", "TODOS PASARON" if all_passed else "ALGUNOS FALLARON")
    print("=" * 70)
    print()



# ANALISIS EMPIRICO

# Tamaños de entrada
SIZES = [10, 20, 30, 40, 50, 100, 200, 300, 500, 750, 1000]
REPETITIONS = 5
RANDOM_SEED = 42

def run_empirical_analysis() -> list[dict]:
    """
    Genera datos aleatorios y mide tiempos

    Retorna lista de resultados
    """
    random.seed(RANDOM_SEED)
    results = []

    print("=" * 70)
    print("  ANÁLISIS EMPÍRICO")
    print(f"  Tamaños: {SIZES}")
    print(f"  Repeticiones por tamaño: {REPETITIONS}")
    print("=" * 70)
    print(f"  {'n':>6}  {'t_dp (s)':>12}  {'t_greedy (s)':>14}  "
          f"{'dp_score':>10}  {'greedy*':>10}  {'calidad%':>9}")
    print("  " + "-" * 71)

    for n in SIZES:
        dp_times, greedy_times, qualities = [], [], []
        dp_score_last = greedy_score_last = greedy_fast_last = 0

        for _ in range(REPETITIONS):
            coins = generate_random_coins(n)
            dp_score, t_dp = timed_dp(coins)
            greedy_fast_score, t_greedy = timed_greedy(coins)
            greedy_score = greedy_score_vs_optimal_opponent(coins)
            dp_times.append(t_dp)
            greedy_times.append(t_greedy)
            qualities.append(quality_percentage(greedy_score, dp_score))
            dp_score_last = dp_score
            greedy_score_last = greedy_score
            greedy_fast_last = greedy_fast_score

        avg_dp = sum(dp_times) / REPETITIONS
        avg_greedy = sum(greedy_times) / REPETITIONS
        avg_quality = sum(qualities) / REPETITIONS

        row = {
            "n": n,
            "tiempo_dp": avg_dp,
            "tiempo_greedy": avg_greedy,
            "ganancia_dp": dp_score_last,
            "ganancia_greedy": greedy_score_last,
            "ganancia_greedy_fast": greedy_fast_last,
            "calidad_porcentaje": avg_quality,
        }
        results.append(row)

        print(f"  {n:>6}  {avg_dp:>12.6f}  {avg_greedy:>14.8f}  "
              f"{dp_score_last:>10}  {greedy_score_last:>13}  {avg_quality:>8.2f}%")

    print("=" * 70)
    print()
    return results


# EXPORTAR CSV

def export_csv(results: list[dict], filepath: str = "resultados.csv"):
    fieldnames = ["n", "tiempo_dp", "tiempo_greedy",
                  "ganancia_dp", "ganancia_greedy", "ganancia_greedy_fast",
                  "calidad_porcentaje"]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"  [CSV] Resultados exportados → {filepath}")


# GRAFICAS

COLORS = {
    "dp":     "#2563EB",   # azul
    "greedy": "#DC2626",   # rojo
    "poly_dp":     "#60A5FA",
    "poly_greedy": "#FCA5A5",
    "quality":     "#16A34A",
}

def _poly_label(coeffs, degree):
    """Formatea la regresion polinomial"""
    terms = []
    for i, c in enumerate(coeffs):
        power = degree - i
        if abs(c) < 1e-15:
            continue
        c_str = f"{c:.4e}"
        if power == 0:
            terms.append(c_str)
        elif power == 1:
            terms.append(f"{c_str}·n")
        else:
            terms.append(f"{c_str}·n^{power}")
    return " + ".join(terms) if terms else "0"



def plot_results(results: list[dict], output_dir: str = "."):
    ns        = np.array([r["n"] for r in results])
    t_dp      = np.array([r["tiempo_dp"]      for r in results])
    t_greedy  = np.array([r["tiempo_greedy"]  for r in results])
    quality   = np.array([r["calidad_porcentaje"] for r in results])

    # grado de ajuste
    deg_dp     = 2
    deg_greedy = 1

    coeffs_dp     = np.polyfit(ns, t_dp,     deg_dp)
    coeffs_greedy = np.polyfit(ns, t_greedy, deg_greedy)

    ns_smooth   = np.linspace(ns.min(), ns.max(), 500)
    fit_dp      = np.polyval(coeffs_dp,     ns_smooth)
    fit_greedy  = np.polyval(coeffs_greedy, ns_smooth)

    # R2
    def r2(y, y_pred):
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot != 0 else 1.0

    r2_dp     = r2(t_dp,     np.polyval(coeffs_dp,     ns))
    r2_greedy = r2(t_greedy, np.polyval(coeffs_greedy, ns))

    # figura 1
    fig1, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig1.suptitle(
        "Optimal Game Strategy — Tiempo de Ejecución vs Tamaño de Entrada",
        fontsize=14, fontweight="bold", y=1.01
    )

    for ax, t_data, color, fit, coeffs, deg, r2_val, algo in [
        (axes[0], t_dp,     COLORS["dp"],     fit_dp,     coeffs_dp,     deg_dp,
         r2_dp,     "Programación Dinámica"),
        (axes[1], t_greedy, COLORS["greedy"], fit_greedy, coeffs_greedy, deg_greedy,
         r2_greedy, "Greedy"),
    ]:
        ax.scatter(ns, t_data, color=color, zorder=5, s=60, label="Datos empíricos")
        ax.plot(ns_smooth, fit, color=color, linewidth=2, linestyle="--",
                label=f"Regresión grado {deg}\n(R²={r2_val:.4f})")
        ax.set_title(algo, fontsize=12)
        ax.set_xlabel("n (tamaño de entrada)", fontsize=11)
        ax.set_ylabel("Tiempo promedio (segundos)", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # ecuacion
        eq = _poly_label(coeffs, deg)
        ax.annotate(f"y ≈ {eq}", xy=(0.03, 0.92), xycoords="axes fraction",
                    fontsize=7.5, color=color,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    fig1.tight_layout()
    path1 = os.path.join(output_dir, "fig1_tiempos_separados.png")
    fig1.savefig(path1, dpi=150, bbox_inches="tight")
    print(f"  [PLOT] {path1}")

    # figura 2
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.scatter(ns, t_dp,     color=COLORS["dp"],     s=60, zorder=5, label="DP (datos)")
    ax2.scatter(ns, t_greedy, color=COLORS["greedy"], s=60, zorder=5, label="Greedy (datos)")
    ax2.plot(ns_smooth, fit_dp,     color=COLORS["poly_dp"],
             linewidth=2, linestyle="--", label=f"DP regresión grado {deg_dp}")
    ax2.plot(ns_smooth, fit_greedy, color=COLORS["poly_greedy"],
             linewidth=2, linestyle="--", label=f"Greedy regresión grado {deg_greedy}")
    ax2.set_yscale("log")
    ax2.set_title("Comparación DP vs Greedy — Escala logarítmica", fontsize=13, fontweight="bold")
    ax2.set_xlabel("n (tamaño de entrada)", fontsize=11)
    ax2.set_ylabel("Tiempo promedio (segundos) — log", fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, which="both", linestyle="--", alpha=0.4)
    fig2.tight_layout()
    path2 = os.path.join(output_dir, "fig2_comparacion_log.png")
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    print(f"  [PLOT] {path2}")

    # figura 3
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.plot(ns, quality, color=COLORS["quality"], marker="o",
             linewidth=2, markersize=7, label="Calidad greedy (%)")
    ax3.axhline(100, color="gray", linestyle="--", linewidth=1, label="Óptimo (100%)")
    ax3.fill_between(ns, quality, 100,
                     where=(quality < 100), interpolate=True,
                     color=COLORS["greedy"], alpha=0.15, label="Pérdida vs óptimo")
    ax3.set_ylim(max(0, quality.min() - 5), 105)
    ax3.set_title("Calidad de Solución Greedy vs Óptimo (DP)", fontsize=13, fontweight="bold")
    ax3.set_xlabel("n (tamaño de entrada)", fontsize=11)
    ax3.set_ylabel("Calidad  (ganancia_greedy / ganancia_dp) × 100", fontsize=11)
    ax3.legend(fontsize=10)
    ax3.grid(True, linestyle="--", alpha=0.5)

    # estadisticas
    stats_text = (f"Media: {quality.mean():.2f}%\n"
                  f"Mín:   {quality.min():.2f}%\n"
                  f"Máx:   {quality.max():.2f}%")
    ax3.text(0.98, 0.05, stats_text, transform=ax3.transAxes,
             fontsize=9, va="bottom", ha="right",
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.8))

    fig3.tight_layout()
    path3 = os.path.join(output_dir, "fig3_calidad_greedy.png")
    fig3.savefig(path3, dpi=150, bbox_inches="tight")
    print(f"  [PLOT] {path3}")

    # figura 4
    speedup = t_dp / np.maximum(t_greedy, 1e-12)

    fig4, ax4 = plt.subplots(figsize=(10, 5))
    ax4.bar(ns, speedup, color=COLORS["dp"], alpha=0.7, width=np.diff(ns, append=ns[-1]+50)*0.6)
    ax4.set_title("Factor de aceleración: DP / Greedy", fontsize=13, fontweight="bold")
    ax4.set_xlabel("n (tamaño de entrada)", fontsize=11)
    ax4.set_ylabel("Speedup (veces más rápido)", fontsize=11)
    ax4.grid(True, axis="y", linestyle="--", alpha=0.5)

    for x, y in zip(ns, speedup):
        ax4.text(x, y + speedup.max() * 0.01, f"{y:.1f}×",
                 ha="center", va="bottom", fontsize=8)

    fig4.tight_layout()
    path4 = os.path.join(output_dir, "fig4_speedup.png")
    fig4.savefig(path4, dpi=150, bbox_inches="tight")
    print(f"  [PLOT] {path4}")

    plt.close("all")

    # resumen
    print()
    print("  REGRESIONES POLINOMIALES")
    print(f"  DP  (grado {deg_dp}): R² = {r2_dp:.6f}")
    print(f"    Ecuación: y ≈ {_poly_label(coeffs_dp, deg_dp)}")
    print(f"  Greedy (grado {deg_greedy}): R² = {r2_greedy:.6f}")
    print(f"    Ecuación: y ≈ {_poly_label(coeffs_greedy, deg_greedy)}")
    print()



