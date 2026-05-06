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